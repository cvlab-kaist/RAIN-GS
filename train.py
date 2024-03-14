import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import numpy as np
import wandb
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    

def training(dataset, opt, pipe, testing_iterations ,saving_iterations, checkpoint_iterations ,checkpoint, debug_from, args_dict):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, args_dict['output_path'], args_dict['exp_name'], args_dict['project_name'])
    
    if args_dict['ours']:
        divide_ratio = 0.7
    else:
        divide_ratio = 0.8
    print(f"Set divide_ratio to {divide_ratio}")
    
    gaussians = GaussianModel(dataset.sh_degree, divide_ratio)
    scene = Scene(dataset, gaussians, args_dict=args_dict)
    gaussians.training_setup(opt) 
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        
        if args_dict['DSV']:
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
        elif args_dict['ours']:
            if iteration >= 5000:
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()
        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        c2f = args_dict['c2f']

        if c2f == True:
            if iteration == 1 or (iteration % args_dict['c2f_every_step'] == 0 and iteration < opt.densify_until_iter) :
                H = viewpoint_cam.image_height
                W = viewpoint_cam.image_width
                N = gaussians.get_xyz.shape[0]
                low_pass = max (H * W / N / (9 * np.pi), 0.3)
                if args_dict['c2f_max_lowpass'] > 0:
                    low_pass = min(low_pass, args_dict['c2f_max_lowpass'])
                print(f"[ITER {iteration}] Low pass filter : {low_pass}")
        else:
            low_pass = 0.3
            
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, low_pass = low_pass)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "num_gaussians" : f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:       
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)         
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                

def prepare_output_and_logger(args, output_path, exp_name, project_name):
    if (not args.model_path) and (not exp_name):
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    elif (not args.model_path) and exp_name:
        args.model_path = os.path.join("./output", exp_name) 
    
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, 'command_line.txt'), 'w') as file:
        file.write(' '.join(sys.argv))

    tb_writer = None   
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        print("Logging progress to Tensorboard at {}".format(args.model_path))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                    ssim_test += ssim(image, gt_image)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS(vgg) {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test, ssim_test))
                with open(os.path.join(args.output_path, args.exp_name, 'log_file.txt'), 'a') as file:
                    file.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS(vgg) {} SSIM {}\n".format(iteration, config['name'], l1_test, psnr_test, lpips_test, ssim_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--output_path", type=str,default='./output/')
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="gaussian-splatting")
    parser.add_argument("--c2f", action="store_true", default=False)
    parser.add_argument("--c2f_every_step", type=float, default=1000, help="Recompute low pass filter size for every c2f_every_step iterations")
    parser.add_argument("--c2f_max_lowpass", type=float, default= 300, help="Maximum low pass filter size")
    parser.add_argument("--num_gaussians", type=int, default=1000000, help="Number of random initial gaussians to start with (default=1M for DSV)")
    parser.add_argument('--DSV', action='store_true', help="Use the initialisation from the paper")
    parser.add_argument("--ours", action="store_true", help="Use our initialisation")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.white_background = args.white_bg
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    
    if args.ours:
        print("========= USING OUR INITIALISATION =========")
        args.c2f = True
        args.c2f_every_step = 1000
        args.c2f_max_lowpass = 300
        args.eval = True
        args.num_gaussians = 10

    if not args.DSV and not args.ours:
        parser.error("Please specify either --DSV or --ours")
    print(f"args: {args}")
    
    while True :
        try:
            network_gui.init(args.ip, args.port)
            print(f"GUI server started at {args.ip}:{args.port}")
            break
        except Exception as e:
            args.port = args.port + 1
            print(f"Failed to start GUI server, retrying with port {args.port}...")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations ,args.save_iterations, args.checkpoint_iterations ,args.start_checkpoint, args.debug_from, args.__dict__)

    
    print("\nTraining complete.")