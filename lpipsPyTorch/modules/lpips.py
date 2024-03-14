import torch
import torch.nn as nn

from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        
        self.net = get_network(net_type)

        
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0), 0, True)
