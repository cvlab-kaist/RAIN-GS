from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    
    try:
        makedirs(folder_path)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
