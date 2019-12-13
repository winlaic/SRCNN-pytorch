from scipy.io import loadmat
import torch
from model import SRCNN
import numpy as np


# You can load the model in matlab format provided by the paper authors by this function.
def load_mat_model(mat_file, net: SRCNN):
    raw_model = loadmat(mat_file)
    with torch.no_grad():
        net.feature.weight.data = torch.tensor(raw_model['weights_conv1'].reshape(9, 9, 64, 1).transpose(2, 3, 1, 0), dtype=torch.float32)
        net.feature.bias.data = torch.tensor(raw_model['biases_conv1'].squeeze(-1), dtype=torch.float32)
        net.remap.weight.data = torch.tensor(raw_model['weights_conv2'].transpose(2, 0, 1)[..., None], dtype=torch.float32)
        net.remap.bias.data = torch.tensor(raw_model['biases_conv2'].squeeze(-1), dtype=torch.float32)
        net.reconstruct.weight.data = torch.tensor(raw_model['weights_conv3'].reshape(1, 32, 5, 5).transpose(0, 1, 3, 2), dtype=torch.float32)
        net.reconstruct.bias.data = torch.tensor(raw_model['biases_conv3'].squeeze(-1), dtype=torch.float32)
    pass

def modcrop(img, mod):
    img = np.array(img)
    img_size = np.array(img.shape[0:2])
    new_size = img_size - img_size % mod
    new_img = img[0:new_size[0], 0:new_size[1], ...]
    return new_img