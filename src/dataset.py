import torch
import os
from winlaic.numpy.image import extract_patches
import numpy as np
from os.path import join
from PIL import Image
import h5py
from winlaic.fs import listimg
from torchvision.transforms.functional import to_tensor
import re
from utils import modcrop

class NPZDataset(torch.utils.data.Dataset):
    def __init__(self, data_pack):
        data_pack = np.load(data_pack)
        self.input = data_pack['input']
        self.label = data_pack['label']

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        input, label = map(lambda x: torch.tensor(x, dtype=torch.float32).permute(2, 0, 1) / 255.0, (self.input[index], self.label[index]))
        return input, label

    def __len__(self):
        return self.input.shape[0]

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_file):
        file = h5py.File(h5_file)
        self.data = np.array(file['data'])
        self.label = np.array(file['label'])

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]
        data, label = map(lambda x: torch.tensor(x, dtype=torch.float32), (data, label))
        return data, label

    def __len__(self):
        return self.data.shape[0]
        

class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, scale_factor, colored=False):
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.colored = colored
        self.images = listimg(self.image_path)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_hr = Image.open(join(self.image_path, self.images[index]))
        if not self.colored:
            img_hr = img_hr.convert('L')
        img_hr = Image.fromarray(modcrop(img_hr, self.scale_factor))
        img_lr = img_hr.resize(tuple(item // self.scale_factor for item in img_hr.size), Image.BICUBIC)
        img_lr = img_lr.resize(img_hr.size, Image.BICUBIC)
        img_lr, img_hr = map(to_tensor, (img_lr, img_hr))
        return img_lr, img_hr



