import cv2 as cv
import numpy as np
import argparse
from os.path import join
from winlaic.fs import listimg
from winlaic.numpy.image import extract_patches
from tqdm import tqdm
from utils import modcrop

imread = lambda img: cv.cvtColor(cv.imread(img), cv.COLOR_BGR2RGB)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str)
    parser.add_argument('target_file', type=str)
    parser.add_argument('-s', '--scale-factor', type=int, default=3, required=False)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--patch-size', type=int, default=33)
    parser.add_argument('--label-size', type=int, default=21)
    parser.add_argument('-c', '--colored', action='store_false', default=True)
    parser.add_argument('--no-shuffle', action='store_true', default=False)
    params = parser.parse_args()
    print('Warning: It seems that Opencv has a different interpolation implementation when specifying cubic, ')
    print('which will cause the network notworking.')
    print('So please consider using PIL version.')
    margin = (params.patch_size - params.label_size) // 2

    imgs = listimg(params.image_dir)
    input = []; label = []
    for img_name in tqdm(imgs):
        img = imread(join(params.image_dir, img_name))
        if params.colored:
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = modcrop(img, params.scale_factor)
        img_size = img.size
        img_lr = cv.resize(img, tuple(item // params.scale_factor for item in img_size)[::-1], interpolation=cv.INTER_CUBIC)
        img_lr = cv.resize(img_lr, tuple(item for item in img_size)[::-1], interpolation=cv.INTER_CUBIC)
        img = extract_patches(img, params.patch_size, strides=params.stride)
        img = img.reshape(-1, *img.shape[2:])[:, margin:-margin, margin:-margin, :]
        img_lr = extract_patches(img_lr, params.patch_size, strides=params.stride)
        img_lr = img_lr.reshape(-1, *img_lr.shape[2:])
        input.append(img_lr); label.append(img)

    input = np.concatenate(input, axis=0); label = np.concatenate(label, axis=0)
    if not params.no_shuffle:
        random_index = np.random.choice(input.shape[0], input.shape[0], replace=False)
        input = input[random_index, ...]
        label = label[random_index, ...]
    np.savez(params.target_file, input=input, label=label)
    print('Done.')

