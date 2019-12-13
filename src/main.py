import argparse
import os
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tensorboardX import SummaryWriter
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from dataset import H5Dataset, NPZDataset, ValidationDataset
from model import SRCNN
from utils import load_mat_model
from winlaic.fs import ensuredir, listimg
from winlaic.nn import (Saver, Trainer, adjust_hyperparams, general_initialize,
                        get_avaliable_devices, start_tensorboard)
from winlaic.utils import Logger, print_params, time_stamp


class SRTrainer(Trainer):
    loss_function = nn.MSELoss()
    def criterion(self, net, seq):
        lr, hr = seq
        lr, hr = lr.cuda(), hr.cuda()
        sr = net(lr)
        loss = self.loss_function(sr, hr)
        return loss

def _test(params):
    net = SRCNN(f1=params.f1, f2=params.f2, f3=params.f3, n1=params.n1, n2=params.n2, c=params.c)

    # load_mat_model('models/Ori_915_91img.mat', net)
    checkpoint = torch.load(params.test_model)
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    output_dir = ensuredir(params.output_dir)
    for img_name in tqdm(listimg(params.test_image_path)):
        img = Image.open(join(params.test_image_path, img_name))
        output_size = tuple(item * params.scale_factor for item in img.size)
        img = img.resize(output_size, Image.BICUBIC)
        img = np.array(img.convert('YCbCr'))
        img_y, img_cb, img_cr = img[..., 0], img[..., 1], img[..., 2]
        img_y = to_tensor(img_y).cuda()[None, ...]
        img_y_sr = net(img_y)
        real_output_size = tuple(item for item in img_y_sr.shape[-2:])[::-1]
        margin = (output_size[0] - real_output_size[0]) // 2
        img_y_sr = img_y_sr.detach().cpu().numpy().clip(0.0, 1.0)
        img_y_sr = (img_y_sr * 255.0).astype(np.uint8)[0, ...].transpose(1, 2, 0).squeeze()
        if params.colored:
            img_cb_sr, img_cr_sr = map(lambda x: np.array(Image.fromarray(x).resize(output_size, Image.BICUBIC))[margin:-margin, margin:-margin],\
                (img_cb, img_cr))
            img_sr = Image.fromarray(np.stack([img_y_sr, img_cb_sr, img_cr_sr]).transpose(1, 2, 0), mode='YCbCr')
            img_sr = img_sr.convert('RGB')
        else:
            img_sr = Image.fromarray(img_y_sr)
        img_sr.save(join(output_dir, img_name))
        

test_hr_added = False
def test(params, net, dataset):
    global logger, tb, devices, test_hr_added
    psnrs = []
    margin = net.margin
    net.eval()
    with torch.no_grad():
        for i, (lr, hr) in enumerate(dataset):
            lr, hr = map(lambda x: x.unsqueeze(0).cuda(), (lr, hr))
            sr = net(lr)
            sr = sr.clamp(0.0, 1.0)

            if not test_hr_added:
                tb.add_images(join('validate', str(i), 'hr'), hr[..., margin:-margin, margin:-margin])
            tb.add_images(join('validate', str(i), 'sr'), sr)

            hr = hr.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            sr = sr.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            psnr = peak_signal_noise_ratio(hr[margin:-margin, margin:-margin, ...], sr)
            psnrs.append(psnr)
    test_hr_added = True
    return {
        'PSNR': np.mean(np.array(psnrs))
    }



def train(params):
    global logger, tb, devices, TIME_STAMP
    net = SRCNN(f1=params.f1, f2=params.f2, f3=params.f3, n1=params.n1, n2=params.n2, c=params.c)
    net.to(devices[0])

    param_groups = [
            {'params': net.feature.parameters(), 'lr': params.feature_lr}, 
            {'params': net.remap.parameters(),   'lr': params.remap_lr}, 
            {'params': net.reconstruct.parameters(), 'lr': params.reconstruct_lr}, 
        ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
    optimizer.param_groups = adjust_hyperparams(optimizer.param_groups, net, bias_lr_mult=0.1)
    optimizer.param_groups[-1]['lr'] = 1e-5

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            nn.init.constant_(m.bias, 0.0)

    validation_set = ValidationDataset(params.validate_images_path, scale_factor=params.scale_factor, colored=params.colored)
    dataset = NPZDataset('Waterloo_ref.npz')
    # dataset = H5Dataset(params.h5_data_pack)
    
    dataloader = torch.utils.data.DataLoader(dataset, 
        num_workers=params.num_workers, pin_memory=True, batch_size=params.batch_size,
        shuffle=True, drop_last=True)

    trainer = SRTrainer(net, optimizer, dataloader, using_tqdm=True)
    saver = Saver(trainer, TIME_STAMP, logger=logger, period_checkpoint=params.period_checkpoint)

    print(test(params, net, validation_set))
    for loss in trainer.epoch(int(10*10**8/len(dataset))):
        tb.add_scalar(join('train', 'loss'), loss, trainer.progress)
        saver.save_checkpoint()
        if trainer.progress % params.period_validate == 0:
            result = test(params, net, validation_set)
            for key in result:
                logger.i = 'EPOCH', trainer.progress, key, result[key]
                tb.add_scalar(join('val', key), result[key], trainer.progress)
            saver.save_maximum(result['PSNR'], 'PSNR')



if __name__ == "__main__":
    TIME_STAMP = time_stamp()
    devices = get_avaliable_devices()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])

    train_test = parser.add_mutually_exclusive_group()

    test_args = train_test.add_argument_group()
    test_args.add_argument('--test-model', type=str)
    test_args.add_argument('--test-image-path', type=str)
    test_args.add_argument('--output-dir', type=str)
    test_args.add_argument('--colored', action='store_true', default=False)

    train_args = train_test.add_argument_group()
    train_args.add_argument('--data-pack', type=str)
    train_args.add_argument('--validate-images-path', type=str, default='test_sets/Set5')
    train_args.add_argument('--period-checkpoint', type=int, default=50)
    train_args.add_argument('--period-validate', type=int, default=1)
    train_args.add_argument('-j', '--num-workers', type=int, default=8)
    train_args.add_argument('--batch-size', type=int, default=128)
    train_args.add_argument('--feature-lr', type=float, default=1e-4)
    train_args.add_argument('--remap-lr', type=float, default=1e-4)
    train_args.add_argument('--reconstruct-lr', type=float, default=1e-5)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--f1', type=int, default=9)
    parser.add_argument('--f2', type=int, default=1)
    parser.add_argument('--f3', type=int, default=5)
    parser.add_argument('--n1', type=int, default=64)
    parser.add_argument('--n2', type=int, default=32)
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--scale-factor', type=int, default=3)
    
    params = parser.parse_args()
    
    logger = Logger(write_to_file=not params.debug)
    tensorboard_dir = join('tensorboard', TIME_STAMP)
    tb = SummaryWriter(tensorboard_dir)
    start_tensorboard(tensorboard_dir, tensorboard='/home/cel/.local/bin/tensorboard')
    print_params(params, logger)

    if params.mode == 'train':
        train(params)
    elif params.mode == 'test':
        _test(params)
