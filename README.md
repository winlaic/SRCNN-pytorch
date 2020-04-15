# SRCNN-Pytorch

Implementation of SRCNN referred in *Image Super-Resolution Using Deep Convolutional Networks*. 

Original implementation in Caffe can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html).



## Requirements

pytorch>=1.3.0

numpy>=1.17.4

tqdm

tensorboardX

Pillow==6.2.1

skimage==0.16.2

winlaic (which can be found in my repositories, use version in https://github.com/winlaic/winlaic/commit/2facccc5bc51894e40589f53a40e93fbbb4be502)



## Useage

### Train

Please follow these steps to train the model.

1. **Generate the data package.**

   As instructed by the authors, images are clipped into 33*33 patches with stride of 14.

   We followed this and save the patches in a npz file.

   You can accumulate images in a directory and use

    `src/generate_datapack_PIL.py [/path/to/images] [datapack_name]` to generate the data package.

   By default, the scale factor is 3. You can specify scale factor by appending `--scale-factor [factor]`.

   Only the gray scale image is saved by default. If you would like to save in RGB format, add `--colored` argument at the end of command.

2. **Train the model.**

   Use `src/main.py train --data-pack [/path/to/data/package] --validate-image-path [/path/to/validate/images]` to start training. 

   The program will test the model on validate images and calculate PSNR. The best model will be saved.

   Remember to specify `--scale-factor` if you changed it in generating data package.

   AdamW optimizing method and Kaiming Initialization is exploited to accelerate training.

### Test

Use `src/main.py test --test-model [/path/to/trained/model] --test-image-path [/path/to/images/tobe/tested] --output-dir [/path/to/output/dir]` to generate super-resolution images. 

`--colored` can be specified to generate colored image. Cb, Cr channel are up scaled by bi-cubic interpolation. 



## Warning

It seems that opencv-python utilize a different bi-cubic interpolation method. 

Low-resolution images generated by `cv.resize(..., interpolation=cv.INTER_CUBIC)` cause model not working.
