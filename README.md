Forked from https://github.com/junyanz/pytorch-cyclegan-and-pix2pix

Required libraries:
torch torchvision numpy scikit-image dominate pillow

Model names:

DOE Removal remove_8

Spectral Classification pix_3

The image pre-processing script is im_preprocess.py. This is a marimo notebook, and probably won't work with your pycharm, but if you want to pre-process your images for feeding into the model you can look into the code to see how it's done. I'll also summarize here:
1. Take your diffractogram and do a global dataset normalization with min = 0 and max = 2^(16)-1.
2. save the image to a folder called val/

To run single images through a model with the test.py script, the arguments in the command line are as follows:
run test.py --dataroot /path/to/val/folder --name <which model you want to use> --model test --dataset_mode single --netG unet_256 --norm batch --nc_input 1 --nc_output <1 for doe removal, 3 for spectral classification> --num_test <maximum number of pictures to test>

By default, this will save images in the results/<model name>/test_latest/ folder. That SHOULD get you what you need. 
