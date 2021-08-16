'''
Script to inference a dataset through a trained generator network
'''

# Import python libraries
import os, shutil, subprocess

# Specify directories
imagesdir = 'data/dataset/images/'
cyclegandir = 'pytorch-CycleGAN-and-pix2pix/'
datasetsdir = cyclegandir + 'datasets/test/'
checkpointsdir = cyclegandir + 'checkpoints/test/'
modeldir = 'data/model/'
srcdir = 'DomainAdaptation/src/'

# Transfer input dataset to test set
print(os.getcwd())
os.makedirs(datasetsdir + 'testA/')
for file in sorted(os.listdir(imagesdir)):
    shutil.copy(imagesdir + file, datasetsdir + 'testA/')

# Transfer model to checkpoints directory
os.makedirs(checkpointsdir)
for file in sorted(os.listdir(modeldir)):
    shutil.copy(modeldir + file, checkpointsdir + 'latest_net_G.pth')

# Transfer the updated networks.py file
shutil.copy(srcdir + 'networks.py', cyclegandir + 'models/')

# Change directories to inference dataset through generator
os.chdir('pytorch-CycleGAN-and-pix2pix')
os.system('python test.py --dataroot datasets/test/testA --name test --model test --no_dropout --netG resnet_3blocks')

# Copy original dataset to new folder and create new directory for gan-inferenced images
shutil.copytree('../data/dataset/', '../data/gandataset/')
os.mkdir('../data/gandataset/ganimages/')

# Copy gan-inferenced dataset to new folder
for ifile, filename in enumerate(sorted(os.listdir('results/test/test_latest/images'))):
    if 'fake' in filename:
        filebase = filename.split('_')[0]
        shutil.copy('results/test/test_latest/images/'+filename,'../data/gandataset/ganimages/'+filebase+'.png')


# shutil.copytree('results/test/test_latest/images', '../data/gandataset/ganimages/')