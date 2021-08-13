'''
Script to inference a dataset through a trained generator network
'''

# Import python libraries
import os, shutil

# Specify directories
imagesdir = '../../data/dataset/images/'
cyclegandir = '../../pytorch-CycleGAN-and-pix2pix/'
datasetsdir = cyclegandir + 'datasets/test/'
checkpointsdir = cyclegandir + 'checkpoints/test/'
modeldir = '../../model/'
srcdir = ''

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
