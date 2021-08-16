'''
Script to inference a dataset through a trained generator network
'''

# Import python libraries
import os, shutil, sys

print(sys.argv)

# Specify directories
# imagesdir = 'data/dataset/images/'
imagesdir = sys.argv[1]
cyclegandir = 'pytorch-CycleGAN-and-pix2pix/'
datasetsdir = cyclegandir + 'datasets/test/'
checkpointsdir = cyclegandir + 'checkpoints/test/'
# modeldir = 'data/model/'
modeldir = sys.argv[2]
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
    # shutil.copy(modeldir + file, checkpointsdir + sys.argv[1])

# Transfer the updated networks.py file
shutil.copy(srcdir + 'networks.py', cyclegandir + 'models/')

# Change directories to inference dataset through generator
os.chdir('pytorch-CycleGAN-and-pix2pix')
os.system('python test.py --dataroot datasets/test/testA --name test --model test --no_dropout --netG resnet_3blocks --load_size 512 --crop_size 512')

# Copy original dataset to new folder and create new directory for gan-inferenced images
shutil.copytree('../data/dataset/', '../data/gandataset/')
os.mkdir('../data/gandataset/ganimages/')

# Copy gan-inferenced dataset to new folder
for ifile, filename in enumerate(sorted(os.listdir('results/test/test_latest/images'))):
    if 'fake' in filename:
        print('filename = {}'.format(filename))
        filebase = filename.split('_')[0]
        shutil.copy('results/test/test_latest/images/'+filename,'../data/gandataset/ganimages/'+filebase+'.png')


# shutil.copytree('results/test/test_latest/images', '../data/gandataset/ganimages/')