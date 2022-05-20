# -*- coding: utf-8 -*-
"""Thesis_May8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fC5IYAlZoZvz8HEKLvDuK8KLpfKP4ayr
"""

#install TensorFlow 2.0 
#!pip install tensorflow==2.9.0

#best is acc = 0.42 for normalized cropped 100 no enhance


#https://learnopencv.com/efficient-image-loading/
#!pip uninstall pillow
#!pip install pillow-simd
import torch
if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')
#print("Device: {}".format(device))

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)


from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')

import PIL
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from matplotlib import image
import pandas as pd
import tensorflow as tf
plt.style.use('default')
filepath = '/content/drive/MyDrive/Thesis/Pics/'

from skimage import io, color, data
from skimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing, disk, remove_small_objects, square
from skimage.segmentation import clear_border, watershed, flood_fill, slic, mark_boundaries, chan_vese
from skimage.filters.rank import median
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import threshold_local
from skimage.morphology import dilation, erosion, opening, closing,reconstruction, extrema, binary_closing
from skimage.morphology import skeletonize
from skimage import exposure

from tqdm import tqdm
from pandas._libs import index
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import Dataset, DataLoader

from google.colab import drive
drive.mount('/content/drive')

#read file with classifications
df = pd.read_csv(filepath + 'gz_decals_volunteers_5.csv')

df = df.sort_values('iauname')

#only included observations with more than 5 votes and 'yes' over 50% and not 'wrong size'
conditions = [
    (df['has-spiral-arms_yes_fraction'] > 0.5) &
    (df['has-spiral-arms_yes'] >= 5) & (df['wrong_size_warning'] == False)
    ]

#label new column with 1 if satisfies all requirements
labels = [1]

#new column, Spiral_Valid
df['Spiral_Valid'] = np.select(conditions, labels)

#create column Arms which says which column has the highest votes
columns = [
 'spiral-arm-count_1_fraction',
 'spiral-arm-count_2_fraction',
 'spiral-arm-count_3_fraction',
 'spiral-arm-count_4_fraction',
 'spiral-arm-count_more-than-4_fraction']

df['Arms'] = df[columns].idxmax(axis=1)

#change labels to 0,1,2,3,4 and create new column Arm Count
conditions = [
    (df['Arms'] == 'spiral-arm-count_1_fraction'),
    (df['Arms'] == 'spiral-arm-count_2_fraction'),
    (df['Arms'] == 'spiral-arm-count_3_fraction'),
    (df['Arms'] == 'spiral-arm-count_4_fraction'),
    (df['Arms'] == 'spiral-arm-count_more-than-4_fraction'),
    ]

#label new column with 1 if satisfies all requirements
labels = [1,2,3,4,0]
    
df['Arm_Count'] = np.select(conditions, labels)

#only included observations where atleast the correct classification has over 50% of the votes
conditions = [
    (df['spiral-arm-count_1_fraction'] >= 0.5) |
    (df['spiral-arm-count_2_fraction'] >= 0.5) |
    (df['spiral-arm-count_3_fraction'] >= 0.5) |
    (df['spiral-arm-count_4_fraction'] >= 0.5) |
    (df['spiral-arm-count_more-than-4_fraction'] >= 0.5)
    ]

#label new column with 1 if satisfies all requirements
labels = [1]

#new column, Spiral_Valid
df['Spiral_Valid_New'] = np.select(conditions, labels)

#subset df into only valid spiral data and selected columns

df_new = df[df['Spiral_Valid'] == 1]
df_new = df_new[df_new['Spiral_Valid_New'] ==1]

columns = ['iauname', 
 'has-spiral-arms_yes',
 'has-spiral-arms_yes_fraction', 
 'spiral-arm-count_1_fraction',
 'spiral-arm-count_2_fraction',
 'spiral-arm-count_3_fraction',
 'spiral-arm-count_4_fraction',
 'spiral-arm-count_more-than-4_fraction',
 'Arms',
 'Arm_Count',
 'Spiral_Valid']

df_final = df_new[columns]

#reset indext
df_final = df_final.reset_index(drop = True)

df_final['Arm_Count'].unique()
values, counts = np.unique(df_final['Arm_Count'], return_counts=True)
print(values, counts/len(df_final))

#subset for training
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_final[['iauname','has-spiral-arms_yes']], df_final['Arm_Count'], test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


df_train = pd.DataFrame([X_train.iauname, X_train['has-spiral-arms_yes'], y_train]).transpose()
df_val = pd.DataFrame([X_val.iauname, X_val['has-spiral-arms_yes'], y_val]).transpose()
df_test = pd.DataFrame([X_test.iauname, X_test['has-spiral-arms_yes'], y_test]).transpose()

df_train['Arm_Count'] = df_train['Arm_Count'].astype('int')
df_val['Arm_Count'] = df_val['Arm_Count'].astype('int')
df_test['Arm_Count'] = df_test['Arm_Count'].astype('int')

from pandas import read_csv
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

#https://imbalanced-learn.org/stable/over_sampling.html

def balance(df):

  # split into input and output elements
  X, y = df[['iauname','has-spiral-arms_yes']], df.Arm_Count
  rus = RandomUnderSampler(random_state=0, sampling_strategy='majority')
  X_u, y_u = rus.fit_resample(X, y)
  # label encode the target ariable
  #y = LabelEncoder().fit_transform(y)
  # transform the dataset
  n = int(round(len(X)/ 5, 0))
  strategy = {0:n, 1:n, 2:n, 3:n, 4:n}
  oversample = SMOTE(sampling_strategy=strategy)
  #X, y = oversample.fit_resample(X, y)
  ros = RandomOverSampler(random_state=0, sampling_strategy=strategy)
  X_resampled, y_resampled = ros.fit_resample(X_u, y_u)
  df_resampled = pd.DataFrame([X_resampled['iauname'], X_resampled['has-spiral-arms_yes'], y_resampled]).transpose()
  df = df_resampled.sample(frac=1).reset_index(drop=True)
  return df

df_train = balance(df_train)
df_valid = balance(df_val)
df_test = balance(df_test)

df_train

df_valid.Arm_Count.hist()

dft = df_train[df_train['Arm_Count'] != 4]
dfv = df_valid[df_valid['Arm_Count'] != 4]
dfte = df_test[df_test['Arm_Count'] != 4]

dft = dft[dft['Arm_Count'] != 2]
dfv = dfv[dfv['Arm_Count'] != 2]
dfte = dfte[dfte['Arm_Count'] != 2]

dft['Arm_Count'].hist()

#check with two arms


im = io.imread("/content/drive/MyDrive/Thesis/Pics/J090/J090940.81-002758.4.png")
im_ = color.rgb2gray(im)
im_c = im_[int(1*im_.shape[0]/5):int(4*im_.shape[0]/5),int(1*im_.shape[1]/5):int(4*im_.shape[1]/5)]
im_clo = median(img_as_ubyte(im_c), disk(4))

#print(sharpened)
plt.imshow(im_clo, cmap = 'gray')


#1. Segmentation using thresholding

plt.figure(figsize= (10,10))
i = 1
#all pixels with intensity below 0.45 set to 1
for b in [55,105,155,205]:
  image = im_clo
  #image = rescale_intensity(sharpened)
  image = exposure.adjust_gamma(image, 1.5)

  block_size = b
  adaptive_thresh = threshold_local(image,
  block_size, offset=0)
  binary_adaptive = image > adaptive_thresh

  plt.subplot(2,2,i)
  plt.imshow(skeletonize(binary_adaptive), cmap = 'gray')
  plt.title('1. Thresholding Applied - < 0.45')
  plt.axis('off')
  i += 1

#check with three arms

from skimage import io, color, data
from skimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing, disk, remove_small_objects, square
from skimage.segmentation import clear_border, watershed, flood_fill, slic, mark_boundaries, chan_vese
from skimage.filters.rank import median
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import threshold_local
from skimage.morphology import dilation, erosion, opening, closing,reconstruction, extrema, binary_closing
from skimage.morphology import skeletonize

im_ = color.rgb2gray(im)
im_c = im_[int(1*im_.shape[0]/5):int(4*im_.shape[0]/5),int(1*im_.shape[1]/5):int(4*im_.shape[1]/5)]
im_clo = median(img_as_ubyte(im_c), disk(4))

#print(sharpened)
plt.imshow(im_clo, cmap = 'gray')


#1. Segmentation using thresholding

plt.figure(figsize= (10,10))
i = 1
#all pixels with intensity below 0.45 set to 1
for b in [55,105,155,205]:
  image = im_clo
  #image = rescale_intensity(sharpened)
  image = exposure.adjust_gamma(image, 1.5)

  block_size = b
  adaptive_thresh = threshold_local(image,
  block_size, offset=0)
  binary_adaptive = image > adaptive_thresh

  plt.subplot(2,2,i)
  plt.imshow(skeletonize(binary_adaptive), cmap = 'gray')
  plt.title('1. Thresholding Applied - < 0.45')
  plt.axis('off')
  i += 1

#check with four arms

from skimage import io, color, data
from scipy import ndimage
from skimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing, disk, remove_small_objects, square
from skimage.segmentation import clear_border, watershed, flood_fill, slic, mark_boundaries, chan_vese
from skimage.filters.rank import median
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import threshold_local
from skimage.morphology import dilation, erosion, opening, closing,reconstruction, extrema, binary_closing
from skimage.morphology import skeletonize
from skimage.exposure import rescale_intensity

im = io.imread("/content/drive/MyDrive/Thesis/Pics/J013/J013200.62-022200.3.png")


im_ = color.rgb2gray(im)

im_c = im_[int((im_.shape[0]/2)-50):int((im_.shape[0]/2)+50),int((im_.shape[0]/2)-50):int((im_.shape[0]/2)+50)]

im_clo = median(img_as_ubyte(im_c), disk(4))


#print(sharpened)
plt.imshow(im_clo, cmap = 'gray')


#1. Segmentation using thresholding

plt.figure(figsize= (10,10))
i = 1
#all pixels with intensity below 0.45 set to 1
for b in [55,105,155,205]:
  image = im_clo
  #image = rescale_intensity(sharpened)
  image = exposure.adjust_gamma(image, 1)
  block_size = b
  adaptive_thresh = threshold_local(image,
  block_size, offset=0)
  binary_adaptive = image > adaptive_thresh
  
  print(skeletonize(binary_adaptive).shape)
  plt.subplot(2,2,i)
  plt.imshow(skeletonize(binary_adaptive), cmap = 'gray')
  plt.title('1. Thresholding Applied - < 0.45')
  plt.axis('off')
  i += 1

skeletonize(binary_adaptive).astype(np.uint8)

#for image printing
plt.figure(figsize = (15,15))


image = io.imread("/content/drive/MyDrive/Thesis/Pics/J090/J090940.81-002758.4.png")
plt.subplot(3,2,1)
plt.title('1. Original Image', fontdict = {'fontsize' : 18})
plt.imshow(image)
plt.axis('off')

image = img_as_float(image)
image = exposure.adjust_gamma(image, 2.5)

plt.subplot(3,2,2)
plt.imshow(image)
plt.title('2. Gamma Exposure Adjustment', fontdict = {'fontsize' : 18})
plt.axis('off')
image = color.rgb2gray(image)

image = image[int((image.shape[0]/2)-50):int((image.shape[0]/2)+50),int((image.shape[0]/2)-50):int((image.shape[0]/2)+50)]
plt.subplot(3,2,3)
plt.imshow(image, cmap = 'gray')
plt.title('3. Cropped to 100 x 100', fontdict = {'fontsize' : 18})
plt.axis('off')

image = median(img_as_ubyte(image), disk(4))
plt.subplot(3,2,4)
plt.title('4. Median Filter Applied', fontdict = {'fontsize' : 18})
plt.imshow(image, cmap = 'gray')
plt.axis('off')

block_size = 65
adaptive_thresh = threshold_local(image, block_size, offset=0)
image2 = image > adaptive_thresh
plt.subplot(3,2,5)
plt.title('5. Adaptive Thresholding Applied', fontdict = {'fontsize' : 18})
plt.imshow(image2, cmap = 'gray')
plt.axis('off')

image = skeletonize(image2, method='lee').astype(np.uint8)*255
plt.subplot(3,2,6)
plt.title('6. Skeletonized Image', fontdict = {'fontsize' : 18})
plt.imshow(image, cmap = 'gray')
plt.axis('off')
plt.subplots_adjust(hspace=1.5)
plt.tight_layout()

def skel_images(filepath, filename):
  
  image = io.imread(filepath + filename[:4] + '/' + filename + '.png')

  image = img_as_float(image)
  image = exposure.adjust_gamma(image, 2.5)
  image = color.rgb2gray(image)

  image = image[int((image.shape[0]/2)-50):int((image.shape[0]/2)+50),int((image.shape[0]/2)-50):int((image.shape[0]/2)+50)]

  image = median(img_as_ubyte(image), disk(4))
  block_size = 65
  adaptive_thresh = threshold_local(image, block_size, offset=0)
  image2 = image > adaptive_thresh
  return skeletonize(image2).astype(np.uint8)

def basic(filepath, filename):
  image = Image.open(filepath + filename[:4] + '/' + filename + '.png')
  #image = np.asarray(image)
  #image = exposure.adjust_gamma(image, 2.5)
  #enhancer = ImageEnhance.Contrast(image)
 # factor = 1.5 #increase contrast
  #image = enhancer.enhance(factor)
  #i = np.asarray(i)

  return image

x = np.asarray(basic(filepath, 'J013200.62-022200.3'))
np.max(x)
#y = io.imread(filepath + 'J013' + '/' + 'J013200.62-022200.3' + '.png')
#image = img_as_float(y)
#image = exposure.adjust_gamma(image, 1)
#x = skel_images(filepath, 'J013200.62-022200.3')
#x2 = images(filepath, 'J013200.62-022200.3')
#print(df_train.iauname['J013200.62-022200.3'])
#plt.imshow(x)
#x = transforms.ToPILImage(x)

#x = transforms.ToPILImage()(c)
#transforms.CenterCrop(255),
#transforms.RandomHorizontalFlip(),
#x = transforms.ToTensor()(x)
#x.shape

#RGB and skel
#create a torch data set

from pandas._libs import index
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize

class imageloader(Dataset):
  def __init__(self, base, df, in_col, out_col, transform1, transform2):
    self.df = df
    self.data1 = []
    self.data2 = []
    self.data3 = []
    self.labels = []
    self.files = []
    for ind in tqdm(range(len(df))):
      i1 = transform1(basic(base,df.iloc[ind][in_col]))
      i2 = transform2(skel_images(base,df.iloc[ind][in_col]))
      i3 = torch.cat((i1.permute(1,2,0),i2.permute(1,2,0)),2).permute(2,0,1)
      i4 = df.iloc[ind][out_col]
      self.data1.append(i1)
      self.data2.append(i2)
      self.data3.append(i3)
      self.labels.append(i4)
      self.files.append(df.iloc[ind][in_col])
  def __len__(self):
    return len(self.data3)
    #return len(self.skel)
  def __getitem__(self, idx):
    return  self.data1[idx], self.data2[idx], self.data3[idx], self.labels[idx], self.files[idx]

#https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
#https://pytorch.org/vision/stable/models.html

#using the pretrained models ResNet18 and VGG16 means need to normalize and transform all images to be atleast size 224,224

import torchvision.transforms as transforms
normalize1 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


#training set is 80% of df_small
train_set = imageloader(filepath, pd.concat([dft, dfv], axis =0), 'iauname', 'Arm_Count', transform1 = transforms.Compose([
            transforms.CenterCrop(100),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomAutocontrast(0.25),
            transforms.ToTensor(),
            normalize1
        ]), transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize 
        ]))

#valid_set = imageloader(filepath, df_valid, 'iauname', 'Arm_Count', transform1 = transforms.Compose([
#            transforms.CenterCrop(224),
#            transforms.RandomHorizontalFlip(),
#            transforms.ToTensor(),
#            normalize1
#        ]), transform2 = transforms.Compose([
#            transforms.ToPILImage(),
#            transforms.Resize(224),
#            transforms.RandomHorizontalFlip(),
#            transforms.ToTensor() ,
#            normalize
#        ]))

test_set = imageloader(filepath, dfte, 'iauname', 'Arm_Count', transform1 = transforms.Compose([
            transforms.CenterCrop(100),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomAutocontrast(0.25),
            transforms.ToTensor(),
            normalize1
        ]), transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor() ,
            normalize
        ]))



#load into dataloaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
#valid_loader = DataLoader(valid_set3, batch_size=64, shuffle=True)
test_loader =  DataLoader(test_set, batch_size=64, shuffle=True)

#display image being fed into model
ims = []
lab = []
files = []
#t = []
for a, b, c, d,file in train_loader:
  files.append(file)
  ims.append(a)
  lab.append(d)
print(files[0][1], lab[0][1])

img = transforms.ToPILImage()((ims[0][1]))#.cpu())
img
#print(ims[:][:][0:3].shape)
#torch.unique(ims[0][0][1])
#print(lab[0])
#img = transforms.ToPILImage()((ims[0][0][0]))#.cpu())
#ims[0][0][0].unique()
#t[0][0][0].unique()

df_train.loc[df_train.iauname == 'J153140.79+173709.1']

#def weight_reset(m):
#    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#        m.reset_parameters()

#for param in resnet18.parameters():
#  param.requires_grad = True
  # Replace the last fully-connected layer
  # Parameters of newly constructed modules have requires_grad=True by default
#resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#resnet18.fc = nn.Sequential(nn.Linear(512, 256), nn.Softmax(), nn.Linear(256,4))

#resnet18.to(device)

#from torchsummary import summary

#summary(resnet18, (1,100,100))

# Create a sequential model
def CNN3(n):
  model = nn.Sequential()

  # Add convolutional and pooling layers
  model.add_module('Conv_1', nn.Conv2d(in_channels=n, out_channels=64, kernel_size=(3,3), stride = (1,1), padding = 'same'))
  model.add_module('Relu_1', nn.ReLU())
  model.add_module('Batch1', nn.BatchNorm2d(64))
  model.add_module('MaxPool_1', nn.MaxPool2d(kernel_size=2))

  model.add_module('Conv_2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride = (1,1), padding = 'same'))
  model.add_module('Relu_2', nn.ReLU())
  model.add_module('Batch2', nn.BatchNorm2d(128))
  model.add_module('MaxPool_2', nn.MaxPool2d(kernel_size=2))

  model.add_module('Conv_3', nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3,3), stride = (1,1), padding = 'same'))
  model.add_module('Relu_3', nn.ReLU())
  model.add_module('Batch3', nn.BatchNorm2d(512))
  model.add_module('MaxPool_3', nn.MaxPool2d(kernel_size=2))

  model.add_module('Flatten', nn.Flatten())
  # Add a Linear layer with relu activation
  model.add_module('Linear_1', nn.Linear(in_features=512*3*3, out_features=576, bias=True))
  model.add_module('Relu_L_1', nn.ReLU())

  # Add the last Linear layer.
  model.add_module('Linear_2', nn.Linear(in_features=576, out_features=5, bias=True))


  return model.to(device)

# Create a sequential model
def CNN6(n):
  model = nn.Sequential()

  # Add convolutional and pooling layers
  model.add_module('Conv_1', nn.Conv2d(in_channels=n, out_channels=64, kernel_size=(3,3), stride = (1,1), padding = 'same'))
  model.add_module('Relu_1', nn.ReLU())
  model.add_module('Batch1', nn.BatchNorm2d(64))
  model.add_module('MaxPool_1', nn.MaxPool2d(kernel_size=2))

  model.add_module('Conv_2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride = (1,1), padding = 'same'))
  model.add_module('Relu_2', nn.ReLU())
  model.add_module('Batch2', nn.BatchNorm2d(128))
  model.add_module('MaxPool_2', nn.MaxPool2d(kernel_size=2))

  model.add_module('Conv_3', nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3,3), stride = (1,1), padding = 'same'))
  model.add_module('Relu_3', nn.ReLU())
  model.add_module('Batch3', nn.BatchNorm2d(512))
  model.add_module('MaxPool_3', nn.MaxPool2d(kernel_size=2))

  model.add_module('Conv_4', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3),  stride = (1,1), padding = 'same'))
  model.add_module('Relu_4', nn.ReLU())
  model.add_module('Batch4', nn.BatchNorm2d(1024))
  model.add_module('MaxPool_4', nn.MaxPool2d(kernel_size=2))

  model.add_module('Conv_5', nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(3,3),  stride = (1,1), padding = 'same'))
  model.add_module('Relu_5', nn.ReLU())
  model.add_module('Batch5', nn.BatchNorm2d(2048))
  model.add_module('MaxPool_5', nn.MaxPool2d(kernel_size=2))

  model.add_module('Conv_6', nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=(3,3),  stride = (1,1), padding = 'same'))
  model.add_module('Relu_6', nn.ReLU())
  model.add_module('Batch6', nn.BatchNorm2d(4096))
  model.add_module('MaxPool_6', nn.MaxPool2d(kernel_size=2))

  model.add_module('Flatten', nn.Flatten())
  # Add a Linear layer and relu activation
  model.add_module('Linear_1', nn.Linear(in_features=4096, out_features=1024, bias=True))
  model.add_module('Relu_L_1', nn.ReLU())
  model.add_module('Dropout_1', nn.Dropout(0.3))

  model.add_module('Linear_2', nn.Linear(in_features=1024, out_features=1024, bias=True))
  model.add_module('Relu_L_2', nn.ReLU())
  model.add_module('Dropout_2', nn.Dropout(0.3))

  # Add the last Linear layer.
  model.add_module('Linear_3', nn.Linear(in_features=1024, out_features=5, bias=True))


  return model.to(device)

from torchsummary import summary 
summary(load_model_2(4), (4,100,100))

from torchsummary import summary
import torch.nn as nn
import torchvision.models as models

resnet18 = models.resnet18().to(device)
alexnet = models.alexnet().to(device)
vgg16 = models.vgg16(pretrained = True).to(device)

vgg16.to(device)
#for param in vgg16.parameters():
 # param.requires_grad = True
  # Replace the last fully-connected layer
  # Parameters of newly constructed modules have requires_grad=True by default
#vgg16.features._modules['0'] = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1, 1), padding=(1, 1))
#vgg16.classifier._modules['4'] = nn.Softmax()
#vgg16.classifier._modules['6'] = nn.Linear(4096, 4) # assuming that the fc7 layer has 512 neurons, otherwise change it 
#vgg16.to(device)

#summary(vgg16, (3,100,100))

#load pretrained models
from torch import nn
def load_model_3(model, n):
  import torchvision.models as models

  resnet18 = models.resnet18().to(device)
  alexnet = models.alexnet().to(device)
  vgg16 = models.vgg16().to(device)

  for param in vgg16.parameters():
    param.requires_grad = True
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
  vgg16.features._modules['0'] = nn.Conv2d(n, 64, kernel_size=(3,3), stride=(1, 1), padding=(1, 1))
  vgg16.classifier._modules['6'] = nn.Linear(4096, 4) # assuming that the fc7 layer has 512 neurons, otherwise change it 
  #vgg16.to(device)

  for param in resnet18.parameters():
    param.requires_grad = True
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
  resnet18.conv1 = nn.Conv2d(n, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  resnet18.fc = nn.Sequential(nn.Linear(512, 5))
  #resnet18.to(device) # assuming that the fc7 layer has 512 neurons, otherwise change it 
  #resnet18.to(device)
  if model == 'vgg16':
    return vgg16.to(device)
  elif model == 'resnet18':
    return resnet18.to(device)

x =0
for images_rgb, images_skel, images_both, labels, files in test_loader:
  x+=1
  print(x)

#Solution in pytorch - DEEP
#train models on data and test on valid set
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
model_name = ['Basic']
learning_rates = [0.0001]
all_labels = []
all_predictions = []
files_test= []
all_files = []

for model_name in model_name:
  for e in [15]:
    for lr in learning_rates:
      for n in [3]:
        model = CNN6(n)
        print('Model: ' + str(model_name) + '\n ' + 'Channels: ' + str(n) + '\n ' + 'Epochs: ' + str(e) + '\n ' + 'LR: ' + str(lr))
        optimizer = optim.AdamW(model.parameters(), lr=lr) 
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        crossentropy_loss = nn.CrossEntropyLoss()

        model.train() 
        train_acc = 0 
        for epoch in tqdm(range(e)):
          labels_test = []
          predictions_test = []
          running_train_loss = 0.0
          for images_rgb, images_skel, images_both, labels, files in train_loader: 
            if n == 1:
              images = images_skel.to(device)
            elif n == 3:
              images = images_rgb.to(device)
            else:
              images = images_both.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()           
            predictions = model(images)     
            loss = crossentropy_loss(predictions, labels) 
            running_train_loss =+ loss.item()
            loss.backward()
            optimizer.step()
            accuracy = (torch.max(predictions, dim=-1, keepdim=True)[1].flatten() == labels).sum() / len(labels)
            train_acc += accuracy.item()
          scheduler.step()
          writer.add_scalar('Loss/Train', running_train_loss / len(train_loader), epoch)
          train_acc /= len(train_loader)
          writer.add_scalar('Accuracy/Train', train_acc, epoch)
          print('Train Loss' + str(running_train_loss / len(train_loader))+'\n')
        
          model.eval() 
          test_acc = 0 
          running_test_loss = 0
          with torch.no_grad(): 
            for images_rgb, images_skel, images_both, labels, files in test_loader: 
                if n == 1:
                  images = images_skel.to(device)
                elif n == 3:
                  images = images_rgb.to(device)
                else:
                  images = images_both.to(device)
                labels = labels.to(device)
                predictions = model(images)
                accuracy = (torch.max(predictions, dim=-1, keepdim=True)[1].flatten() == labels).sum() / len(labels)
                test_acc += accuracy.item()
                loss = crossentropy_loss(predictions, labels)
                running_test_loss =+ loss.item()
                labels_test.append(labels)
                files_test.append(files)
                predictions_test.append(torch.max(predictions, dim=-1, keepdim=True)[1].flatten())
          all_labels.append(labels_test)
          all_predictions.append(predictions_test)
          all_files.append(files_test)
          writer.add_scalar('Loss/Test', running_test_loss / len(test_loader), epoch)
          test_acc /= len(test_loader)
          writer.add_scalar('Accuracy/Test', test_acc, epoch)
          print('Acc: ' + str(test_acc) + ', Test Loss' + str(running_test_loss / len(test_loader))+'\n')
        writer.close()
### END ###

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext tensorboard
# %tensorboard --logdir runs

#save results for basic 3 layer CNN
acc_basic = acc
prec_basic = prec
train_loss_values_basic = train_loss_values

len(all_labels[2])

sum1 = 0
sum2 = 0
sum3 = 0
sum4 = 0
sum0 = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count0 = 0
tot1 = []
tot2 = []
tot3 =[]
tot4 = []
tot0 = []
g = 10
for x, y in zip(all_labels[g], all_predictions[g]):
  for z in range(len(x)):
    if x[z] == 1:
      sum1 +=  (x[z]  == y[z])
      count1 += 1
    elif x[z] == 2:
      sum2 += (x[z]  == y[z])
      count2 += 1
    elif x[z] == 3:
      sum3 += (x[z]  == y[z])
      count3 +=1
    elif x[z] == 4:
      sum4 += (x[z]  == y[z])
      count4 += 1
    elif x[z] == 0:
      sum0 += (x[z]  == y[z])
      count0 += 1
  tot1.append(sum1 / count1)
  #tot2.append(sum2 / count2)
  tot3.append(sum3 / count3)
  #tot4.append(sum4 / count4)
  tot0.append(sum0 / count0)

x1 = sum(tot1) / len(all_labels[g])
#x2 = sum(tot2) / len(all_labels[g])
x3 = sum(tot3) / len(all_labels[g])
#x4 = sum(tot4) / len(all_labels[g])
x0 = sum(tot0) / len(all_labels[g])
print(x1,x3,x0)

labels_test

# channel 1, 5 epochs
acc_1_5 = acc_basic[0:5]
prec_1_5 = prec_basic[0:5]
train_loss_1_5 = train_loss_values_basic[0:5]

# channel 3, 5 epochs
acc_3_5 = acc_basic[5:10]
prec_3_5 = prec_basic[5:10]
train_loss_3_5 = train_loss_values_basic[5:10]

# channel 4, 5 epochs
acc_4_5 = acc_basic[10:15]
prec_4_5 = prec_basic[10:15]
train_loss_4_5 = train_loss_values_basic[10:15]

print(acc_1_5[4], acc_3_5[4], acc_4_5[4])

# channel 1, 10 epochs
acc_1_10 = acc_basic[15:25]
prec_1_10 = prec_basic[15:25]
train_loss_1_10 = train_loss_values_basic[15:25]

# channel 3, 10 epochs
acc_3_10 = acc_basic[25:35]
prec_3_10 = prec_basic[25:35]
train_loss_3_10 = train_loss_values_basic[25:35]

# channel 4, 10 epochs
acc_4_10 = acc_basic[35:45]
prec_4_10 = prec_basic[35:45]
train_loss_4_10 = train_loss_values_basic[35:45]

print(acc_1_10[9], acc_3_10[9], acc_4_10[9])

# channel 1, 15 epochs
acc_1_10 = acc_basic[15:25]
prec_1_10 = prec_basic[15:25]
train_loss_1_10 = train_loss_values_basic[15:25]

# channel 3, 10 epochs
acc_3_10 = acc_basic[25:35]
prec_3_10 = prec_basic[25:35]
train_loss_3_10 = train_loss_values_basic[25:35]

# channel 4, 10 epochs
acc_4_10 = acc_basic[35:45]
prec_4_10 = prec_basic[35:45]
train_loss_4_10 = train_loss_values_basic[35:45]

print(acc_1_10[9], acc_3_10[9], acc_4_10[9])

"""IGNORE BELOW"""

plt.plot(train_loss_values)

from skimage.morphology import skeletonize
i = 1
for threshold in [0,50,100,200]:
  # Grayscale
  #image_file = im[2].convert('L')
  # Threshold
  image_file = im[2].point( lambda p: 255 if p >= threshold else 0 )
  # To mono
  image_file = image_file.convert('1')
  #image_file = skeletonize(image_file)
  plt.figure()
  plt.subplot(2,2, i)
  plt.imshow(image_file)
  i += 1

#Solution in pytorch - DEEP
#train models on data and test on valid set
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
epochs = [3,5]
models = [vgg16,resnet18]
model_name = ['vgg16', 'resnet18']
acc = []
prec = []
rec = []

for model, names in zip(models,model_name):
  for epoch in epochs:
    print('Model: ' + str(names) + '\n ' + 'Epochs: ' + str(epoch) + ' \n')
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001) 
    crossentropy_loss = nn.CrossEntropyLoss(reduction='mean')

    model.train() 

    for epoch in tqdm(range(epoch)):
        for images, labels in train_loader: 
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()           
            predictions = model(images)     
            loss = crossentropy_loss(predictions, labels) 
            loss.backward() 
            optimizer.step()

    model.eval() 
    test_acc = 0 
    test_prec = 0
    test_recall = 0
    for images, labels in valid_loader: 
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        accuracy = (torch.max(predictions, dim=-1, keepdim=True)[1].flatten() == labels).sum() / len(labels)
        test_acc += accuracy.item()
        test_prec += precision_score(y_true = labels.cpu(), y_pred = torch.max(predictions, dim=-1, keepdim=True)[1].flatten().cpu(), average = 'weighted', zero_division = 0)
        test_recall += recall_score(y_true = labels.cpu(), y_pred = torch.max(predictions, dim=-1, keepdim=True)[1].flatten().cpu(), average = 'micro', zero_division = 0)
    test_acc /= len(valid_loader)
    test_prec /= len(valid_loader)
    test_recall /= len(valid_loader)
    print('Acc: ' + str(test_acc) + ', Precision: ' + str(test_prec) + ', Recall' + str(test_recall)+'\n')
    acc.append(test_acc)
    prec.append(test_prec)
    rec.append(test_recall)
    model.apply(weight_reset)
### END ###

precision_score(y_true = labels.cpu(), y_pred = torch.max(predictions, dim=-1, keepdim=True)[1].flatten().cpu(), average = 'weighted')

#https://stackoverflow.com/questions/41488279/neural-network-always-predicts-the-same-class
torch.max(predictions, dim=-1, keepdim=True)[1].flatten()

test_acc

"""OTHER WORKINGS"""

# Commented out IPython magic to ensure Python compatibility.
from skimage.morphology import skeletonize, thin
from skimage.morphology import square, rectangle, diamond, disk, cube,  octahedron, ball, star, octagon 
from skimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing
from skimage.morphology import dilation, erosion, opening, closing, white_tophat
from skimage import img_as_float
from skimage import io
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('default')

im =  color.rgb2gray(img_as_float(io.imread('J000\J000002.29-042805.0.png')))

im_oc = closing(opening(im.astype('float'),disk(2)), disk(2))

bina = im_oc > 0.3
skel = skeletonize(bina)
#thin = thin(im_oc)

plt.figure(figsize = (10,10))
plt.subplot(1,3,1)
plt.imshow(im, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(bina, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(skel, cmap='gray')

plt.figure()
plt.imshow(im.astype('float'))

#OLD

from pandas import read_csv
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

#https://imbalanced-learn.org/stable/over_sampling.html

# split into input and output elements
X, y = df_final[['iauname','has-spiral-arms_yes']], df_final.Arm_Count
rus = RandomUnderSampler(random_state=0, sampling_strategy='majority')
X_u, y_u = rus.fit_resample(X, y)
# label encode the target ariable
#y = LabelEncoder().fit_transform(y)
# transform the dataset
strategy = {0:7500, 1:7500, 2:7500, 3:7500, 4:7500}
oversample = SMOTE(sampling_strategy=strategy)
#X, y = oversample.fit_resample(X, y)
ros = RandomOverSampler(random_state=0, sampling_strategy=strategy)
X_resampled, y_resampled = ros.fit_resample(X_u, y_u)
# summarize distribution
counter = Counter(y_resampled)
for k,v in counter.items():
	per = v / len(y_resampled) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()