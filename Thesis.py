
#import and install all necessary programs

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
from pandas import read_csv
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from skimage import io, color, data
from skimage.filters.rank import median
from skimage.util import img_as_float, img_as_ubyte
from skimage.morphology import skeletonize
from tqdm import tqdm
from pandas._libs import index
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch.optim as optim

plt.style.use('default')
filepath = ()

from google.colab import drive
drive.mount('/content/drive')



#read file with classifications
df = pd.read_csv(filepath + 'gz_decals_volunteers_5.csv')
df = df.sort_values('iauname')

#only included observations with more than 5 votes and 'yes' over 50% and not 'wrong size'
conditions = [
    (df['has-spiral-arms_yes_fraction'] > 0.5) &
     (df['wrong_size_warning'] == False) & (df['has-spiral-arms_yes'] >= 5)
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


#function to resample using RandomOverSampler and RandomUnderSampler
def balance(df):

  # split into input and output elements
  X, y = df[['iauname','has-spiral-arms_yes']], df.Arm_Count
  rus = RandomUnderSampler(random_state=0, sampling_strategy='majority')
  X_u, y_u = rus.fit_resample(X, y)
  n = int(round(len(X)/ 5, 0))
  strategy = {0:n, 1:n, 2:n, 3:n, 4:n}
  ros = RandomOverSampler(random_state=0, sampling_strategy=strategy)
  X_resampled, y_resampled = ros.fit_resample(X_u, y_u)
  df_resampled = pd.DataFrame([X_resampled['iauname'], X_resampled['has-spiral-arms_yes'], y_resampled]).transpose()
  df = df_resampled.sample(frac=1).reset_index(drop=True)
  return df

#resample train, validation and test set 

df_train = balance(df_train)
df_valid = balance(df_val)
df_test = balance(df_test)


#function to load RGB images
def basic(filepath, filename, gamma):
  
  image = io.imread(filepath + filename[:4] + '/' + filename + '.png')
  image = img_as_float(image)

  return img_as_ubyte(image)

#function to skeletonize images
def skel_images(filepath, filename, gamma_size, block_size, disk_size):
  
  image = io.imread(filepath + filename[:4] + '/' + filename + '.png')

  image = img_as_float(image)
  image = exposure.adjust_gamma(image, gamma_size)
  image = color.rgb2gray(image)

  image = image[int((image.shape[0]/2)-50):int((image.shape[0]/2)+50),int((image.shape[0]/2)-50):int((image.shape[0]/2)+50)]
  disk_size = disk_size
  image = median(img_as_ubyte(image), disk(disk_size))
  adaptive_thresh = threshold_local(image, block_size, offset=0)
  image2 = image > adaptive_thresh
  return skeletonize(image2).astype(np.uint8)



#Define Maxout activation function
class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
        m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]).max(2)
        return m


#Create Baseline CNN model
class Baseline(nn.Module):
    def __init__(self, n, drop):
        super(Baseline, self).__init__()

        self.Conv_1 = nn.Conv2d(in_channels=n, out_channels=32, kernel_size=(6,6))
        self.Relu_1 = nn.ReLU()
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=2)

        self.Conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5))
        self.Relu_2 = nn.ReLU()
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=2)

        self.Conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3))
        self.Relu_3 = nn.ReLU()

        self.Conv_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3))
        self.Relu_4 = nn.ReLU()
        self.MaxPool_4 = nn.MaxPool2d(kernel_size=2)

        self.Flatten = nn.Flatten()
        
        #Add a Linear layer with 64 units and relu activation
        self.Linear_1 = nn.Linear(in_features=128*8*8, out_features=2048, bias=True)
        self.Dropout_1 = nn.Dropout(drop)
        self.MaxOut_1 = Maxout(2)
        
        # Add the last Linear layer.
        self.Linear_2 = nn.Linear(in_features=1024, out_features=2048, bias=True)
        self.Dropout_2 = nn.Dropout(drop)
        self.MaxOut_2 = Maxout(2)

        self.Linear_3 = nn.Linear(in_features=1024, out_features=5, bias=True)
        self.Dropout_3 = nn.Dropout(drop)
        self.Out_activation = nn.Softmax(-1)

    def forward(self, inputs):
        c1 = self.Conv_1(inputs)
        r1 = self.Relu_1(c1)
        m1 = self.MaxPool_1(r1)

        c2 = self.Conv_2(m1)
        r2 = self.Relu_2(c2)
        m2 = self.MaxPool_2(r2)

        c3 = self.Conv_3(m2)
        r3 = self.Relu_3(c3)
        
        c4 = self.Conv_4(r3)
        r4 = self.Relu_3(c4)
        m4 = self.MaxPool_2(r4)
        x = self.Flatten(m4)

        x = self.Linear_1(x)
        x = self.Dropout_1(x)
        x = self.MaxOut_1(x)
        x = self.Linear_2(x)
        x = self.Dropout_2(x)
        x = self.MaxOut_2(x)
        x = self.Linear_3(x)
        x = self.Dropout_3(x)
        x = self.Out_activation(x)

        return x, c1, m1, c2, m2, c3, c4,m4


#Create Wide Dieleman CNN model

class WideDieleman(nn.Module):
    def __init__(self, k, n, drop):
        super(WideDieleman, self).__init__()

        self.Conv_1 = nn.Conv2d(in_channels=n, out_channels=64, kernel_size=(6,6), stride = 1)
        self.BatchNorm_1 = nn.BatchNorm2d(64)
        self.Relu_1 = nn.ReLU()

        self.Conv_2 = nn.Conv2d(in_channels=64, out_channels=64*k, kernel_size=(5,5))
        self.BatchNorm_2 = nn.BatchNorm2d(64*k)
        self.Relu_2 = nn.ReLU()

        self.Conv_3 = nn.Conv2d(in_channels=64*k, out_channels=64*k, kernel_size=(3,3), stride = (2,2))
        self.BatchNorm_3 = nn.BatchNorm2d(64*k)
        self.Relu_3 = nn.ReLU()

        self.Conv_4 = nn.Conv2d(in_channels=64*k, out_channels=128*k, kernel_size=(1,1))
        self.BatchNorm_4 = nn.BatchNorm2d(128*k)
        self.Relu_4 = nn.ReLU()

        self.Conv_5 = nn.Conv2d(in_channels=256, out_channels=128*k, kernel_size=(3,3), stride = 2)
        self.BatchNorm_5 = nn.BatchNorm2d(128*k)
        self.Relu_5 = nn.ReLU()

        self.Conv_6 = nn.Conv2d(in_channels=128*k, out_channels= 128*k, kernel_size=(1,1))
        self.BatchNorm_6 = nn.BatchNorm2d(128*k)
        self.Relu_6 = nn.ReLU()

        self.Conv_7 = nn.Conv2d(in_channels=128*k, out_channels= 512*k, kernel_size=(3,3), stride = 2)
        self.BatchNorm_7 = nn.BatchNorm2d(512*k)
        self.Relu_7 = nn.ReLU()

        self.Flatten = nn.Flatten()
        
        #Add a Linear layer with 64 units and relu activation
        self.Linear_1 = nn.Linear(in_features=1024*10*10, out_features=2048, bias=True)
        self.Dropout_L_1 = nn.Dropout(drop)
        self.MaxOut_1 = Maxout(2)
        
        # Add the last Linear layer.
        self.Linear_2 = nn.Linear(in_features=1024, out_features=5, bias=True)
        self.Out_activation = nn.Softmax(-1)


    def forward(self, inputs):
        c1 = self.Conv_1(inputs)
        b1 = self.BatchNorm_1(c1)
        r1 = self.Relu_1(b1)

        c2 = self.Conv_2(r1)
        b2 = self.BatchNorm_2(c2)
        r2 = self.Relu_2(b2)

        c3 = self.Conv_3(r2)
        b3 = self.BatchNorm_3(c3)
        r3 = self.Relu_3(b3)
        
        c4 = self.Conv_4(r3)
        b4 = self.BatchNorm_4(c4)
        r4 = self.Relu_4(b4)

        c5 = self.Conv_5(r4)
        b5 = self.BatchNorm_5(c5)
        r5 = self.Relu_5(b5)

        c6 = self.Conv_6(r5)
        b6 = self.BatchNorm_6(c6)
        r6 = self.Relu_6(b6)

        c7 = self.Conv_7(r6)
        b7 = self.BatchNorm_7(c7)
        r7 = self.Relu_7(b7)

        x = self.Flatten(r7)

        x = self.Linear_1(x)
        x = self.Dropout_L_1(x)
        x = self.MaxOut_1(x)
        x = self.Linear_2(x)
        x = self.Out_activation(x)

        return x, c1, c2, c3, c4, c5, c6, c7


#Create  Adapted Baseline CNN model

class AdaptBaseline(nn.Module):
    def __init__(self, n, drop):
        super(AdaptBaseline, self).__init__()

        self.Conv_1 = nn.Conv2d(in_channels=n, out_channels=32, kernel_size=(3,3))
        self.Relu_1 = nn.ReLU()
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=2)

        self.Conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5))
        self.Relu_2 = nn.ReLU()
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=2)

        self.Conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3))
        self.Relu_3 = nn.ReLU()

        self.Conv_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3))
        self.Relu_4 = nn.ReLU()
        self.MaxPool_4 = nn.MaxPool2d(kernel_size=2)

        self.Flatten = nn.Flatten()
        
        #Add a Linear layer with 64 units and relu activation
        self.Linear_1 = nn.Linear(in_features=128*9*9, out_features=2048, bias=True)
        self.Dropout_1 = nn.Dropout(drop)
        self.MaxOut_1 = Maxout(2)
        
        # Add the last Linear layer.
        self.Linear_2 = nn.Linear(in_features=1024, out_features=2048, bias=True)
        self.Dropout_2 = nn.Dropout(drop)
        self.MaxOut_2 = Maxout(2)

        self.Linear_3 = nn.Linear(in_features=1024, out_features=5, bias=True)
        self.Dropout_3 = nn.Dropout(drop)
        self.Out_activation = nn.Softmax(-1)

    def forward(self, inputs):
        c1 = self.Conv_1(inputs)
        r1 = self.Relu_1(c1)
        m1 = self.MaxPool_1(r1)

        c2 = self.Conv_2(m1)
        r2 = self.Relu_2(c2)
        m2 = self.MaxPool_2(r2)

        c3 = self.Conv_3(m2)
        r3 = self.Relu_3(c3)
        
        c4 = self.Conv_4(r3)
        r4 = self.Relu_3(c4)
        m4 = self.MaxPool_2(r4)
        x = self.Flatten(m4)

        x = self.Linear_1(x)
        x = self.Dropout_1(x)
        x = self.MaxOut_1(x)
        x = self.Linear_2(x)
        x = self.Dropout_2(x)
        x = self.MaxOut_2(x)
        x = self.Linear_3(x)
        x = self.Dropout_3(x)
        x = self.Out_activation(x)

        return x, c1, m1, c2, m2, c3, c4,m4



#create a torch data set

      class imageloader(Dataset):
        def __init__(self, base, df, in_col, out_col, transform1, transform2):
          self.df = df
          self.data1 = []
          self.data2 = []
          self.data3 = []
          self.labels = []
          self.files = []
          for ind in tqdm(range(len(df))):
            i1 = transform1(basic(base,df.iloc[ind][in_col],2))
            i2 = transform2(skel_images(base,df.iloc[ind][in_col], gamma, block, disk_size))
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

      import torchvision.transforms as transforms
      normalize1 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])


      #training set
      train_set = imageloader(filepath, df_train, 'iauname', 'Arm_Count', transform1 = transforms.Compose([
                  transforms.ToPILImage(),
                  transforms.CenterCrop(100),
                  transforms.Resize(100),
                  transforms.RandomHorizontalFlip(),
                  transforms.RandomVerticalFlip(),
                  transforms.RandomApply([transforms.RandomRotation(180)], p = 0.5),
                  transforms.RandomApply([transforms.RandomRotation(90)], p = 0.5),
                  transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
                  transforms.ToTensor(),
                  normalize1
              ]), transform2 = transforms.Compose([
                  transforms.ToPILImage(),
                  transforms.Resize(100),
                  transforms.RandomVerticalFlip(),
                  transforms.RandomApply([transforms.RandomRotation(180)], p = 0.5),
                  transforms.RandomApply([transforms.RandomRotation(90)], p = 0.5),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize 
              ]))

      valid_set = imageloader(filepath, df_valid, 'iauname', 'Arm_Count', transform1 = transforms.Compose([
                 transforms.ToPILImage(),
                 transforms.CenterCrop(100),
                 transforms.Resize(100),
                 transforms.ToTensor(),
                 normalize1
             ]), transform2 = transforms.Compose([
                 transforms.ToPILImage(),
                  transforms.Resize(100),
                 transforms.ToTensor() ,
                 normalize
             ]))

      test_set = imageloader(filepath, df_test, 'iauname', 'Arm_Count', transform1 = transforms.Compose([
                 transforms.ToPILImage(),
                 transforms.CenterCrop(100),
                 transforms.Resize(100),
                 transforms.ToTensor(),
                 normalize1
             ]), transform2 = transforms.Compose([
                 transforms.ToPILImage(),
                  transforms.Resize(100),
                 transforms.ToTensor() ,
                 normalize
             ]))

#load into dataloaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=True)
test_loader =  DataLoader(test_set, batch_size=64, shuffle=True)



#train models on data and test on valid set

learning_rates = 0.0001
loader = test_loader
drop = 0.5
epoch = 50

all_labels = []
all_predictions = []
files_test= []
all_files = []


model = Baseline(n,drop).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr) 
crossentropy_loss = nn.CrossEntropyLoss()

train_accuracy = []
train_loss = []
test_accuracy = []
test_loss = []
model.train() 
train_acc = 0 
for e in tqdm(range(epoch)):
  labels_test = []
  predictions_test = []
  running_train_loss = 0.0
  if e == 50:
    for g in optimizer.param_groups:
      g['lr'] = 0.00005
  for images_rgb, images_skel, images_both, labels, files in train_loader: 
    if n == 1:
      images = images_skel.to(device)
    elif n == 3:
      images = images_rgb.to(device)
    else:
      images = images_both.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()           
    predictions = model(images)[0]     
    loss = crossentropy_loss(predictions, labels) 
    running_train_loss =+ loss.item()
    loss.backward()
    optimizer.step()
    accuracy = (torch.max(predictions, dim=-1, keepdim=True)[1].flatten() == labels).sum() / len(labels)
    train_acc += accuracy.item()
  train_loss.append(running_train_loss / len(train_loader))
  train_acc /= len(train_loader)
  print('Train Loss' + str(running_train_loss / len(train_loader))+'\t Accuracy'+ str(train_acc))
  train_accuracy.append(train_acc)
  

  model.eval() 
  test_acc = 0 
  running_test_loss = 0
  with torch.no_grad(): 
    for images_rgb, images_skel, images_both, labels, files in loader: 
        if n == 1:
          images = images_skel.to(device)
        elif n == 3:
          images = images_rgb.to(device)
        else:
          images = images_both.to(device)
        labels = labels.to(device)
        predictions = model(images)[0]
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
  test_acc /= len(loader)
  test_accuracy.append(test_acc)
  test_loss.append(running_test_loss / len(loader))
  print('Acc: ' + str(test_acc) + ', Test Loss' + str(running_test_loss / len(loader))+'\n')

#create confusion matrix from predictions
from sklearn import metrics
labels_list = []
pred_list = []
for x, x2 in zip(labels_test, predictions_test):
  for y,y2 in zip(x,x2):
    labels_list.append(float(y.cpu()))
    pred_list.append(float(y2.cpu()))
print(metrics.confusion_matrix(y_true = labels_list, y_pred = pred_list))


#download data into files
zipped = list(zip(train_loss, test_loss, train_accuracy, test_accuracy))
df = pd.DataFrame(zipped, columns=['Train Loss', 'Test Loss','Train Acc','Test Acc'])
df.to_csv( filepath + '/results/both_adapt_new.csv', index=False, encoding='utf-8-sig')

df = pd.read_csv(filepath + '/results/both_adapt_new.csv', encoding='utf-8-sig')
df_smooth = df.ewm(alpha=(1 - 0.6)).mean()
df_smooth.to_csv( filepath + '/results/both_adapt_smooth_new.csv', index=False, encoding='utf-8-sig')
