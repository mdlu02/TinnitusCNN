#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io as sio
import numpy as np
import torch
from glob import glob
from numpy import *
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import csv
import copy

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, MSELoss
from torch.optim import Adam, SGD, Adagrad, AdamW, RMSprop, lr_scheduler
from collections import OrderedDict
import torch.nn as nn
import argparse

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA available")
    
print(args.n_gpu)
# place marker
print('Finished imports')


# In[2]:


# labels
labels_path = '/data/tinnitus2/BIDS_T2/participants.tsv'

channels = ['alpha', 'beta', 'delta', 'delta-theta', 'gamma', 'higam', 'theta']

# get all the data path
root = '/data/tinnitus3/MEG_resting/'

sub_dir = []
for channel in tqdm(channels):
    sub_dir += glob(root+'MINN*/ses-01/Closed/'+channel+'/ImCOH*.mat')
    sub_dir += glob(root+'UCSF*/ses-01/Closed/'+channel+'/ImCOH*.mat')
    
sub_dict = dict()
for iden in tqdm(sub_dir):
    if iden[28:35] not in sub_dict:
        sub_dict[iden[28:35]] = [iden]
    else:
        sub_dict[iden[28:35]].append(iden)


# In[3]:


with open(labels_path) as f:
    reader = csv.reader(f)
    labels = list(reader)
ref = dict()
for i in tqdm(range(1, len(labels))):
    if labels[i][0][4:11] in sub_dict:
        if 'Tinnitus' in labels[i][0]:
            ref[labels[i][0][4:11]] = True
        else:
            ref[labels[i][0][4:11]] = False


# In[4]:


sub_paths_control, sub_paths_tinnitus = [], []
for k, v in tqdm(sub_dict.items()):
    if ref[k]:
        sub_paths_tinnitus.append(v)
    else:
        sub_paths_control.append(v)
        
print(len(sub_paths_control), len(sub_paths_tinnitus))


# In[5]:


random.seed(1)
random.shuffle(sub_paths_control)
random.shuffle(sub_paths_tinnitus)

sets = [
    sub_paths_control[:41] + sub_paths_tinnitus[:38], 
    sub_paths_control[41:81] + sub_paths_tinnitus[38:76],
    sub_paths_control[81:121] + sub_paths_tinnitus[76:114], 
    sub_paths_control[121:161] + sub_paths_tinnitus[114:152], 
    sub_paths_control[161:] + sub_paths_tinnitus[152:]
]

num_folds = len(sets)

for i in range(num_folds):
    random.shuffle(sets[i])

print('Data partitioned')


# In[6]:


train_data = [[np.array([])] * len(channels) for _ in range(num_folds)]
train_label = [np.array([]) for _ in range(num_folds)]

for i in tqdm(range(num_folds)):
    fold_dir = sets[i]
    for j in fold_dir:
        sub = j[0][28:35]
        for channel in range(len(channels)):
            loading = sio.loadmat(j[channel])
            mtx = loading['icoh_val']
            np.fill_diagonal(mtx, 0)
            mtx = np.pad(mtx, 5, mode='constant')
            mtx = mtx.reshape((1, len(mtx[0]), len(mtx[1])))
            train_data[i][channel] = np.vstack([train_data[i][channel], mtx]) if train_data[i][channel].size else mtx
            if channel == 0:
                if ref[sub]:
                    train_label[i] = np.append(train_label[i], float(1))
                else:
                    train_label[i] = np.append(train_label[i], float(0))

print('data loaded')


# In[7]:


training_data = [[np.array([]) for _ in range(len(channels))] for _ in range(num_folds)]
training_label = [np.array([]) for _ in range(num_folds)]
val_data = train_data
val_label = train_label

for fold in tqdm(range(num_folds)):
    for skip in range(num_folds):
        if fold == skip:
            continue
        else:
            for channel in range(len(channels)):
                training_data[fold][channel] = np.vstack([training_data[fold][channel], train_data[skip][channel]]) if training_data[fold][channel].size else train_data[skip][channel]
                if channel == 0:
                    training_label[fold] = np.concatenate((training_label[fold], train_label[skip])) if training_label[fold].size else train_label[skip]


# In[8]:


loaded_train_data, loaded_train_labels = [[] for _ in range(num_folds)], []
loaded_val_data, loaded_val_labels = [[] for _ in range(num_folds)], []

for fold in tqdm(range(len(train_data))):
    for channel in range(len(channels)):
        img = training_data[fold][channel]
        train_x = img.reshape(img.shape[0], 1, 256, 256)
        train_x = torch.from_numpy(train_x)
        train_x = train_x.float()
        loaded_train_data[fold].append(train_x)
    
for fold in tqdm(range(len(val_data))):
    for channel in range(len(channels)):
        img = val_data[fold][channel]
        train_x = img.reshape(img.shape[0], 1, 256, 256)
        train_x = torch.from_numpy(train_x)
        train_x = train_x.float()
        loaded_val_data[fold].append(train_x)

for lab in tqdm(training_label):
    train_y = lab.astype(int);
    train_y = torch.from_numpy(train_y)
    train_y = Variable(train_y)
    loaded_train_labels.append(train_y)

for lab in tqdm(val_label):
    val_y = lab.astype(int);
    val_y = torch.from_numpy(val_y)
    val_y = Variable(val_y)
    loaded_val_labels.append(val_y)

    
print('data reshaped')


# In[9]:


# u-net architecture
class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, 1, name="dec1")


    def forward(self, x):
        enc1 = self.encoder1(x)
        #print(enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))
        #print(enc2.shape)
        enc3 = self.encoder3(self.pool2(enc2))
        #print(enc3.shape)
        enc4 = self.encoder4(self.pool3(enc3))
        #print(enc4.shape)

        bottleneck = self.bottleneck(self.pool4(enc4))
        #print(bottleneck.shape)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        #print(dec4.shape)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        #print(dec3.shape)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        #print(dec2.shape)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        #print(dec1.shape)

        return dec1
        
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
model = UNet()
print(model)
base_model_wts = copy.deepcopy(model.state_dict())
print('UNet defined')


# In[10]:


criterion = MSELoss()

model = model.cuda()
criterion = criterion.cuda()
print('model and criterion to CUDA')

n_epochs = 100

train_losses, val_losses, final_val_losses, final_images = [], [], [], []

best_model_wts = copy.deepcopy(model.state_dict())


# In[11]:


# training the model
print('start training')

for channel in tqdm(range(len(channels) - 6)):
    print('channel: ' + channels[channel])
    cross_train_loss, cross_val_loss, final_cross_loss = [], [], []
    for fold in range(num_folds):
        print('fold: ', fold)
        optimizer = Adam(model.parameters(), lr=0.0001)
        model.load_state_dict(base_model_wts)
        x_train = Variable(loaded_train_data[fold][channel]); x_val = Variable(loaded_val_data[fold][channel])
        x_train = x_train.cuda()
        x_val = x_val.cuda()
        print('data to CUDA')
        fold_train_loss, fold_val_loss = [], []
            
        for epoch in tqdm(range(n_epochs)): 
            if True:
                print('-' * 40)
                print('Epoch {}/{}'.format(epoch+1, n_epochs))
            model.train(); optimizer.zero_grad()
            output_train = model(x_train)
            with torch.no_grad():
                model.eval()
                output_val = model(x_val)
            
            loss_train = criterion(output_train, x_train)
            loss_val = criterion(output_val, x_val)
            fold_train_loss.append(loss_train.item()); fold_val_loss.append(loss_val.item())
            if loss_val.item() <= max(fold_val_loss):
                print('updating')
                best_model_wts = copy.deepcopy(model.state_dict())
            print(' train loss : ', loss_train.data.numpy(), '\n', 'validation loss : ', loss_val.data.numpy())
            loss_train.backward()
            optimizer.step()
        model.load_state_dict(best_model_wts)
        temp_train_output = model(x_train)
        temp_val_output = model(x_val)
        final_loss = criterion(temp_val_output, x_val)

        original = x_train.data[0]
        original = original[0,:,:]
        reconstructed = temp_train_output.data[0]
        reconstructed = reconstructed[0,:,:]
        final_images.append([original, reconstructed])
        
        if len(final_cross_loss) == 0:
            print('new best model')
            torch.save(model.state_dict(), 'reconstruction_best_' + channels[channel] + '_channel_weights.pth')
        elif final_loss.item() >= max(final_cross_loss):
            print('new best model')
            torch.save(model.state_dict(), 'reconstruction_best_' + channels[channel] + '_channel_weights.pth')
        print('best loss: ', '\t', final_loss)
        cross_train_loss.append(fold_train_loss); cross_val_loss.append(fold_val_loss); final_cross_loss.append(final_loss.data)
    train_losses.append(cross_train_loss); val_losses.append(cross_val_loss); final_val_losses.append(final_cross_loss)
    


# In[12]:


# plot the learning curves
for channel in range(len(train_losses)):
    for fold in range(num_folds):
        plt.plot(train_losses[channel][fold],label='channel ' + channels[channel] + ' training losses: fold ' + str(fold))
plt.legend(loc='best'); plt.title("Training losses"); plt.ylabel("Loss value"); plt.xlabel("Epoch number")  ; plt.show()

for channel in range(len(val_losses)):
    for fold in range(num_folds):
        plt.plot(val_losses[channel][fold],label='channel ' + channels[channel] + ' validation losses: fold ' + str(fold))
plt.legend(loc='best'); plt.title("Validation losses"); plt.ylabel("Loss value"); plt.xlabel("Epoch number") ; plt.show()

# plot images

for pair in range(len(final_images)):
    figure = plt.figure()
    original, reconstructed = final_images[pair][0], final_images[pair][1]
    figure.add_subplot(1, 2, 1)
    plt.imshow(original, label='Original image (fold {})'.format(counter - 2))
    figure.add_subplot(1, 2, 2)
    plt.imshow(reconstructed, label='Reconstructed image (fold {})'.format(counter - 2))
    counter += 1
    plt.show(block=True)


# In[ ]:


print('Train loss:'); print(train_losses); print('Validation loss:'); print(val_losses)
print('Final losses: '); print(final_val_losses); print('Images: '); print(final_images)

