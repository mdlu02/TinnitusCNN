# IMPORTS
import scipy.io as sio
import numpy as np
import torch
from glob import glob
from numpy import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import csv
import copy

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d
from torch.optim import Adam, SGD, Adagrad, AdamW, RMSprop, lr_scheduler
from collections import OrderedDict
import torch.nn as nn

print('Finished imports')


# LABELS
labels_path = '/data/tinnitus2/BIDS_T2/participants.tsv'

# CHANNELS
channels = ['alpha', 'beta', 'delta', 'delta-theta', 'gamma', 'higam', 'theta']

# ROOT DIRECTORY
root = '/data/tinnitus3/MEG_resting/'

# IMAGE PATHS
sub_dir = []
for channel in tqdm(channels):
    sub_dir += glob(root+'MINN*/ses-01/Closed/'+channel+'/ImCOH*.mat')
    sub_dir += glob(root+'UCSF*/ses-01/Closed/'+channel+'/ImCOH*.mat')

# SUBJECT DICTIONARY
sub_dict = dict()
for iden in tqdm(sub_dir):
    if iden[28:35] not in sub_dict:
        sub_dict[iden[28:35]] = [iden]
    else:
        sub_dict[iden[28:35]].append(iden)

# TINNITUS/CONTROL DICTIONARY
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

# SPLIT TRAIN/VALIDATION DATA

# Data distribution
ratio = sum(list(ref.values())) / len(ref)

# Train/validation split ratio
split = 0.75

train_len = int(len(ref) * split)
train_pos, train_neg = int(train_len * ratio), int(train_len * (1 - ratio))
val_dir, train_dir = [], []
pos_count, neg_count = 0, 0

for k, v in tqdm(ref.items()):
    if v == True and pos_count < train_pos:
        pos_count += 1
        train_dir.append(sub_dict[k])
    elif v == False and neg_count < train_neg:
        neg_count += 1
        train_dir.append(sub_dict[k])
    else:
        val_dir.append(sub_dict[k])
    
print('Train/validation split', len(val_dir), len(train_dir))

# TRAINING/VALIDATION DATA AND LABELS
train_data = [np.array([])] * len(channels)
train_labels = np.array([])
val_data = [np.array([])] * len(channels)
val_labels = np.array([])

for pat in tqdm(range(len(train_dir))):
    for chan in range(len(train_dir[pat])):
        sub = train_dir[pat][chan][28:35]
        loading = sio.loadmat(train_dir[pat][chan])
        mtx = loading['icoh_val']
        np.fill_diagonal(mtx, 1)
        mtx = np.pad(mtx, 5, mode='constant')
        mtx = mtx.reshape((1, len(mtx[0]), len(mtx[1])))
        train_data[chan] = np.vstack([train_data[chan], mtx]) if train_data[chan].size else mtx
        if chan == 0:
            if ref[sub]:
                train_labels = np.append(train_labels, float(1))
            else:
                train_labels = np.append(train_labels, float(0))

for pat in tqdm(range(len(val_dir))):
    for chan in range(len(val_dir[pat])):
        sub = val_dir[pat][chan][28:35]
        loading = sio.loadmat(val_dir[pat][chan])
        mtx = loading['icoh_val']
        np.fill_diagonal(mtx, 1)
        mtx = np.pad(mtx, 5, mode='constant')
        mtx = mtx.reshape((1, len(mtx[0]), len(mtx[1])))
        val_data[chan] = np.vstack([val_data[chan], mtx]) if val_data[chan].size else mtx
        if chan == 0:
            if ref[sub]:
                val_labels = np.append(val_labels, float(1))
            else:
                val_labels = np.append(val_labels, float(0))

for channel in tdqm(train_data):
    print(channel.shape)
print(train_labels.shape)
for channel in tqdm(val_data):
    print(channel.shape)
print(val_labels.shape)
print('Data loaded')

# RESHAPING AND LOADING DATA
loaded_train_data, loaded_val_data = [], []

for channel in tqdm(train_data):
    # reshape the training data for the usage of dataloader
    train_x = channel.reshape(293, 1, 256, 256)
    train_x = torch.from_numpy(train_x)
    train_x = train_x.float()
    loaded_train_data.append(train_x)
    
for channel in tqdm(val_data):
    # reshape the training data for the usage of dataloader
    val_x = channel.reshape(100, 1, 256, 256)
    val_x = torch.from_numpy(val_x)
    val_x = val_x.float()
    loaded_val_data.append(val_x)
    
train_y = train_labels.astype(int);
train_y = torch.from_numpy(train_y)

val_y = val_labels.astype(int);
val_y = torch.from_numpy(val_y)

y_train = Variable(train_y)
y_val = Variable(val_y)


for channel in tqdm(loaded_train_data):
    print(channel.shape)
print(y_train.shape)
for channel in tqdm(loaded_val_data):
    print(channel.shape)
print(y_val.shape)
print('data reshaped')


# MODEL ARCHETECTURE
class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        #self.encoder3 = UNet._block(features * 2, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        
        # PREVIOUS LAYERS FOR FULL UNET
        
        #self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        #self.upconv3 = nn.ConvTranspose2d(
            #features * 8, features * 4, kernel_size=2, stride=2
        #)
        #self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        #self.upconv2 = nn.ConvTranspose2d(
            #features * 4, features * 2, kernel_size=2, stride=2
        #)
        #self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        #self.upconv1 = nn.ConvTranspose2d(
            #features * 2, features, kernel_size=2, stride=2
        #)
        #self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=32, out_channels=out_channels, kernel_size=1
        )
        
        self.linear_layers = Sequential(
            Linear(in_features=131072, out_features=400),
            ReLU(),
            Linear(in_features=400, out_features=2)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        #print(enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))
        #print(enc2.shape)
        enc3 = self.encoder3(self.pool2(enc2))
        #print(enc3.shape)
        enc4 = self.encoder4(self.pool3(enc3))
        #print(enc4.shape)

        bottleneck = self.bottleneck(self.pool3(enc4))
        
        # PREVIOUS LAYERS FOR FULL UNET
        
        #dec4 = self.upconv4(bottleneck)
        #dec4 = torch.cat((dec4, enc4), dim=1)
        #dec4 = self.decoder4(dec4)
        #print(dec4.shape)
        #dec3 = self.upconv3(dec4)
        #dec3 = torch.cat((dec3, enc3), dim=1)
        #dec3 = self.decoder3(dec3)
        #print(dec3.shape)
        #dec2 = self.upconv2(dec3)
        #dec2 = torch.cat((dec2, enc2), dim=1)
        #dec2 = self.decoder2(dec2)
        #print(dec2.shape)
        #dec1 = self.upconv1(dec2)
        #dec1 = torch.cat((dec1, enc1), dim=1)
        #dec1 = self.decoder1(dec1)
        #print(dec1.shape)
        
        x = torch.flatten(bottleneck,1)
        #print(x.size())
        x = self.linear_layers(x)
        
        #x = torch.sigmoid(x)
        #x = self.linear_layers(x)
        return x
        
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
print('UNet defined')
model = UNet()

# OPTIMIZER
optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.75)

# LOSS FUNCTION
criterion = CrossEntropyLoss()

# SCHEDULER
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.4)


# START TRAINING
print('start training')

# SAVE BASE MODEL WEIGHTS
base_model_wts = copy.deepcopy(model.state_dict())

# NUMBER OF EPOCHS
n_epochs = 10

train_losses, val_losses, train_acc, val_acc = [], [], [], []

# RUNNING BEST MODEL WEIGHTS
best_model_wts = copy.deepcopy(model.state_dict())

# TRAINING LOOP
for channel in tqdm(range(len(loaded_train_data))):
    print('channel: ' + channels[channel])
    model.load_state_dict(base_model_wts)
    x_train = Variable(loaded_train_data[channel])
    x_val = Variable(loaded_val_data[channel])
    channel_train_loss, channel_val_loss, channel_train_acc, channel_val_acc = [], [], [], []
    for epoch in tqdm(range(n_epochs)):    
        print('-' * 40)
        print('Epoch {}/{}'.format(epoch+1, n_epochs))
        
        model.train()
        optimizer.zero_grad()
        output_train = model(x_train)
        output_val = model(x_val)
        
        train_prob = list(output_train.data.numpy())
        train_predictions = np.argmax(train_prob, axis=1)
        
        val_prob = list(output_val.data.numpy())
        val_predictions = np.argmax(val_prob, axis=1)
        
        train_accuracy = accuracy_score(y_train, train_predictions) * 100
        val_accuracy = accuracy_score(y_val, val_predictions) * 100
        
        channel_train_acc.append(train_accuracy)
        channel_val_acc.append(val_accuracy)
        
        loss_train = criterion(output_train, y_train)
        loss_val = criterion(output_val, y_val)
        
        channel_train_loss.append(loss_train.item())
        channel_val_loss.append(loss_val.item())
        print(' train loss : ', loss_train, '\n', 'val loss : ', loss_val, '\n', 'train accuracy: ', train_accuracy, '\n', 'test accuracy: ', val_accuracy)
        if val_accuracy >= max(channel_val_acc):
            print('updating')
            best_model_wts = copy.deepcopy(model.state_dict())
        loss_train.backward()
        optimizer.step()
        #scheduler.step()
        
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'channel_' + channels[channel] + '_weights.pth')
    print('saved')
    train_acc.append(channel_train_acc)
    val_acc.append(channel_val_acc)
    train_losses.append(channel_train_loss)
    val_losses.append(channel_val_loss)
    

# PLOTS
for losses in range(len(train_losses)):
    plt.plot(train_losses[losses],label='channel ' + channels[losses] + ' training losses')
plt.legend(bbox_to_anchor=(0.07, 1))
plt.show()

for losses in range(len(val_losses)):
    plt.plot(val_losses[losses],label='channel ' + channels[losses] + ' validation losses')
plt.legend(bbox_to_anchor=(0.07, 1))
plt.show()

for accuracy in range(len(train_acc)):
    plt.plot(train_acc[accuracy],label='channel ' + channels[accuracy] + ' training accuracy')
plt.legend(bbox_to_anchor=(0.07, 1))
plt.show()

for accuracy in range(len(val_acc)):
    plt.plot(val_acc[accuracy],label='channel ' + channels[accuracy] + ' validation accuracy')
plt.legend(bbox_to_anchor=(0.07, 1))
plt.show()

# DATA
print(train_losses)
print(val_losses)
print(train_acc)
print(val_acc)


# LOADING PREVIOUS CHANNEL MODELS
alpha, beta, theta, delta_theta, delta, gamma, higam = UNet(), UNet(), UNet(), UNet(), UNet(), UNet(), UNet()

alpha.load_state_dict(torch.load('channel_alpha_weights.pth'))
beta.load_state_dict(torch.load('channel_beta_weights.pth'))
theta.load_state_dict(torch.load('channel_theta_weights.pth'))
delta_theta.load_state_dict(torch.load('channel_delta-theta_weights.pth'))
delta.load_state_dict(torch.load('channel_delta_weights.pth'))
gamma.load_state_dict(torch.load('channel_gamma_weights.pth'))
higam.load_state_dict(torch.load('channel_higam_weights.pth'))


# PARENT MODEL
class parentModel(nn.Module):
    def __init__(self, m1=alpha, m2=beta, m3=theta, m4=delta_theta, m5=delta, m6=gamma, m7=higam):
        super(parentModel, self).__init__()
        self.m1=m1
        self.m2=m2
        self.m3=m3
        self.m4=m4
        self.m5=m5
        self.m6=m6
        self.m7=m7
        
    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        l1 = self.m1(x1)
        l2 = self.m1(x2)
        l3 = self.m1(x3)
        l4 = self.m1(x4)
        l5 = self.m1(x5)
        l6 = self.m1(x6)
        l7 = self.m1(x7)
        
        x=torch.cat((l1, l2, l3, l4, l5, l6, l7))
        
        return x


# OUTPUTS
final_model = parentModel()
final_model.train()
final_output = final_model(
    Variable(loaded_val_data[0]),
    Variable(loaded_val_data[1]),
    Variable(loaded_val_data[2]),
    Variable(loaded_val_data[3]),
    Variable(loaded_val_data[4]),
    Variable(loaded_val_data[5]),
    Variable(loaded_val_data[6])
)


# CALCULATING FINAL VALIDATION
final_prob = list(final_output.data.numpy())
final_predictions = np.argmax(final_prob, axis=1)

fused_preds = []
for test in range(100):
    preds = []
    for add in range(0, 600, 100):
        preds.append(final_output[test + add].detach().numpy())
    if sum(preds) > 3:
        fused_preds.append(float(1))
    else:
        fused_preds.append(float(0))

final_accuracy = accuracy_score(y_val, fused_preds) * 100
print(final_accuracy)