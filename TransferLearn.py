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

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA available")

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

print('Data partitioned') #len(sets[0]), len(sets[1]), len(sets[2]))


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
#print(train_data[0][0].shape, train_label[0].shape)
#print(train_data[1][0].shape, train_label[1].shape)
#print(train_data[2][0].shape, train_label[2].shape)


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
                    
#print(len(training_data), len(training_data[0]), training_data[0][0].shape)
#print(len(training_label), training_label[1].shape)
#print(len(val_data), len(val_data[0]), val_data[0][0].shape)
#print(len(val_label), val_label[1].shape)        


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
        
        self.linear_layers = Sequential(
            Linear(in_features=2097152, out_features=500, ),
            ReLU(),
            Linear(in_features=500, out_features=2),
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

        bottleneck = self.bottleneck(self.pool4(enc4))
        
        x = torch.flatten(bottleneck,1)
        #print(x.size())
        x = self.linear_layers(x)
        
        #return torch.sigmoid(self.conv(dec1))
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
model = UNet()
model_dict = model.state_dict()
pretrained_dict = torch.load('third_full_unet_best_alpha_channel_weights.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(pretrained_dict)
model.linear_layers._modules['0'] = nn.Linear(in_features=131072, out_features=500, bias=True)
#for param in model.parameters():
    #param.requires_grad = False
print(model)
base_model_wts = copy.deepcopy(model.state_dict())
print('UNet defined')


# In[ ]:


#optimizer = Adam(model.parameters(), lr=0.001)

criterion = CrossEntropyLoss()
#criterion = MSELoss()

#scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    print('model to CUDA')

n_epochs = 40

min_max_avg = []
train_losses, val_losses, train_acc, val_acc = [], [], [], []
train_sens, train_spec, val_sens, val_spec = [], [], [], []

best_model_wts = copy.deepcopy(model.state_dict())


# In[ ]:


# training the model
print('start training')

for channel in tqdm(range(len(channels) - 6)):
    print('channel: ' + channels[channel])
    cross_final_acc, best_acc = [], None
    cross_train_acc, cross_val_acc, cross_train_loss, cross_val_loss, cross_train_sens, cross_val_sens, cross_train_spec, cross_val_spec = [], [], [], [], [], [], [], []
    for fold in range(num_folds):
        print('fold: ', fold)
        optimizer = Adam(model.parameters(), lr=0.0001)
        model.load_state_dict(base_model_wts)
        x_train = Variable(loaded_train_data[fold][channel]); x_val = Variable(loaded_val_data[fold][channel])
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            x_val = x_val.cuda()
            print('data to CUDA')
        #list_x_train = torch.tensor_split(x_train, 1)
        y_train = loaded_train_labels[fold]; y_val = loaded_val_labels[fold]
        if torch.cuda.is_available():
            y_train = y_train.cuda()
            y_val = y_val.cuda()
            print('labels to CUDA')
        #list_x_labels = torch.tensor_split(train_loss_labels[fold], 1)
        y_train_list, y_val_list = y_train.tolist(), y_val.tolist()
        fold_train_pos, fold_train_neg, fold_val_pos, fold_val_neg = y_train_list.count(1.), y_train_list.count(0.), y_val_list.count(1.), y_val_list.count(0.)
        fold_train_loss, fold_val_loss, fold_train_acc, fold_val_acc = [], [], [], []; fold_train_sens, fold_train_spec, fold_val_sens, fold_val_spec = [], [], [], []
        for epoch in tqdm(range(n_epochs)): 
            if True:
                print('-' * 40)
                print('Epoch {}/{}'.format(epoch+1, n_epochs))
            model.train(); optimizer.zero_grad()
            
            #running_loss, running_preds = 0, np.array([])
            
            # Batch training
            #for batch in range(len(list_x_train)):
                #output_train = model(list_x_train[batch])
                #train_prob = list(output_train.data.numpy())
                #train_predictions = np.argmax(train_prob, axis=1)
                #running_preds = np.concatenate([running_preds, train_predictions]) if running_preds.size else train_predictions
                #loss_train = criterion(output_train, list_x_labels[batch].float())
                #loss_train.backward()
                #optimizer.step()
                #running_loss += loss_train.item()
                
            #running_loss /= len(list_x_train)
            
            output_train = model(x_train)
            #print(output_train)
            output_val = model(x_val)
            train_prob = list(output_train.data.numpy())
            val_prob = list(output_val.data.numpy())
            train_predictions = np.argmax(train_prob, axis=1)
            val_predictions = np.argmax(val_prob, axis=1)
            train_preds_list, val_preds_list = train_predictions.tolist(), val_predictions.tolist()
            #train_preds_list, val_preds_list = running_preds.tolist(), val_predictions.tolist()
            train_tp, train_tf, val_tp, val_tf = 0, 0, 0, 0
            for i in range(len(train_preds_list)):
                if train_preds_list[i] == y_train_list[i] == 1:
                    train_tp += 1
                elif train_preds_list[i] == y_train_list[i] == 0:
                    train_tf += 1
                else:
                    continue
            for i in range(len(val_preds_list)):
                if val_preds_list[i] == y_val_list[i] == 1:
                    val_tp += 1
                elif val_preds_list[i] == y_val_list[i] == 0:
                    val_tf += 1
                else:
                    continue
            curr_train_sens = train_tp / fold_train_pos * 100; curr_train_spec = train_tf / fold_train_neg * 100
            curr_val_sens = val_tp / fold_val_pos * 100; curr_val_spec = val_tf / fold_val_neg * 100
            fold_train_sens.append(curr_train_sens); fold_train_spec.append(curr_train_spec); fold_val_sens.append(curr_val_sens); fold_val_spec.append(curr_val_spec)
            #train_accuracy = accuracy_score(y_train, running_preds) * 100
            train_accuracy = accuracy_score(y_train, train_predictions) * 100
            val_accuracy = accuracy_score(y_val, val_predictions) * 100
            if best_acc == None:
                best_acc = val_accuracy
            fold_train_acc.append(train_accuracy); fold_val_acc.append(val_accuracy)
            loss_train = criterion(output_train, y_train)
            #print(loss_train)
            loss_val = criterion(output_val, y_val)
            fold_train_loss.append(loss_train.item()); fold_val_loss.append(loss_val.item())
            print(
                ' t_loss : ', loss_train.data.numpy(), '\t', 'v_loss : ', loss_val.data.numpy(), '\n', 't_acc: ', train_accuracy, '\t',  'v_acc: ',
                val_accuracy, '\n', 't_sens: ', curr_train_sens, '\t', 'v_sens: ', curr_val_sens, '\n', 't_spec: ', curr_train_spec, '\t', 'v_spec: ', curr_val_spec)
            if val_accuracy >= max(fold_val_acc):
                best_acc = val_accuracy
                print('updating')
                best_model_wts = copy.deepcopy(model.state_dict())
            loss_train.backward(); #scheduler.step()
            optimizer.step()
        model.load_state_dict(best_model_wts)
        temp_output = model(x_val)
        temp_prob = list(temp_output.data.numpy())
        temp_predictions = np.argmax(temp_prob, axis=1)
        temp_preds_list = temp_predictions.tolist()
        temp_tp, temp_tf = 0, 0
        for i in range(len(temp_preds_list)):
            if temp_preds_list[i] == y_val_list[i] == 1:
                temp_tp += 1
            elif temp_preds_list[i] == y_val_list[i] == 0:
                temp_tf += 1
            else:
                continue
        temp_sens, temp_spec = temp_tp / fold_val_pos * 100, temp_tf / fold_val_neg * 100
        temp_accuracy = accuracy_score(y_val, temp_predictions) * 100
        if len(cross_final_acc) == 0:
            print('new best model')
            torch.save(model.state_dict(), 'half_unet_' + channels[channel] + '_best_channel_weights.pth')
        elif temp_accuracy >= max(cross_final_acc):
            print('new best model')
            torch.save(model.state_dict(), 'half_unet_' + channels[channel] + '_best_channel_weights.pth')
        cross_final_acc.append(temp_accuracy)
        print(' best accuracy: ', '\t', temp_accuracy, '\n', 'best sensitivity: ', '\t', temp_sens, '\n', 'best specificity: ', '\t', temp_spec, '\n')
        #torch.save(model.state_dict(), 'full_unet_' + channels[channel] + '_temp_channel_weights.pth')
        print('saved')
        cross_train_acc.append(fold_train_acc); cross_val_acc.append(fold_val_acc); cross_train_loss.append(fold_train_loss); cross_val_loss.append(fold_val_loss)
        cross_train_sens.append(fold_train_sens); cross_train_spec.append(fold_train_spec); cross_val_sens.append(fold_val_sens); cross_val_spec.append(fold_val_spec)
    min_max_avg.append(cross_final_acc); train_losses.append(cross_train_loss); val_losses.append(cross_val_loss); train_acc.append(cross_train_acc); val_acc.append(cross_val_acc)
    train_sens.append(cross_train_sens); train_spec.append(cross_train_spec); val_sens.append(cross_val_sens); val_spec.append(cross_val_spec)


# In[ ]:


# plot the learning curves
for channel in range(len(train_losses)):
    for fold in range(num_folds):
        plt.plot(train_losses[channel][fold],label='channel ' + channels[channel] + ' training losses: fold ' + str(fold))
plt.legend(loc='best'); plt.title("Training losses"); plt.ylabel("Loss value"); plt.xlabel("Epoch number")  ; plt.show()

for channel in range(len(train_acc)):
    for fold in range(num_folds):
        plt.plot(train_acc[channel][fold],label='channel ' + channels[channel] + ' training accuracy: fold ' + str(fold))
plt.legend(loc='best'); plt.title("Training accuracies"); plt.ylabel("Percent (% accuracy)"); plt.xlabel("Epoch number") ; plt.show()

for channel in range(len(train_sens)):
    for fold in range(num_folds):
        plt.plot(train_sens[channel][fold],label='channel ' + channels[channel] + ' training sensitivity: fold ' + str(fold))
plt.legend(loc='best'); plt.title("Training sensitivities"); plt.ylabel("Percent (% sensitivity)"); plt.xlabel("Epoch number") ; plt.show()

for channel in range(len(train_spec)):
    for fold in range(num_folds):
        plt.plot(train_spec[channel][fold],label='channel ' + channels[channel] + ' training specificity: fold ' + str(fold))
plt.legend(loc='best'); plt.title("Training specificities"); plt.ylabel("Percent (% specificity)"); plt.xlabel("Epoch number") ; plt.show()

for channel in range(len(val_losses)):
    for fold in range(num_folds):
        plt.plot(val_losses[channel][fold],label='channel ' + channels[channel] + ' validation losses: fold ' + str(fold))
plt.legend(loc='best'); plt.title("Validation losses"); plt.ylabel("Loss value"); plt.xlabel("Epoch number") ; plt.show()

for channel in range(len(val_acc)):
    for fold in range(num_folds):
        plt.plot(val_acc[channel][fold],label='channel ' + channels[channel] + ' validation accuracy: fold ' + str(fold))
plt.legend(loc='best'); plt.title("Validation accuracies"); plt.ylabel("Percent (% accuracy)"); plt.xlabel("Epoch number") ; plt.show()

for channel in range(len(val_sens)):
    for fold in range(num_folds):
        plt.plot(val_sens[channel][fold],label='channel ' + channels[channel] + ' training sensitivity: fold ' + str(fold))
plt.legend(loc='best'); plt.title("Validation sensitivities"); plt.ylabel("Percent (% sensitivity)"); plt.xlabel("Epoch number") ; plt.show()

for channel in range(len(val_spec)):
    for fold in range(num_folds):
        plt.plot(val_spec[channel][fold],label='channel ' + channels[channel] + ' training specificity: fold ' + str(fold))
plt.legend(loc='best'); plt.title("Validation specificty"); plt.ylabel("Percent (% specificity)"); plt.xlabel("Epoch number") ; plt.show()


# In[ ]:


print('Train loss:'); print(train_losses); print('Validation loss:'); print(val_losses); print('Train accuracy:'); print(train_acc); print('Validation accuracy:'); print(val_acc)
print('Train sensitivity:'); print(train_sens); print('Train specificity:'); print(train_spec); print('Validation sensitivity:'); print(val_sens); print('Validation specificity:'); print(val_spec)


# In[ ]:


final_acc = []
for channel in val_acc:
    for fold in channel:
        final_acc.append(max(fold))
print('average accuracy: {}\nmaximum accuracy: {} \nminimum accuracy: {}'.format(mean(final_acc), max(final_acc), min(final_acc)))


# In[ ]:




