import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils
import torch
import os
import PIL.Image as Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import accuracy_score, f1_score
import time
from torch import optim,nn
from torchvision import models
from torch.autograd import Variable
from tqdm import *

from IPython.display import FileLink

use_gpu = True
train_folder = 'data/train_/'
test_folder = 'data/test_/'
data_folder = 'data/'

class EqHist(object):

    def __call__(self, img):
        
        img = np.asarray(img)
#         print(type(img))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(img)
        img2 = np.zeros((eq.shape[0],eq.shape[1],3))
        img2[:,:,0] = eq
        img2[:,:,1] = eq
        img2[:,:,2] = eq
        return img2

class add_gausian_noise(object):

    def __call__(self, img):
        
        img = np.asarray(img)
        noise = np.random.normal(size = (img.shape))
        img += noise
        return img
    
    


class XRayDataset(Dataset):
    
    def __init__(self, csv_file, path, transform = None,is_train=True, subset=700):
        self.df = pd.read_csv(csv_file)
        
        self.path = path
        self.transform = transform
        self.is_train = is_train
        
        if is_train:
            self.df['detected_id'] = self.df.detected.astype('category').cat.codes
            self.idx_to_classes = dict(enumerate(self.df.detected.astype('category').cat.categories))
            self.subset = subset
            self.class_weights = compute_class_weight('balanced',np.arange(14),self.df.detected.astype('category').cat.codes)
            indices = np.arange(len(self.df))
            tmp,self.test_idx =  train_test_split(indices,test_size=0.2,random_state=42)
            self.tr_idx, self.val_idx = train_test_split(tmp, test_size=0.25,random_state=42)
            self.sml_tr_idx = np.random.choice(self.tr_idx,size=self.subset,replace=False)
            self.sml_val_idx = np.random.choice(self.val_idx,size=int(self.subset * 0.2),replace=False)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self,idx):
        
        if self.is_train:
            img_path = os.path.join(self.path,'train_',self.df.loc[idx,'image_name'])
        else:
            img_path = os.path.join(self.path,'test_',self.df.loc[idx,'image_name'])
        
        img = cv2.imread(img_path,1)
        if self.is_train:
            label = self.df.loc[idx,'detected_id'] # detected id is cat converted to numbers
            sample = {'image':img,
                  'label':label
                 }
        else:
            sample = {'image':img,
                      'label':0
                     }
            
        
        if self.transform:
            if self.is_train:
                if idx in self.val_idx or idx in self.test_idx:
                    sample['image'] = self.transform['test_aug'](sample['image'])
                else:
                    sample['image'] = self.transform['train_aug'](sample['image'])
            else:
                sample['image'] = self.transform['test_aug'](sample['image'])
            
        return sample   

def get_tfms(sz):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tfms = {    
        'train_aug': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((sz+20,sz+20)),
            transforms.CenterCrop(sz),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.05),
            EqHist(),
            add_gausian_noise(),
#             transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize

        ]),

        'test_aug': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((sz,sz)),
            EqHist(),
    #         transforms.CenterCrop(227), #will try this later
            transforms.ToTensor(),
            normalize
        ]),

        'no_aug' : transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((sz,sz)),

        ])
    }
    
    return tfms

def get_pm_for_dl():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        num_workers = 8 #stupid code
    else:
        num_workers = 4
    return use_gpu, num_workers

def get_dataloaders(dataset,bs,num_workers,use_gpu):
    train_sampler = sampler.SubsetRandomSampler(dataset.tr_idx)
    val_sampler = sampler.SubsetRandomSampler(dataset.val_idx)
    test_sampler = sampler.SubsetRandomSampler(dataset.test_idx)
    small_train_sampler = sampler.SubsetRandomSampler(dataset.sml_tr_idx)
    small_val_sampler = sampler.SubsetRandomSampler(dataset.sml_val_idx)


    samplers = {
        'train':train_sampler,
        'val': val_sampler,
        'test':test_sampler,
        'sml_tr':small_train_sampler,
        'sml_val':small_val_sampler
    }

    dataloaders = {k : DataLoader(dataset,
                            batch_size=bs,
                            sampler=v,
                            num_workers=num_workers,
                            pin_memory=use_gpu
                           ) for k,v in samplers.items()}

    
    return samplers, dataloaders

def get_dt_szs(samplers):
    dataset_sizes = {k : len(v) for k,v in samplers.items()}
    return dataset_sizes


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, data_split = ['train','val'],name='',save_model=True):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    overall_train_losses = []
    overall_val_losses = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in data_split:
            if phase == 'train' or phase == 'sml_tr':
                print('if ' + phase)
                if scheduler:
                    scheduler.step()
                model.train(True)  # Set model to training mode
            elif phase == 'val' or phase == 'sml_val':
                print('else ' + phase)
                model.train(False)  # Set model to evaluate mode
            else:
                print(phase + ' not allowed')
                

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data['image'],data['label']
                labels = labels.type(torch.LongTensor)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
#                 print(labels.size(),outputs.size())
                loss = criterion(outputs, labels)
                    
                if phase == 'train' or phase == 'sml_tr':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
#                 print('batch passed')
                

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            

#             deep copy the model
            if (phase == 'val' or phase == 'sml_val') and epoch_acc > best_acc:
                print('setting best')
                best_acc = epoch_acc
                best_model_wts = model.state_dict()          

    if save_model:
        torch.save(best_model_wts,f'./models/model_{name}_{best_acc}')
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    return model,best_acc,best_model_wts

def change_to_cuda(model):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        return model.cuda()
    else:
        return model    
    
def predict(model, dset,dataloaders,dataset_sizes,criterion):
    
    model.train(False)
    overall_preds = []
    overall_labels = []
    for epoch in range(1): # lazy loop so that I do not have to change indentation
        for phase in [dset]: # lazy again
            print(dset)
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs, labels = data['image'],data['label']

                labels = labels.type(torch.LongTensor)
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                overall_preds += list(preds)
                overall_labels += list(labels.data)
                
                loss = criterion(outputs, labels)
                
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
    return overall_preds,overall_labels   

def lr_train_model(model, criterion, optimizer, scheduler, num_epochs=25, data_split = ['train','val'], name='', save_model = True):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    overall_train_losses = []
    overall_val_losses = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in data_split:
            if phase == 'train' or phase == 'sml_tr': # change here
#                 print(phase)               
                model.train(True)  # Set model to training mode
            elif phase == 'val' or phase == 'sml_val' or phase == 'test':
#                 print('else ' + phase)
                model.train(False)  # Set model to evaluate mode
            else:
                print(phase + ' not allowed')

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs, labels = data['image'],data['label']
                labels = labels.type(torch.LongTensor)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                if phase == 'train' or phase == 'sml_tr':
#                     print(phase)
                    loss.backward()
                    optimizer.step()



                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
#                 print('batch passed')
                

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            if phase == 'val' or 'sml_val':
                scheduler.step(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            

#             deep copy the model
            if (phase == 'val' or phase == 'sml_val') and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                

            

                

#         print()
    if save_model:
        torch.save(best_model_wts,f'./models/model_{name}{best_acc}')
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,overall_train_losses,overall_val_losses

def get_subm_link(criterion):
    test_dataset = XRayDataset(f'{data_folder}test.csv',f'{data_folder}',transform=tfms,is_train=False)


    test_samplers = {'test':sampler.SequentialSampler(test_dataset)}
    test_dataloaders = {'test':DataLoader(test_dataset,batch_size=32,sampler=test_samplers['test'],num_workers=8,pin_memory=True)}
    test_dt_szs = get_dt_szs(test_samplers)

    # use_gpu = False
    criterion = criterion
    t_pdted,t_lbs = predict(dropout_model,'test',test_dataloaders,test_dt_szs)

    print(np.bincount(t_pdted))


    test_df = pd.read_csv(f'{data_folder}test.csv')
    test_df.head()

    test_df['detected'] = pd.Series([transformed_dataset.idx_to_classes[i] for i in t_pdted]).astype('category')

    test_df.drop(['age','gender','view_position','image_name'],axis=1).to_csv('sdir/fst.csv',index=False)

    return FileLink('./sdir/fst.csv')

def get_subm_link(model,criterion,tfms,transformed_dataset):
    test_dataset = XRayDataset(f'{data_folder}test.csv',f'{data_folder}',transform=tfms,is_train=False)


    test_samplers = {'test':sampler.SequentialSampler(test_dataset)}
    test_dataloaders = {'test':DataLoader(test_dataset,batch_size=32,sampler=test_samplers['test'],num_workers=1,pin_memory=True)}
    test_dt_szs = get_dt_szs(test_samplers)

    # use_gpu = False
    criterion = criterion
    t_pdted,t_lbs = predict(model,'test',test_dataloaders,test_dt_szs,criterion=criterion)

    np.bincount(t_pdted)


    test_df = pd.read_csv(f'{data_folder}test.csv')
    test_df.head()

    test_df['detected'] = pd.Series([transformed_dataset.idx_to_classes[i] for i in t_pdted]).astype('category')

    test_df.drop(['age','gender','view_position','image_name'],axis=1).to_csv('sdir/fst.csv',index=False)

    return FileLink('./sdir/fst.csv')

    
def merge_dataloaders(dataset):
    train_sampler = sampler.SubsetRandomSampler(np.concatenate((dataset.val_idx,dataset.test_idx)))


    samplers = {
        'train':train_sampler
    }

    dataloaders = {k : DataLoader(dataset,
                            batch_size=bs,
                            sampler=v,
                            num_workers=num_workers,
                            pin_memory=use_gpu
                           ) for k,v in samplers.items()}

    
    return samplers, dataloaders

def show_sample(img,label = None,mapping=None):
    plt.imshow(img)
    if label:
        plt.title("label={} and class_name={}".format(label,mapping[label]))
