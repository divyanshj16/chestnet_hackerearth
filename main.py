from chestnet_utils import *

_splits_ = ['sml_tr','sml_val']
ne = 1
sm = False
decay = 0.7

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
                print(phase)
                if scheduler:
                    scheduler.step()
                model.train(True)  # Set model to training mode
            elif phase == 'val' or phase == 'sml_val':
                print(phase)
                model.train(False)  # Set model to evaluate mode
            else:
                print(phase + ' not allowed')                

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data['image'],data['label']
                labels = labels.type(torch.LongTensor)
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                    
                if phase == 'train' or phase == 'sml_tr':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' or phase == 'sml_val' and epoch_acc > best_acc:
                print('setting best')
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                if save_model:
                    torch.save(best_model_wts,f'./models/model_{name}{epoch}/{num_epochs}_{best_acc}')
    


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    return model,best_acc,best_model_wts


model_ft = models.resnet34(pretrained=True)
for i,param in enumerate(model_ft.parameters()):
    if i ==  5:
        break
    param.requires_grad = False
model_ft.conv1.kernel_size = (3,3)
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
model_ft.fc = nn.Sequential(nn.Dropout(p=0.5,inplace=True),nn.Linear(512,14))
model_ft = change_to_cuda(model_ft)

tfms = get_tfms(64)
transformed_dataset = XRayDataset(f'{data_folder}train.csv',f'{data_folder}',transform=tfms)
use_gpu, num_workers = get_pm_for_dl()
samplers, dataloaders = get_dataloaders(transformed_dataset,64,num_workers,use_gpu)
dataset_sizes = get_dt_szs(samplers)
    

model_ft.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=decay)
model_ft,bacc,bwt = train_model(model_ft, criterion, optimizer_ft, scheduler, dataloaders, dataset_sizes, num_epochs=ne, data_split=_splits_,save_model=sm, name = '_epoch_64_')

tfms = get_tfms(128)
transformed_dataset = XRayDataset(f'{data_folder}train.csv',f'{data_folder}',transform=tfms)
use_gpu, num_workers = get_pm_for_dl()
samplers, dataloaders = get_dataloaders(transformed_dataset,32,num_workers,use_gpu)
dataset_sizes = get_dt_szs(samplers)

model_ft.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=decay)
model_ft,bacc,bwt = train_model(model_ft, criterion, optimizer_ft, scheduler, dataloaders, dataset_sizes, num_epochs=ne, data_split=_splits_,save_model=sm, name = '_epoch_128_')

tfms = get_tfms(180)
transformed_dataset = XRayDataset(f'{data_folder}train.csv',f'{data_folder}',transform=tfms)
use_gpu, num_workers = get_pm_for_dl()
samplers, dataloaders = get_dataloaders(transformed_dataset,32,num_workers,use_gpu)
dataset_sizes = get_dt_szs(samplers)

model_ft.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=decay)
model_ft,bacc,bwt = train_model(model_ft, criterion, optimizer_ft, scheduler, dataloaders, dataset_sizes, num_epochs=ne, data_split=_splits_,save_model=sm, name = '_epoch_180_')


tfms = get_tfms(224)
transformed_dataset = XRayDataset(f'{data_folder}train.csv',f'{data_folder}',transform=tfms)
use_gpu, num_workers = get_pm_for_dl()
samplers, dataloaders = get_dataloaders(transformed_dataset,16,num_workers,use_gpu)
dataset_sizes = get_dt_szs(samplers)

model_ft.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=decay)
model_ft,bacc,bwt = train_model(model_ft, criterion, optimizer_ft, scheduler, dataloaders, dataset_sizes, num_epochs=ne, data_split=_splits_,save_model=sm, name = '_epoch_224_')










