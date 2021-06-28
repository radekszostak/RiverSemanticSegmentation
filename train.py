import os
#os.chdir("C:/Users/marci/RiverSemanticSegmentation/")

#imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataloader import Dataset
from torchinfo import summary
import time
import copy
import pdb
from tqdm import tqdm
import evolution
from models.attention_based import AttentionBased
import torchvision

PARAMS = {
    "input_size": 416,
    "output_size": 416,
    "model": "vgg_unet",
    "learning_rate": 0.0001,
    "batch_size": 8,
    'epochs': 80,
    'patience': 10,
    "train_dataset_size": -1, # set train dataset subset. Useful when neet to 
                              # overtrain model with small amount of images.
                              # -1 -all images from train directories.
    "test_dataset_size": -1,  # set test dataset subset.
                              # -1 -all images from train directories.
    "n_classes": 2,
    'image_preload': False,
}

import configparser
config = configparser.ConfigParser()
config.read("./config.cfg")

dataset_dir = os.path.normpath("/work/sentinel-river-segmentation-dataset-main")
x_train_dir = os.path.join(dataset_dir,"x_train")
y_train_dir = os.path.join(dataset_dir,"y_train")
x_test_dir = os.path.join(dataset_dir,"x_test")
y_test_dir = os.path.join(dataset_dir,"y_test")

train_set = Dataset(x_train_dir, y_train_dir, input_size=PARAMS['input_size'], output_size=PARAMS['output_size'], n_classes=PARAMS["n_classes"], count=PARAMS["train_dataset_size"])
test_set = Dataset(x_test_dir, y_test_dir, input_size=PARAMS['input_size'], output_size=PARAMS['output_size'], n_classes=PARAMS["n_classes"], count=PARAMS["test_dataset_size"])

batch_size = PARAMS['batch_size']
dataloaders = {
    'train': DataLoader(train_set, batch_size=PARAMS['batch_size'], shuffle=True, num_workers=0),
    'val': DataLoader(test_set, batch_size=PARAMS['batch_size'], shuffle=True, num_workers=0)
}

# load images - useful if you want to save some time by preloading images (very time-consuming) when 
# the model is still not fuctional and cant run standard training.
if PARAMS['image_preload']:
    for phase in dataloaders:
        for inputs, labels in tqdm(dataloaders[phase]):
            pass

#model loading
if PARAMS['model'] == "simple":
    from models.simple import Simple
    model = Simple()
elif PARAMS['model'] == "vgg_unet":
    from models.vgg_unet import VggUnet
    model = VggUnet()
elif PARAMS['model'] == "vgg_unet_ks":
    from models.vgg_unet_ks import VggUnetKs
    model = VggUnetKs()
elif PARAMS['model'] == "unet":
    from models.unet import UNet
    model = UNet()
elif PARAMS['model'] == "vgg_deconvnet":
    from models.vgg16_deconvnet import VggDeconvNet
    model = VggDeconvNet()

#model structure preview
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
#model_stats = summary(model, input_size=(PARAMS['batch_size'], 3, PARAMS['input_size'], PARAMS['input_size']))


from collections import defaultdict
import torch.nn.functional as F
SMOOTH = 1e-6

def iou_metric(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs[:,1,:,:]  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels[:,1,:,:]
    intersection = (outputs * labels).sum(2).sum(1)  # Will be zero if Truth=0 or Prediction=0
    union = (outputs + labels).sum(2).sum(1) - intersection  # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    return iou.mean()
    

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy(pred, target)
    pred = torch.round(pred)
    dice = dice_loss(pred, target)
    loss = bce# * bce_weight + dice * (1 - bce_weight)
    iou = iou_metric(pred, target)
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    metrics['iou'] += iou.data.cpu().numpy() * target.size(0)
    return loss


def print_metrics(metrics, epoch_samples, phase):   
    print(epoch_samples) 
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, dataloaders, optimizer, device, num_epochs=25, patience=-1):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0
    no_improvement = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    loss = calc_loss(outputs, labels, metrics)
                    #print(model.encoder[0].weight.grad)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        #pdb.set_trace()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
                #pdb.set_trace()

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            epoch_accuracy = metrics['iou'] / epoch_samples

            # deep copy the model
            if phase == 'val':
              if epoch_accuracy > best_accuracy:
                no_improvement = 0
                print("Val IoU improved by {}. Saving best model.".format(epoch_accuracy-best_accuracy))
                best_accuracy = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
              else:
                no_improvement += 1
                print("No accuracy improvement since {}/{} epochs.".format(no_improvement,patience))
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if patience >= 0 and no_improvement > patience:
          break
    print('Best accuracy: {:4f}'.format(best_accuracy))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, testloader):
    intersection=0
    union=0

    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.data.cpu().numpy()
        pred = model(inputs)
        pred = torch.round(pred)
        pred = pred.data.cpu().numpy()
        target = labels[:,1,:,:]
        predict = pred[:,1,:,:]
        temp = (target * predict).sum()
        intersection+=temp
        union+=((target + predict).sum() - temp)

    iou = intersection/union
    print("IoU: {}".format(iou))


models, _ = evolution.generate_models(10)
model_ = evolution.generate_model()

model = AttentionBased(2, 4)
model.cuda()
#pdb.set_trace()

#model training
optimizer_ft = optim.Adam(model.parameters(), lr=PARAMS['learning_rate'])
#optimizer_ft = optim.Adam(models['0'].parameters(), lr=PARAMS['learning_rate'])
model = train_model(model, dataloaders, optimizer_ft, device, num_epochs=PARAMS['epochs'], patience=PARAMS['patience'])
# save weights
torch.save(model.state_dict(), "state_dict.pth")

model.load_state_dict(torch.load("state_dict.pth", map_location="cpu"))

from torchvision import transforms
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

def reverse_transform(inp):
    print(inp.shape)
    inp = inv_normalize(inp)
    inp = inp.numpy()
    inp = np.swapaxes(inp, 1, 3)
    inp = np.swapaxes(inp, 1, 2)
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    
    return inp

def labels2mask(labels):
    return labels[:,1,:,:]

# helper function to plot input, ground truth and predict images in grid
import matplotlib.pyplot as plt
def plot_side_by_side(rgb,ground_truth,predict):
    assert rgb.shape[0] == ground_truth.shape[0] == predict.shape[0]
    batch_size = rgb.shape[0]
    fig, axs = plt.subplots(batch_size, 3, figsize=(30,50))
    for i in range(batch_size):
        axs[i, 0].imshow(rgb[i])
        axs[i, 1].imshow(ground_truth[i])
        axs[i, 2].imshow(predict[i])

import math
model.eval()   # Set model to evaluate mode
test_dataset = Dataset(x_test_dir, y_test_dir, input_size=PARAMS['input_size'], output_size=PARAMS['output_size'], n_classes=PARAMS['n_classes'])
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=True, num_workers=0)
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

#preprocess = torchvision.transforms.Resize((208, 208))
#transform_im = torchvision.transforms.ToPILImage()
#transform_te = torchvision.transforms.ToTensor()
#inputs = transform_te(preprocess(transform_im(inputs[0].cpu())))

#pdb.set_trace()

#for i in range(len(models.keys())):
#     print(i)
#     pred = models[str(i)].cpu()(inputs.cpu())

#print(model_)
#pred = model_.cuda()(inputs.cuda())
#pred = ab.cuda()(inputs.cuda())
#pdb.set_trace()

labels = labels.data.cpu().numpy()
pred = model(inputs)

#print(pred)
pred = torch.round(pred)
#print(pred.size())

pred = pred.data.cpu().numpy()
inputs = inputs.data.cpu()

# dataloader return normalized input image, so we have to denormalize before viewing
#input_images = reverse_transform(inputs)
# target and predict mask are single channel, so squeeze
#target_masks = labels2mask(labels)

#pred = labels2mask(pred)

# use helper function to plot
#plot_side_by_side(input_images, target_masks, pred)

test_dataset = Dataset(x_test_dir, y_test_dir, input_size=PARAMS['input_size'], output_size=PARAMS['output_size'], n_classes=PARAMS['n_classes'])
test_loader = DataLoader(test_dataset, batch_size=PARAMS["batch_size"], shuffle=True, num_workers=0)


intersection=0
union=0

for inputs, labels in tqdm(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    labels = labels.data.cpu().numpy()
    pred = model(inputs)
    pred = torch.round(pred)
    pred = pred.data.cpu().numpy()
    target = labels[:,1,:,:]
    predict = pred[:,1,:,:]
    temp = (target * predict).sum()
    intersection+=temp
    union+=((target + predict).sum() - temp)

iou = intersection/union
print("IoU: {}".format(iou))

#setup resnet50, attention and graph NN
