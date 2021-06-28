import os
os.chdir("C:/Users/marci/RiverSemanticSegmentation/")
dataset_dir = os.path.normpath("C:/Users/marcin/sentinel-river-segmentation-dataset/")

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

from tqdm import tqdm

from models.unet import UNet
from models.simple import Simple

from dataloader import Dataset

PARAMS = {
    "input_size": 416,
    "model": "vgg_unet",
    "learning_rate": 0.001,
    "output_size": 416,
    "nr_classes": 2 
}

x_train_dir = os.path.join(dataset_dir,"x_train")
y_train_dir = os.path.join(dataset_dir,"y_train")
x_test_dir = os.path.join(dataset_dir,"x_test")
y_test_dir = os.path.join(dataset_dir,"y_test")

train_set = Dataset(x_train_dir, y_train_dir, PARAMS["input_size"], PARAMS["input_size"], PARAMS["nr_classes"])
test_set = Dataset(x_test_dir, y_test_dir, PARAMS["input_size"], PARAMS["input_size"], PARAMS["nr_classes"])
batch_size = 10
dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

if PARAMS['model'] == "simple":
  from models.simple import Simple
  model = Simple()
elif PARAMS['model'] == "vgg_unet":
  from models.vgg_unet import VggUnet
  model = VggUnet()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
summary(model, input_size=(batch_size, 3, PARAMS['input_size'], PARAMS['input_size']))
model = torch.load("model.obj")

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

import matplotlib.pyplot as plt
def plot_side_by_side(rgb,ground_truth,predict):
  assert rgb.shape[0] == ground_truth.shape[0] == predict.shape[0]
  batch_size = rgb.shape[0]
  fig, axs = plt.subplots(batch_size, 3, figsize=(30,50))
  for i in range(batch_size):
    print(ground_truth[i].min())
    print(predict[i].min())
    axs[i, 0].imshow(rgb[i])
    axs[i, 1].imshow(ground_truth[i])
    axs[i, 2].imshow(predict[i])

import math
#import helper
model.eval()   # Set model to evaluate mode
test_dataset = Dataset(x_test_dir, y_test_dir)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True, num_workers=0)
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

labels = labels.data.cpu().numpy()
pred = model(inputs)
pred = torch.sigmoid(pred)
pred = pred.data.cpu().numpy()
inputs = inputs.data.cpu()
#print(inputs.shape)
#print(labels)
#print(torch.sigmoid(torch.from_numpy(pred)).round())

# Change channel-order and make 3 channels for matplot
input_images_rgb = reverse_transform(inputs)#[reverse_transform(x) for x in inputs.cpu()]
#print(input_images_rgb.shape)
# Map each channel (i.e. class) to each color
target_masks_rgb = np.squeeze(labels)#[helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
#print(target_masks_rgb.shape)
pred_rgb = np.squeeze(pred)#[helper.masks_to_colorimg(x) for x in pred]
#print(input_images_rgb)
plot_side_by_side(input_images_rgb, target_masks_rgb, pred_rgb)

