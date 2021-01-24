import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataloader import Dataset
from torchinfo import summary
import time
import copy

from models.unet import UNet
from models.simple import Simple
from models.vgg_unet import VggUnet
from train_helper import *
from dataloader import Dataset

x_train_dir = os.path.join("dataset","x_train")
y_train_dir = os.path.join("dataset","y_train")
x_test_dir = os.path.join("dataset","x_test")
y_test_dir = os.path.join("dataset","y_test")

train_set = Dataset(x_train_dir, y_train_dir)
test_set = Dataset(x_test_dir, y_test_dir)

batch_size = 2

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VggUnet()
model = model.to(device)

summary(model, input_size=(batch_size, 3, 416, 416))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = VggUnet().to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

model = train_model(model, dataloaders, optimizer_ft, device, num_epochs=40)

