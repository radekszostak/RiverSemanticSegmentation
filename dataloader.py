from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms
import os
import cv2
import numpy as np
import torch

class Dataset(BaseDataset):
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            input_size,
            output_size,
            count=-1
    ):
        self.count = count
        self.input_size = (input_size, input_size)
        self.output_size = (output_size, output_size)
        self.ids = [file_name.split(".")[0] for file_name in os.listdir(images_dir)]
        if count>0:
          self.ids = self.ids[:count]
        self.images_fps = [os.path.join(images_dir, image_id + ".jpg") for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id + ".png") for image_id in self.ids]
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        ])
   
    def __getitem__(self, i):

        image = cv2.imread(self.images_fps[i])
        image = cv2.resize(image, self.input_size, interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image)

        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, self.output_size, interpolation = cv2.INTER_NEAREST)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask, 0)
        
        return image, mask
        
    def __len__(self):
        return len(self.ids)