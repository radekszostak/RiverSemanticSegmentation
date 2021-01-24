from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms
import os
import cv2
import numpy as np
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    #
    #CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
    #           'tree', 'signsymbol', 'fence', 'car', 
    #           'pedestrian', 'bicyclist', 'unlabelled']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
    ):
        self.ids = [file_name.split(".")[0] for file_name in os.listdir(images_dir)]
        self.images_fps = [os.path.join(images_dir, image_id + ".jpg") for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id + ".png") for image_id in self.ids]
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        ## convert str names to class values on masks
        #self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
   
    def __getitem__(self, i):
        size = (416,416)
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, size, interpolation = cv2.INTER_NEAREST)
        ## extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        return image, mask
        
    def __len__(self):
        return len(self.ids)