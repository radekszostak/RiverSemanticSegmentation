import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

vgg16_pretrained = models.vgg16(pretrained=True)

class VggDeconvNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            # conv1
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv2
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv3
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv4
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv5
            #torch.nn.ReLU(),
            #torch.nn.Conv2d(512, 512, 3, padding=1),
            #torch.nn.ReLU(),
            #torch.nn.Conv2d(512, 512, 3, padding=1),
            #torch.nn.Conv2d(512, 512, 3, padding=1),
            #torch.nn.ReLU(),
            #torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        )

        # declare pool indices dictionary
        self.pool_indices = dict()

        # initialize weights
        for i in range(len(self.encoder)):
            if isinstance(self.encoder[i], torch.nn.Conv2d):
                self.encoder[i].weight.data = vgg16_pretrained.features[i].weight.data
                self.encoder[i].bias.data = vgg16_pretrained.features[i].bias.data
        
        self.bottom = nn.Sequential(
          nn.Conv2d(512, 512, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(512, 512, 3, padding=1),
          nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
          #deconv5
          #nn.MaxUnpool2d(2, stride=2),
          #nn.ConvTranspose2d(512, 512, 3, padding=1),
          #nn.ReLU(inplace=True),
          #nn.ConvTranspose2d(512, 512, 3, padding=1),
          #nn.ReLU(inplace=True),
          #nn.ConvTranspose2d(512, 512, 3, padding=1),
          #nn.ReLU(inplace=True),
          #nn.BatchNorm2d(512),
          #deconv4
          nn.MaxUnpool2d(2, stride=2),
          nn.ConvTranspose2d(512, 512, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(512, 512, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(512, 256, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(256),
          #deconv3
          nn.MaxUnpool2d(2, stride=2),
          nn.ConvTranspose2d(256, 256, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(256, 256, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(256, 128, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(128),
          #deconv2
          nn.MaxUnpool2d(2, stride=2),
          nn.ConvTranspose2d(128, 128, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(128, 64, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(64),
          #deconv1
          nn.MaxUnpool2d(2, stride=2),
          nn.ConvTranspose2d(64, 64, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(64, n_classes, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(n_classes)
        )
        self.activation = torch.nn.Softmax(dim=1)


    def forward(self, x):
      #block index initialization
      i=0
      #forward encoder
      for layer in self.encoder:
          if isinstance(layer, torch.nn.MaxPool2d):
              i+=1 # 1 -> 5
              x, indices = layer(x)
              self.pool_indices[i] = indices
              
          else:
              x = layer(x)
      #bottom
      x = self.bottom(x)
      #forward decoder
      for layer in self.decoder:
          if isinstance(layer, torch.nn.MaxUnpool2d):
              x = layer(x, self.pool_indices[i])
              i-=1 # 5 -> 1
          else:
              x = layer(x)
      x = self.activation(x)
      return x  
  