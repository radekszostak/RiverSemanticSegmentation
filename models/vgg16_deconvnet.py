import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
vgg16_pretrained = models.vgg16(pretrained=True)

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        #nn.BatchNorm2d(out_channels)
    )
def triple_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        #nn.BatchNorm2d(out_channels)
    )
def double_deconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, in_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )
def triple_deconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, in_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels, in_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )
class VggDeconvNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        
        self.conv1 = double_conv(3, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = triple_conv(128, 256)
        self.conv4 = triple_conv(256, 512)
        self.conv5 = triple_conv(512, 512)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.bottom = nn.Sequential(
          nn.Conv2d(512, 512, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(512, 512, 3, padding=1),
          nn.ReLU(inplace=True))
        self.deconv5 = triple_deconv(512,512)
        self.deconv4 = triple_deconv(512,256)
        self.deconv3 = triple_deconv(256,128)
        self.deconv2 = double_deconv(128,64)
        self.deconv1 = double_deconv(64,n_classes)
        self.maxunpool = nn.MaxUnpool2d(2, stride=2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512*13*13, 4096), 
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 1))
        
        #initialize weights with pretrained
        
    def forward(self, x):
        x = self.conv1(x)
        x, ind1 = self.maxpool(x)
        x = self.conv2(x)
        x, ind2 = self.maxpool(x)
        x = self.conv3(x)
        x, ind3 = self.maxpool(x)
        x = self.conv4(x)
        x, ind4 = self.maxpool(x)
        x = self.conv5(x)
        x, ind5 = self.maxpool(x)
        x = self.bottom(x)
        #x = x.view(x.size()[0], -1)
        #print(x.size())
        #x = self.classifier(x)
        #print(x.size())
        x = self.maxunpool(x, ind5)
        x = self.deconv5(x)
        x = self.maxunpool(x, ind4)
        x = self.deconv4(x)
        x = self.maxunpool(x, ind3)
        x = self.deconv3(x)
        x = self.maxunpool(x, ind2)
        x = self.deconv2(x)
        x = self.maxunpool(x, ind1)
        x = self.deconv1(x)

        return x  
  