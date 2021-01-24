import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )
def triple_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )
def conv_up(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels,0.001,0.99)
    )
class VggUnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_down1 = double_conv(3, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = triple_conv(128, 256)
        self.conv_down4 = triple_conv(256, 512)
        #self.conv_bottom = nn.Conv2d(512,512,3,padding=1)
        #self.batch_norm = nn.BatchNorm2d(512,0.001,0.99)
        self.conv_bottom = conv_up(512,512)
        self.conv_up3 = conv_up(512+256,256)
        self.conv_up2 = conv_up(256+128,128)
        self.conv_up1 = conv_up(128+64,64)
        self.conv_last = nn.Conv2d(64,1,3,padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.activation = nn.Softmax()

    def forward(self, x):
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.conv_down4(x)
        #x = self.maxpool(conv4)
        x = self.conv_bottom(conv4)
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up1(x)
        x = self.conv_last(x)
        
        out = self.activation(x)
        return out  
  