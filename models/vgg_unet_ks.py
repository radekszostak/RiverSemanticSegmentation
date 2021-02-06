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
class VggUnetKs(nn.Module):
    def __init__(self, n_channels=2 ):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            # conv1
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),
            # conv2
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),
            # conv3
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),
            # conv4
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2)

        )
        self.pool_outputs = dict()

        # initialize weights
        for i in range(len(self.encoder)):
            if isinstance(self.encoder[i], torch.nn.Conv2d):
                self.encoder[i].weight.data = vgg16_pretrained.features[i].weight.data
                self.encoder[i].bias.data = vgg16_pretrained.features[i].bias.data

        #self.batch_norm = nn.BatchNorm2d(512,0.001,0.99)
        self.conv_up4 = conv_up(512,512)
        self.conv_up3 = conv_up(512+256,256)
        self.conv_up2 = conv_up(256+128,128)
        self.conv_up1 = conv_up(128+64,64)
        self.conv_last = nn.Conv2d(64,n_channels,3,padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.activation = nn.Softmax()
    
    def forward_encoder(self, x):
        output = x
        i=1
        for layer in self.encoder:
            output = layer(output)
            if isinstance(layer, torch.nn.MaxPool2d):
                self.pool_outputs[i]=output
                i+=1
        return output
    
    def forward(self, x):
        x = self.forward_encoder(x)
        x = self.conv_up4(x)
        #x = self.batch_norm(x)
        x = self.upsample(x)
        x = torch.cat([x, self.pool_outputs[3]], dim=1)
        x = self.conv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, self.pool_outputs[2]], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, self.pool_outputs[1]], dim=1)
        x = self.conv_up1(x)
        out = self.conv_last(x)
        
        #out = self.activation(x)
        return out  
  