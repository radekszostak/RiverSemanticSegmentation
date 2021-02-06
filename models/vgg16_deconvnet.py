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
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        )

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
    def forward_encoder(self, x):
        output = x
        i=1
        for layer in self.encoder:
            if isinstance(layer, torch.nn.MaxPool2d):
                output, indices = layer(output)
                #self.feature_outputs[i] = output
                self.pool_indices[i] = indices
            else:
                output = layer(output)
                #self.feature_outputs[i] = output
        return output

    def forward(self, x):
        x = self.forward_encoder(x)
        x = self.bottom(x)
        #x = x.view(x.size()[0], -1)
        #print(x.size())
        #x = self.classifier(x)
        #print(x.size())
        x = self.maxunpool(x, self.pool_indices[5])
        x = self.deconv5(x)
        x = self.maxunpool(x, self.pool_indices[4])
        x = self.deconv4(x)
        x = self.maxunpool(x, self.pool_indices[3])
        x = self.deconv3(x)
        x = self.maxunpool(x, self.pool_indices[2])
        x = self.deconv2(x)
        x = self.maxunpool(x, self.pool_indices[1])
        x = self.deconv1(x)

        return x  
  