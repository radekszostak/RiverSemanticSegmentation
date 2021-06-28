import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

vgg16_pretrained = models.vgg16(pretrained=True)
import pdb

class AttentionBased(nn.Module):

    #def __init__(self, empty, blocks=4):
    #    super().__init__()
    #    self.blocks = blocks
    #    self.depths = []

    #    self.depths.append(64)
    #    self.depths.append(128) 
    #    self.depths.append(256)
    #    self.depths.append(512)

    def __init__(self, n_classes=2, blocks=4):
        super().__init__()


        #for multiple scales
        self.blocks = blocks
        self.depths = []

        #self.linear_1 = nn.Linear(416*416*3, 1024) #conv1d
        self.conv_1 = nn.Conv2d(3, 64, 3, 1, 1) #conv1d
        self.relu_1 = nn.ReLU()
        #self.conv_1 = nn.Conv2d(3, 256, 3, stride=2, padding = 1) #conv1d
        #self.conv_2 = nn.Conv2d(3, 256, 3, stride=1, padding = 1)

        self.conv_2 = nn.Conv2d(64, 64, 3, 1, 1) #conv1d
        self.relu_2 = nn.ReLU()
        self.maxpool_1 = torch.nn.MaxPool2d(2, stride=2)

        self.conv_3 = nn.Conv2d(64, 128, 3, 1, 1) #conv1d
        self.relu_3 = nn.ReLU()
        self.conv_4 = nn.Conv2d(128, 128, 3, 1, 1) #conv1d
        self.relu_4 = nn.ReLU()
        self.maxpool_2 = torch.nn.MaxPool2d(2, stride=2)

        #self.deconv_1 = nn.Conv2d(256, 3, 3, 1, 1)

        self.deconv_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.drelu_1 = nn.ReLU(inplace=True)
        self.deconv_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.drelu_2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
 
        self.deconv_3 = nn.Conv2d(128, 64, 3, padding=1)
        self.drelu_3 = nn.ReLU(inplace=True)
        self.deconv_4 = nn.Conv2d(64, 64, 3, padding=1)
        self.drelu_4 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.deconv_5 = nn.Conv2d(64, n_classes, 3, padding=1)
        
        self.depths.append(64)
        self.depths.append(128) 
        self.depths.append(256)
        self.depths.append(512)

        self.conv_out = dict()
        self.activation = nn.Sigmoid()

        self.conv_1.weight.data = vgg16_pretrained.features[0].weight.data
        self.conv_1.bias.data = vgg16_pretrained.features[0].bias.data

        self.conv_2.weight.data = vgg16_pretrained.features[2].weight.data
        self.conv_2.bias.data = vgg16_pretrained.features[2].bias.data

        self.conv_3.weight.data = vgg16_pretrained.features[5].weight.data
        self.conv_3.bias.data = vgg16_pretrained.features[5].bias.data

        self.conv_4.weight.data = vgg16_pretrained.features[7].weight.data
        self.conv_4.bias.data = vgg16_pretrained.features[7].bias.data
    

    def forward(self, x):

        x = self.conv_1(x)
        x = self.relu_1(x)
        #x = self.maxpool_1(x)

        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.maxpool_1(x)

        x = self.conv_3(x)
        x = self.relu_3(x)

        x = self.conv_4(x)
        x = self.relu_4(x)
        x = self.maxpool_2(x)

        tmp = torch.zeros(x.size()).cuda()
        tmp_ = torch.zeros(x.size()).cuda()
        #pdb.set_trace()
        
        for k in range(4):
            for l in range(4):

                attention_w = []
                for m in range(4):
                    for n in range(4):
                        att = 0
                        for z in range(x.size()[0]):
                            for y in range(128):
                                #pdb.set_trace()
                                #attention_w.append((torch.sum(x[][][m*52:(m+1)*52][n*52:(n+1)*52] * x[][][k*52:(k+1)*52][l*52:(l+1)*52])/64).item())
                                
                                att += ((torch.sum(x[z,y,m*26:(m+1)*26,n*26:(n+1)*26] * x[z,y,k*26:(k+1)*26,l*26:(l+1)*26])/676).item())
                        #att += ((torch.sum(x[z,y,m*52:(m+1)*52,n*52:(n+1)*52] * x[z,y,k*52:(k+1)*52,l*52:(l+1)*52])/64).item())
                        attention_w.append(att)
                #attention_w = torch.reshape(self.softmax(torch.from_numpy(np.asarray(attention_w))), [1, 64])

                #pdb.set_trace()
                
                attention_w = torch.from_numpy(np.asarray(attention_w))
                
                for m in range(4):
                    for n in range(4):
                        tmp[z,y,m*26:(m+1)*26,n*26:(n+1)*26] += attention_w[m*4 + n]*x[z,y,m*26:(m+1)*26,n*26:(n+1)*26]

        #pdb.set_trace()
        for k in range(4):
            for l in range(4):

                attention_w = []
                for m in range(4):
                    for n in range(4):
                        att = 0
                        for z in range(x.size()[0]):
                            for y in range(128):
                                #pdb.set_trace()
                                
                                att += ((torch.sum(tmp[z,y,m*26:(m+1)*26,n*26:(n+1)*26] * tmp[z,y,k*26:(k+1)*26,l*26:(l+1)*26])/676).item())
                        #att += ((torch.sum(x[z,y,m*52:(m+1)*52,n*52:(n+1)*52] * x[z,y,k*52:(k+1)*52,l*52:(l+1)*52])/64).item())
                        attention_w.append(att)
                #attention_w = torch.reshape(self.softmax(torch.from_numpy(np.asarray(attention_w))), [1, 64])
                
                attention_w = torch.from_numpy(np.asarray(attention_w))
                
                for m in range(4):
                    for n in range(4):
                        tmp_[z,y,m*26:(m+1)*26,n*26:(n+1)*26] += attention_w[m*4 + n]*tmp[z,y,m*26:(m+1)*26,n*26:(n+1)*26]
              
        x = self.deconv_1(tmp_)
        x = self.drelu_1(x)

        x = self.deconv_2(x)
        x = self.drelu_2(x)
        x = self.bn2(x)
        x = self.up2(x)
             
        x = self.deconv_3(x)
        x = self.drelu_3(x)

        x = self.deconv_4(x)
        x = self.drelu_4(x)
        x = self.bn4(x)
        x = self.up4(x)
        x = self.deconv_5(x)

        x = self.activation(x)

        #conv1 = torch.reshape(outputs, [1, 32, int(math.ceil(float(self.sensor)/1)), int(math.ceil(float(self.sensor)/1))])
      
        #block index initialization
        #i=0
        #forward encoder
        #for layer in self.encoder:
        #    if isinstance(layer, torch.nn.MaxPool2d):
        #        i+=1 # 1 -> 5
        #        self.conv_out[i] = x
        #        x = layer(x)
        #    else:
        #        x = layer(x)
        #forward decoder
        #for layer in self.decoder:
        #    if isinstance(layer, torch.nn.Upsample):
        #        x = layer(x)
        #        x = torch.cat([x, self.conv_out[i]], dim=1)
        #        i-=1 # 5 -> 1
        #    else:
        #        x = layer(x)
        #x = self.activation(x)
        return x  