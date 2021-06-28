import torch
from models.vgg_unet import VggUnet
import random
import pdb
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import torch.nn as nn   
import copy

#first modify vgg_unet

def generate_models(nr):
    model = VggUnet()
    models = {}
    layers = {}
    skip_connections = {}

    models['0'] = model
    #modify depths
    #choose base model
    #set hyperparameters
    for i in range(1, nr): 
        model_ = copy.deepcopy(model)
        model_ = mutation(model_)
        models[str(i)] = model_

    return models, layers


def insert_block():
    pass


def modify_block():
    pass


def crossover(models, layers, model_1, model_2):
    children = {}

    return models, layers


def mutation(model):
    opt = random.randint(0, 2)
    activations = (nn.ReLU(), nn.PReLU(), nn.RReLU(), nn.LeakyReLU(0.01), nn.ReLU6(), nn.ELU())
    
    #enc = model.encoder
    #dec = model.decoder

    block_counter = 0

    if opt == 0:
        block = random.randint(0, model.blocks-1)
        i = 0
        while block_counter < block:
            block_counter += 1
            while i < len(model.encoder) and not isinstance(model.encoder[i], nn.MaxPool2d):
                i += 1
            i += 1

        #get in and out, modify out
        if isinstance(model.encoder[i], nn.Conv2d):
            in_ = model.encoder[i].in_channels
            out_ = model.encoder[i].out_channels
            out_ = random.randint(in_, out_ + 128)
            model.encoder[i].out_channels = out_
            i += 1
            
            while i < len(model.encoder) and not isinstance(model.encoder[i], nn.MaxPool2d):
                if isinstance(model.encoder[i], nn.Conv2d):
                    model.encoder[i].in_channels = out_
                    model.encoder[i].out_channels = out_
                i += 1 

            model.depths[block] = out_ 

            if i+1 < len(model.encoder) and isinstance(model.encoder[i+1], nn.Conv2d):
                model.encoder[i+1].in_channels = out_

            block_counter = model.blocks - 1
            i = 0
            while block_counter > 0:
                if block_counter == block:
                    break

                block_counter -= 1
                while i < len(model.decoder) and not isinstance(model.decoder[i], nn.Upsample):
                    i += 1
                i += 1

            if i+1 < len(model.decoder) and isinstance(model.decoder[i+1], nn.Conv2d):
               model.decoder[i+1].in_channels = out_ + model.decoder[i+1].out_channels 

    elif opt == 2:   
        block = random.randint(0, model.blocks-2)
        i = 0
        while block_counter < block:
            block_counter += 1
            while i < len(model.decoder) and not isinstance(model.decoder[i], nn.Upsample):
                i += 1
            i += 1

        #get in and out, modify out
        if isinstance(model.decoder[i], nn.Conv2d):
            in_ = model.decoder[i].in_channels
            out_ = model.decoder[i].out_channels
            out_ = random.randint(32, in_)
            model.decoder[i].out_channels = out_
            i += 1
            
            while i < len(model.decoder) and not isinstance(model.decoder[i], nn.Upsample):
                if isinstance(model.decoder[i], nn.Conv2d):
                    model.decoder[i].in_channels = out_ 
                    model.decoder[i].out_channels = out_
                if isinstance(model.decoder[i], nn.BatchNorm2d):
                    model.decoder[i].num_features =  out_
                i += 1

            if i+1 < len(model.decoder) and isinstance(model.decoder[i+1], nn.Conv2d):
                model.decoder[i+1].in_channels = out_ + model.depths[-2-block]

    elif opt == 1:
        for i in range(len(model.encoder)):
            if isinstance(model.encoder[i], nn.ReLU) or isinstance(model.encoder[i], nn.PReLU) or isinstance(model.encoder[i], nn.RReLU) or isinstance(model.encoder[i], nn.LeakyReLU) or isinstance(model.encoder[i], nn.ReLU6) or isinstance(model.encoder[i], nn.ELU):
                model.encoder[i] = activations[random.randint(0, 5)]
        for i in range(len(model.decoder)):
            if isinstance(model.decoder[i], nn.ReLU) or isinstance(model.decoder[i], nn.PReLU) or isinstance(model.decoder[i], nn.RReLU) or isinstance(model.decoder[i], nn.LeakyReLU) or isinstance(model.decoder[i], nn.ReLU6) or isinstance(model.decoder[i], nn.ELU):
                model.decoder[i] = activations[random.randint(0, 5)]

    elif opt == 2: 
        pass

    return model

def generate_block():
    pass

def generate_model():
    opt = random.randint(0, 2)
    opt = 0    

    model = VggUnet()
    model.encoder = None
    model.decoder = None
    encoder = []
    decoder = [] 
    model.depths = []
    lengths = []

    blocks = random.randint(2, 5)
    for j in range(blocks):
        layers = random.randint(2, 3)

        if j == 0:
            in_ = 3
            out_ = random.randint(32, 100)
            prev_out = out_
         
        else:
            in_ = prev_out
            out_ = random.randint(in_ + 64, in_ + 200)
            prev_out = out_

        model.depths.append(out_)

        for k in range(layers):   
            if k == 0:
                encoder.append(torch.nn.Conv2d(in_, out_, 3, padding=1))
                encoder.append(torch.nn.ReLU())
                #encoder.append(torch.nn.MaxPool2d(2, stride=2))
            else:
                encoder.append(torch.nn.Conv2d(out_, out_, 3, padding=1))
                encoder.append(torch.nn.ReLU())
                if j < blocks - 1 and k == layers - 1:
                    encoder.append(torch.nn.MaxPool2d(2, stride=2))


    #for i in range(len(self.encoder)):
    #    if isinstance(self.encoder[i], torch.nn.Conv2d):
    #        self.encoder[i].weight.data = vgg16_pretrained.features[i].weight.data
    #        self.encoder[i].bias.data = vgg16_pretrained.features[i].bias.data

    for j in range(blocks):
           
        if j == 0:
            in_ = model.depths[-1-j]
            decoder.append(nn.Conv2d(in_, in_, 3, padding=1))
            decoder.append(nn.ReLU(inplace=True))
            decoder.append(nn.Conv2d(in_, in_, 3, padding=1))
            decoder.append(nn.ReLU(inplace=True))
            decoder.append(nn.BatchNorm2d(in_))
            decoder.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            out = in_   
        else:
            decoder.append(nn.Conv2d(out + model.depths[-1-j], model.depths[-1-j], 3, padding=1))
            decoder.append(nn.ReLU(inplace=True))
            decoder.append(nn.Conv2d(model.depths[-1-j], model.depths[-1-j], 3, padding=1))
            decoder.append(nn.ReLU(inplace=True))
            decoder.append(nn.BatchNorm2d(model.depths[-1-j]))
            if j < blocks - 1:
                decoder.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            out = model.depths[-1-j]

    model.encoder = nn.Sequential(*encoder)
    model.decoder = nn.Sequential(*decoder)

    #if opt == 0:
        
    #    j=0
    #    #forward encoder
    #    for layer in self.encoder:
    #        if isinstance(layer, torch.nn.MaxPool2d):
    #            j+=1 # 1 -> 5
    #            self.conv_out[j] = x
    #            x = layer(x)
    #        else:
    #            x = layer(x)
    #    #forward decoder
    #    for layer in self.decoder:
    #        if isinstance(layer, torch.nn.Upsample):
    #            x = layer(x)
    #            x = torch.cat([x, self.conv_out[j]], dim=1)
    #            j-=1 # 5 -> 1
    #        else:
    #            x = layer(x)
    #    x = self.activation(x)
    return model

def mutation_(models, layers):
    return models, layers


def mutate_layers(layers):
    return layers


def main_algorithm(nr, iterations, mutation_ratio):
    models, layers = {}

    models, layers = generate_models(nr)   

    for i in range(iterations):
        models, layers = mutation(model, layers)
    
    
#models, _ = generate_models(10)
#model = generate_model()


