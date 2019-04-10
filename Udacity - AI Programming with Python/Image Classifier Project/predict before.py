#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:27:35 2018

@author: Jake
"""

import torch
from torch import nn
from torchvision import datasets, transforms, models
import argparse
import numpy as np
from PIL import Image
import json

# Create parser
parser = argparse.ArgumentParser(description = 'Image Classifier')
parser.add_argument('image_path')
parser.add_argument('--checkpoint', default = 'checkpoint.pth', type = str, help = 'Checkpoint to load')
parser.add_argument('--topk', default = 3, type = int, help = 'Top k most likely classes')
parser.add_argument('--category_names', default = 'cat_to_name.json', type = str, help = 'Map of category to names')
parser.add_argument('--device', default = 'cpu', type = str, help = 'Device to train model on') 
args = parser.parse_args()
image_path, checkpoint, topk, category_names, device = args.image_path, args.checkpoint, args.topk, args.category_names, args.device

# Assign category names
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load checkpoint
def load_checkpoint(filepath):
    '''
    Loads checkpoint of saved model at "filepath"
    Returns model
    '''
    checkpoint = torch.load(filepath)
    model = models.vgg19(pretrained = True)
    model.classifier = nn.Sequential(checkpoint['Classifier Dictionary'])
    model.load_state_dict(checkpoint['State Dict'])
    model.class_to_idx = checkpoint['Class to Index']

    return model

# Process image
def process_image(image_path):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model
    Returns a Numpy array
    '''
    im = Image.open(image_path)
    min_index = im.size.index(min(im.size))
    if min_index == 0:
        size = (256, int(256/min(im.size) * max(im.size)))
    else:
        size = (int(256/min(im.size) * max(im.size)), 256)

    im.thumbnail(size)
    
    transformations = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transformed_image = transformations(im)
    np_image = np.array(transformed_image).transpose(0,1,2)
    
    return np_image

# Predict image
def predict(image_path, model, topk):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    Returns probabilities, indices of topk classes
    '''
    model.eval()
    model.to(device)
    proc_im = torch.from_numpy(process_image(image_path))
    proc_im.unsqueeze_(0)
    output = model.forward(proc_im.float())
    return torch.exp(output).data[0].topk(topk)

# Display top 5 class predictions
def predict_flower(probs, indices):
    '''
    Returns probabilities, names of classes
    '''
    inv_map = {v: k for k, v in model.class_to_idx.items()}
    
    flower_names = [cat_to_name[inv_map[int(i)]] for i in indices]
    probs = [float(prob) for prob in probs]
    
    return probs, flower_names

model = load_checkpoint(checkpoint)
probs, indices = predict(image_path, model, topk)
probs, flower_names = predict_flower(probs, indices)
for i,j in zip(probs, flower_names):
    print(str(round(100 * i, 2)) + '% ' + j.title())