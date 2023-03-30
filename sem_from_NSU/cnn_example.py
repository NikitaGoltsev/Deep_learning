import numpy as np
import torch

# nn.Conv2d - one of the most important commands at CNN
# That helps us to create convolution functions

from typing import Tuple
from random import randrange
import inspect

import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
# Maybe, I won't need most of libs under that comment

class ConvolutionNetwork(nn.Module):
    def __init__(
        self, 
        img_size: Tuple[int, int] = (1, 28, 28),  # (кол-во каналов, высота px, ширина px)
        num_classes: int = 10,
    ):
        super().__init__()
        in_channels = img_size[0]
        height = img_size[1]
        width = img_size[2]
                
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1)
        self.fc3 = nn.Linear(in_features=16 * height * width, out_features=num_classes)
        
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        
        return x

#model = ConvolutionNetwork()
#[(name, parameter.shape) for name, parameter in model.named_parameters()] # To check parametrs
#
