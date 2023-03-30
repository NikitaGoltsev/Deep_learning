import numpy as np
import torch

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

from cnn_example import ConvolutionNetwork  # That's over file

class Local_CNN(ConvolutionNetwork):

    None 

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.5],
        std=[0.5],
    ),
])

# FashionMNIST: https://pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html
# I going to use dataset with 
# 

train_dataset = torchvision.datasets.FashionMNIST(
    root='datasets',
    download=True,
    train=True,
    transform=transform,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,
)

valid_dataset = torchvision.datasets.FashionMNIST(
    root='datasets',
    download=True,
    train=False,
    transform=transform,
)

valid_dataloader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,
)

batch, _ = next(iter(train_dataloader))