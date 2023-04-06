import torch
import torch.nn as nn
import torchvision # for datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# By steps
# MNIST
# 1 - Work with date
# 2 - Multilayer nn with actiwation functions
# 3 - Loss and optimizer
# 4 - Training loop
# 5 - Create model evolution
# 6(+) - add gpu support for tje model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # With that comand we will be able to use GPU
# if gpu is avalible

# hyper params
input_size = 784 # image 28*28
hidden_size = 100 # We are able to play with that
num_class = 10 # that's because there are 10 classes from /0 to 9/
num_epochs = 2
batch_size = 100 # ? repeat
learning_rate = 0.001 # ? repeat

#MNIST - is the name of the dateset
# 1) work with data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
    transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
    transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size,
    shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size,
    shuffle = False)

examples = iter(train_loader)

samples, labels = next(examples)
print(samples.shape, labels.shape)