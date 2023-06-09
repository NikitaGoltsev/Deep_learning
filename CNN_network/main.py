import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision # We are able to get data sets from here with func
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parametrs 
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# Core dataset has only PILImage of range [0,1]
# We need to transform them to tenzor
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # ?

# Get from torchvision
train_dataset = torchvision.datasets.CIFAR10(root='./date', train=True,
                                                download= True, transform=transform)
                                                #        we are able to use transform.Compose func from up ads param

test_dataset = torchvision.datasets.CIFAR10(root='./date', train = False,
                                                download = True, transform=transform)
                
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size,
    shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size,
    shuffle = False)

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # List of classes for our net

class ConvNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # to specify size
        #       1) channel_size = color
        #       2)  output_cannel_size = 6
        #       3)  kernel_zize = 5
        self.conv1 = nn.Conv2d(3, 6, 5)
        #                 in pool just kernel sizes
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #
        #
        self.fc1 = nn.Linear(16*5*5, 120) #? # fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        pass

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs): # That's a training loop
    for i, (images, labels) in enumerate(train_loader):
        # origin shape [4, 3, 32, 32] = 4, 3, 1024
        # input layer: 3 input layers, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Use forward pass 
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'epoch: {epoch}/{num_epochs}, step: {i}, Loss: {loss.item():.4f}')

# Finished training
print(f'Trainig has finished')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predict = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predict == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predict[i]
            if(label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'accuravy of the network is {acc} %')

    # For each single class
    for i in range(10):
        lc_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {lc_acc} %')


