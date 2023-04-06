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

# Size check
#samples, labels = next(examples)
#print(samples.shape, labels.shape)
# output here is ----> 100, 1, 28, 28

# How dose it looks ?
#for i in range(6):
    #plt.subplot(2, 3, i+1)
    #plt.imshow(samples[i][0], cmap='gray')
#plt.show()

class NeuralNet(nn.Module):

    def __init__(self, input_size, hiden_size, num_classes) -> None:
        super(NeuralNet, self).__init__() # ?
        # Creation of lyers
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_class)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # No softmax because we are going to use crossentripy
        return out

model = NeuralNet(input_size, hidden_size, num_class)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Training loop 
n_total_srep = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): # ?

        # 100, 1, 28, 28
        # 100, 784 --> We need to reshape out tenzor
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Bachwards - ?
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i+1}/{n_total_srep}, loss = {loss.item():.4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, prediciton = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (prediciton == labels).sum().item()

    acc = 100.0 * n_correct/n_samples
    
    print(f'Final result is: accuracy = {acc}')
