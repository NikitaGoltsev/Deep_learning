#### The main steps in model
# 1) Design of model(input, output, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   -- forward pass: compute prediction
#   -- backward pass: gradients
#   -- update wigths
#

import torch
import torch.nn as nn # Nn modules
import numpy as np
# We are able to get datesets from sklearn
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare date

x_numpy, y_numpy = datasets.make_regression(n_samples = 100, n_features=1, noise = 20, random_state=1)

X = torch.tensor(x_numpy.astype(np.float32))
Y = torch.tensor(y_numpy.astype(np.float32))

#let's reshape them
Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape

# 1) model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

#2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) training loop 
num_epochs = 100
for epoch in range(num_epochs):
    # Steps in the trainign loop
    # Forward pass and loss
    y_predict = model(X)
    loss = criterion(y_predict, Y)

    #Baack pass
    loss.backward()

    #Update
    optimizer.step()

    optimizer.zero_grad()

    if epoch + 1 % 10 == 0:
        print(f'epoch: {epoch + 1}, loss item - {loss.item():.4f}')
    
# plot prediction

predicted = model(X).detach().numpy()

plt.plot(x_numpy,y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b') 
plt.show()