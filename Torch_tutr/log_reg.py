import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.preprocessing import StandardScaler ### ?
from sklearn.model_selection import train_test_split # to work with date before start of the main part

#### The main steps in model
# 1) Design of model(input, output, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   -- forward pass: compute prediction
#   -- backward pass: gradients
#   -- update wigths
#

# 0) prepair date
bc = datasets.load_breast_cancer() # LogReg problemset

x, y = bc.data, bc.target

n_samples, n_features = x.shape

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# scale ?
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

#transfor for preparing 
Y_train = Y_train.view(Y_train.shape[0], 1)
Y_test = Y_test.view(Y_test.shape[0], 1)

# 1) model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_feuters):
        
        super(LogisticRegression, self).__init__
        self.linear = nn.Linear(n_input_feuters, 1)
    
    def forward(self, x):
        y_predict = torch.sigmoid(self.linear(x))
        return y_predict
    
model = LogisticRegression(n_features)

# 2) loss and optimizer 
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parametrs(), lr = learning_rate)

# 3) trinign loop
