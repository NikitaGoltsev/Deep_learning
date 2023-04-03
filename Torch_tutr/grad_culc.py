import numpy as np

# f = w * x

# f = 2 * x

x = np.array([1, 2, 3, 4], dtype=np.float32)

y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# In that exampl we want to use gradient with backpropagation
# Our aim is to calculate correct wids

def forward(x):
    return w * x

def loss(y, y_predict):
    return ((y_predict - y)**2).mean()

#gradient
#MSE = 1/N * (w*x - y)**2
#dj/dw = 1/N 2x (w*x - y)
def gradient(x, y, y_predict):
    return np.dot(2*x, y_predict - y).mean()

# Prediction befor treaning
print(f'Prediction befor treaning: f(5) = {forward(5):.3f}')