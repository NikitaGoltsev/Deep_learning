import torch

# Now I need to rebuild code from numpy to torch libs
# That's the same aim
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

###                                               Gradient is on      
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # Also in that example we need to use wigths as tensor dtype
###

# Forward and lost are the same as in numpy
def forward(x):
    return w * x

def loss(y, y_predict):
    return ((y_predict - y)**2).mean()

# Prediction befor treaning
print(f'Prediction befor treaning: f(5) = {forward(5):.3f}')

# Traning
learning_rate = 0.01
n_iters = 100 # With torch we are need more iterations( first try was with 10 iters)

for epoch in range(n_iters):

    # prediction = forward pass
    y_pred = forward(x)

    ls = loss(y, y_pred)

    #dw = gradient(x, y, y_pred)
    #Here to use grad we need to use default bp
    ls.backward() # that's dl/dw

    #Wigths update
    with torch.no_grad():      # w.grad eual to dw
        w -= learning_rate * w.grad
    
    # Update gradient to zero
    w.grad.zero_()

    if epoch % 1 == 0:
        print(f'epoch: {epoch+1}: w = {w:.3f}, loss = {ls:.8f}')

print(f'Prediction after treaning: f(5) = {forward(5):.3f}')