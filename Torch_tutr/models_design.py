
import torch
import torch.nn as nn # the set of function

# Now I need to rebuild code from numpy to torch libs
# That's the same aim 
#x = torch.tensor([1, 2, 3, 4], dtype=torch.float32) set are have a new shape for net
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

x_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = x.shape
print(n_samples, n_features)

###                                               Gradient is on      
#w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # Also in that example we need to use wigths as tensor dtype
###

# Model needs input and output sizes = n_features
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size) # we don't need in w anymore


#print(f'Prediction befor treaning: f(5) = {forward(5):.3f}')
# We are able to use .item to get the actual value
print(f'Prediction befor treaning: f(5) = {model(x_test).item():.3f}')
#                                           \- that returns our result
#### The main steps in model
# 1) Design of model(input, output, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   -- forward pass: compute prediction
#   -- backward pass: gradients
#   -- update wigths
#

#We are able to write optimiser and losses

learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()

#               we put our weigths here/ lr - is learning rate from our nate                        
#optimizer = torch.optim.SGD([w], lr = learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
# \--that's a collable functions, so the don't need to update our weigth
# model.parametrs - that's weigths

for epoch in range(n_iters):

    # prediction = forward pass
    y_pred = model(x) # model(x) - is prediction

    ls = loss(y, y_pred) # Loss steals the same

    #dw = gradient(x, y, y_pred)
    #Here to use grad we need to use default bp
    ls.backward() # that's dl/dw

    #Wigths update
    #with torch.no_grad():      # w.grad eual to dw
        #w -= learning_rate * w.grad
    #The are going to use default updates
    optimizer.step()

    # Update gradient to zero
    #w.grad.zero_()
    optimizer.zero_grad() # UPDATE FROM OPTIMIZER

    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch: {epoch+1}: w = {w[0][0].item():.3f}, loss = {ls:.8f}')

print(f'Prediction after treaning: f(5) = {model(x_test).item():.3f}')
# We able to play with hiperpar for the best out