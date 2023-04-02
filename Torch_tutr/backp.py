import torch

x = torch.tensor(1.0)

y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

#Forward we goes forward
y_hat = w * x
loss = (y_hat - y) ** 2

print(loss)

#Back pass after calculate
loss.backward()
print(w.grad)

### update weiths
### next step


### We use bp with auto wigths
