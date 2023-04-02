import torch

                # With that parametr we automativcle get bp for our gradient
x = torch.rand(3, requires_grad=True)

print(x)
y = x + 2
print(y)
z = y*y*2
print(z)
z = z.mean()
print(z)

# After we can calculate the gradient with only one comand

z.backward()
print(x.grad)