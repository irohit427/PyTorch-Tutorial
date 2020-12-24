import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([1, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0, dtype=torch.float32, requires_grad=True)

def forward(x):
  return w * x
loss = nn.MSELoss()
print(f'Prediction before training: f(5) = {forward(5):.3f}')

learning_rate = 1e-2
iterations = 20
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(iterations):
  y_pred = forward(X)
  
  l = loss(y, y_pred)
  
  l.backward()
  
  optimizer.step()
  optimizer.zero_grad()
  
  if epoch % 2 == 0:
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')