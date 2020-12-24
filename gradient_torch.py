import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([1, 4, 6, 8], dtype=torch.float32)
X = X.view(4, 1)
y = y.view(4, 1)
X_test = torch.tensor([5], dtype=torch.float32)
# w = torch.tensor(0, dtype=torch.float32, requires_grad=True)

# def forward(x):
#   return w * x
n_sample, n_features = X.shape
output = 1
model = nn.Linear(n_features, output)

loss = nn.MSELoss()
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

learning_rate = 1e-2
iterations = 40
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(iterations):
  y_pred = model(X)
  
  l = loss(y, y_pred)
  
  l.backward()
  
  optimizer.step()
  optimizer.zero_grad()
  
  if epoch % 2 == 0:
    print(f'epoch {epoch+1}: loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')