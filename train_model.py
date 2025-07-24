import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

X = np.load("X_train.npy")  # shape: [N, 21]
Y = np.load("Y_train.npy")  # shape: [N, 21]

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

class VCNet(nn.Module):
    def __init__(self):
        super(VCNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 21)
        )

    def forward(self, x):
        return self.net(x)

model = VCNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(30):
    pred = model(X)
    loss = loss_fn(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "vc_model.pth")
