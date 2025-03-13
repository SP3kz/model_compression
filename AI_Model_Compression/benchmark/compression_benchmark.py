import time
from models.compressed_transformer import CompressedTransformer
from optimizers.curvature_backprop import curvature_backprop
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CompressedTransformer(input_dim=10, hidden_dim=20, output_dim=5, num_heads=2, num_layers=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

data = torch.randn(32, 10).to(device)
labels = torch.randn(32, 5).to(device)

start_time = time.time()

for i in range(10):
    loss = curvature_backprop(model, loss_fn, optimizer, data, labels)
    print(f"Iteration {i+1}, Loss: {loss}")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.4f} seconds.")
