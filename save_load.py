import torch
import os
from model import SimpleCNN


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved in {path}")


def load_model(path, device):
    model = SimpleCNN()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.to(device)
    print(f"Model loaded from {path}")
    return model
