import os
import torch
from models.models import MnistCNN, Cifar10CNN, Cifar100CNN


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved in {path}")


def load_mnist_model(path, device):
    model = MnistCNN()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    print(f"Model loaded from {path}")
    return model

def load_cifar10_model(path, device):
    model = Cifar10CNN()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    print(f"Model loaded from {path}")
    return model

def load_cifar100_model(path, device):
    model = Cifar100CNN()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    print(f"Model loaded from {path}")
    return model
