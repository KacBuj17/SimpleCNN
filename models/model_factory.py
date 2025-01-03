import torch.nn as nn
import torch.optim as optim
from models.models import MnistCNN, Cifar10CNN, Cifar100CNN
from models.model_utils import load_cifar100_model, load_cifar10_model, load_mnist_model

class ModelFactory:
    @staticmethod
    def create_train_model(model_type: str, learning_rate: float):
        if model_type == "mnist":
            model = MnistCNN()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            return model, criterion, optimizer
        elif model_type == "cifar10":
            model = Cifar10CNN()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            return model, criterion, optimizer
        elif model_type == "cifar100":
            model = Cifar100CNN()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            return model, criterion, optimizer
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")

    @staticmethod
    def create_eval_model(path, device, model_type):
        if model_type == "mnist":
            return load_mnist_model(path, device)
        elif model_type == "cifar10":
            return load_cifar10_model(path, device)
        elif model_type == "cifar100":
            return load_cifar100_model(path, device)
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")