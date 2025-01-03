from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.models import MnistCNN, Cifar10CNN, Cifar100CNN


def get_mnist_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar10_loaders(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_cifar100_loaders(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR100(root='./data/CIFAR100', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR100(root='./data/CIFAR100', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_data_loaders(model, batch_size):
    if isinstance(model, MnistCNN):
        train_loader, test_loader = get_mnist_loaders(batch_size)
    elif isinstance(model, Cifar10CNN):
        train_loader, test_loader = get_cifar10_loaders(batch_size)
    elif isinstance(model, Cifar100CNN):
        train_loader, test_loader = get_cifar100_loaders(batch_size)
    else:
        raise ValueError(f"Unsupported model type")
    return train_loader, test_loader
