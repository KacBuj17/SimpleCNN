import pytest
from torch.utils.data import DataLoader
from models.models import MnistCNN, Cifar10CNN, Cifar100CNN
from data.data_loader import get_mnist_loaders, get_cifar10_loaders, get_cifar100_loaders, get_data_loaders


def test_get_mnist_loaders():
    train_loader, test_loader = get_mnist_loaders(batch_size=32)

    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Sprawdź długość datasetów
    assert len(train_loader) == 60000, "MNIST train dataset size is incorrect"
    assert len(test_loader) == 10000, "MNIST test dataset size is incorrect"


def test_get_cifar10_loaders():
    train_loader, test_loader = get_cifar10_loaders(batch_size=32)

    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Sprawdź długość datasetów
    assert len(train_loader) == 50000, "CIFAR-10 train dataset size is incorrect"
    assert len(test_loader) == 10000, "CIFAR-10 test dataset size is incorrect"


def test_get_cifar100_loaders():
    train_loader, test_loader = get_cifar100_loaders(batch_size=32)

    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Sprawdź długość datasetów
    assert len(train_loader) == 50000, "CIFAR-100 train dataset size is incorrect"
    assert len(test_loader) == 10000, "CIFAR-100 test dataset size is incorrect"


def test_get_data_loaders_mnist():
    model = MnistCNN()
    train_loader, test_loader = get_data_loaders(model, batch_size=32)

    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert len(train_loader) == 60000, "MNIST train dataset size is incorrect"


def test_get_data_loaders_cifar10():
    model = Cifar10CNN()
    train_loader, test_loader = get_data_loaders(model, batch_size=32)

    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert len(train_loader) == 50000, "CIFAR-10 train dataset size is incorrect"


def test_get_data_loaders_cifar100():
    model = Cifar100CNN()
    train_loader, test_loader = get_data_loaders(model, batch_size=32)

    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert len(train_loader) == 50000, "CIFAR-100 train dataset size is incorrect"


def test_get_data_loaders_invalid_model():
    class UnsupportedModel:
        pass

    model = UnsupportedModel()
    with pytest.raises(ValueError, match="Unsupported model type"):
        get_data_loaders(model, batch_size=32)
