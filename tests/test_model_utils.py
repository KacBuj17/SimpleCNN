import os
import pytest
import torch
from unittest.mock import MagicMock
from models.model_utils import save_model, load_mnist_model, load_cifar10_model, load_cifar100_model
from models.models import MnistCNN, Cifar10CNN, Cifar100CNN


@pytest.fixture
def mock_model():
    # Tworzymy prosty model testowy
    model = MnistCNN()  # Można również zmienić na Cifar10CNN lub Cifar100CNN
    for param in model.parameters():
        param.data.fill_(0.1)
    return model


@pytest.fixture
def mock_path(tmpdir):
    # Tworzymy tymczasową ścieżkę do zapisania modelu
    return os.path.join(tmpdir, "model.pth")


def test_save_model(mock_model, mock_path):
    # Testowanie zapisu modelu
    save_model(mock_model, mock_path)

    # Sprawdzamy, czy plik został zapisany
    assert os.path.exists(mock_path), "Model file was not saved."


def test_load_mnist_model(mock_path):
    # Testowanie ładowania modelu MNIST
    device = torch.device("cpu")
    model = load_mnist_model(mock_path, device)

    # Sprawdzamy, czy model jest instancją MnistCNN
    assert isinstance(model, MnistCNN), "Loaded model is not of type MnistCNN."


def test_load_cifar10_model(mock_path):
    # Testowanie ładowania modelu CIFAR10
    device = torch.device("cpu")
    model = load_cifar10_model(mock_path, device)

    # Sprawdzamy, czy model jest instancją Cifar10CNN
    assert isinstance(model, Cifar10CNN), "Loaded model is not of type Cifar10CNN."


def test_load_cifar100_model(mock_path):
    # Testowanie ładowania modelu CIFAR100
    device = torch.device("cpu")
    model = load_cifar100_model(mock_path, device)

    # Sprawdzamy, czy model jest instancją Cifar100CNN
    assert isinstance(model, Cifar100CNN), "Loaded model is not of type Cifar100CNN."


def test_load_model_invalid_path():
    # Testowanie błędu przy próbie załadowania modelu z nieistniejącej ścieżki
    device = torch.device("cpu")
    with pytest.raises(FileNotFoundError):
        load_mnist_model("invalid_path.pth", device)
