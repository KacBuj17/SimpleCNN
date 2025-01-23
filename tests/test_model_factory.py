import pytest
import torch
from unittest.mock import patch, MagicMock
from models.model_factory import ModelFactory
from models.models import MnistCNN, Cifar10CNN, Cifar100CNN
from models.model_utils import load_mnist_model, load_cifar10_model, load_cifar100_model


@pytest.fixture
def mock_device():
    # Tworzymy mock urządzenia
    return torch.device("cpu")


@pytest.fixture
def mock_path():
    # Ścieżka do modelu (używana w testach ładowania)
    return "mock_model.pth"


def test_create_train_model_mnist():
    model_type = "mnist"
    learning_rate = 0.001

    model, criterion, optimizer = ModelFactory.create_train_model(model_type, learning_rate)

    # Testowanie, czy zwrócony model jest instancją MnistCNN
    assert isinstance(model, MnistCNN)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(optimizer, torch.optim.Adam)


def test_create_train_model_cifar10():
    model_type = "cifar10"
    learning_rate = 0.01

    model, criterion, optimizer = ModelFactory.create_train_model(model_type, learning_rate)

    # Testowanie, czy zwrócony model jest instancją Cifar10CNN
    assert isinstance(model, Cifar10CNN)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(optimizer, torch.optim.SGD)


def test_create_train_model_cifar100():
    model_type = "cifar100"
    learning_rate = 0.01

    model, criterion, optimizer = ModelFactory.create_train_model(model_type, learning_rate)

    # Testowanie, czy zwrócony model jest instancją Cifar100CNN
    assert isinstance(model, Cifar100CNN)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(optimizer, torch.optim.SGD)


def test_create_train_model_invalid():
    model_type = "invalid_model"
    learning_rate = 0.001

    # Testowanie, czy dla nieznanego modelu rzucany jest wyjątek
    with pytest.raises(ValueError):
        ModelFactory.create_train_model(model_type, learning_rate)


def test_create_eval_model_mnist(mock_path, mock_device):
    model_type = "mnist"

    # Zamiast ładować model, mockujemy funkcję ładowania
    with patch("models.model_factory.load_mnist_model") as mock_load_mnist_model:
        mock_model = MagicMock(spec=MnistCNN)
        mock_load_mnist_model.return_value = mock_model

        model = ModelFactory.create_eval_model(mock_path, mock_device, model_type)

        # Testowanie, czy zwrócony model jest tym, który załadowaliśmy
        mock_load_mnist_model.assert_called_once_with(mock_path, mock_device)
        assert model == mock_model


def test_create_eval_model_cifar10(mock_path, mock_device):
    model_type = "cifar10"

    with patch("models.model_factory.load_cifar10_model") as mock_load_cifar10_model:
        mock_model = MagicMock(spec=Cifar10CNN)
        mock_load_cifar10_model.return_value = mock_model

        model = ModelFactory.create_eval_model(mock_path, mock_device, model_type)

        # Testowanie, czy zwrócony model jest tym, który załadowaliśmy
        mock_load_cifar10_model.assert_called_once_with(mock_path, mock_device)
        assert model == mock_model


def test_create_eval_model_cifar100(mock_path, mock_device):
    model_type = "cifar100"

    with patch("models.model_factory.load_cifar100_model") as mock_load_cifar100_model:
        mock_model = MagicMock(spec=Cifar100CNN)
        mock_load_cifar100_model.return_value = mock_model

        model = ModelFactory.create_eval_model(mock_path, mock_device, model_type)

        # Testowanie, czy zwrócony model jest tym, który załadowaliśmy
        mock_load_cifar100_model.assert_called_once_with(mock_path, mock_device)
        assert model == mock_model


def test_create_eval_model_invalid(mock_path, mock_device):
    model_type = "invalid_model"

    with pytest.raises(ValueError):
        ModelFactory.create_eval_model(mock_path, mock_device, model_type)
