import os
import pytest
import torch
from models.model_utils import save_model, load_mnist_model, load_cifar10_model, load_cifar100_model
from models.models import MnistCNN, Cifar10CNN, Cifar100CNN


@pytest.fixture
def mock_mnist_model():
    model = MnistCNN()
    for param in model.parameters():
        param.data.fill_(0.1)
    return model


@pytest.fixture
def mock_cifar10_model():
    model = Cifar10CNN()
    for param in model.parameters():
        param.data.fill_(0.1)
    return model


@pytest.fixture
def mock_cifar100_model():
    model = Cifar100CNN()
    for param in model.parameters():
        param.data.fill_(0.1)
    return model


@pytest.fixture
def mock_path(tmp_path):
    return os.path.join(tmp_path, "model.pth")


@pytest.mark.parametrize(
    "model_fixture, loader_function, model_type",
    [
        ("mock_mnist_model", load_mnist_model, MnistCNN),
        ("mock_cifar10_model", load_cifar10_model, Cifar10CNN),
        ("mock_cifar100_model", load_cifar100_model, Cifar100CNN),
    ],
)
def test_save_and_load_model(request, model_fixture, loader_function, model_type, mock_path):
    model = request.getfixturevalue(model_fixture)

    save_model(model, mock_path)
    assert os.path.exists(mock_path), "Model file was not saved."

    device = torch.device("cpu")
    loaded_model = loader_function(mock_path, device)

    assert isinstance(loaded_model, model_type), f"Loaded model is not of type {model_type.__name__}."


def test_load_model_invalid_path():
    device = torch.device("cpu")
    with pytest.raises(FileNotFoundError):
        load_mnist_model("invalid_path.pth", device)
