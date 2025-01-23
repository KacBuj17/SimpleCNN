import pytest
import torch
from unittest.mock import MagicMock
from PIL import Image
from scripts.predict import preprocess_image, predict
from models.models import MnistCNN, Cifar10CNN


@pytest.fixture
def mock_image_path(tmp_path):
    image_path = tmp_path / "mock_image.png"
    image = Image.new("RGB", (32, 32), color="white")  # Tworzy biały obraz 32x32
    image.save(image_path)
    return str(image_path)


@pytest.fixture
def mock_image():
    return MagicMock(spec=Image.Image)


@pytest.fixture
def mock_mnist_model():
    return MagicMock(spec=MnistCNN)


@pytest.fixture
def mock_cifar_model():
    return MagicMock(spec=Cifar10CNN)


@pytest.fixture
def mock_device():
    return torch.device("cpu")


def test_preprocess_image_mnist(mock_image_path):
    image_tensor = preprocess_image(mock_image_path, model_type="mnist")

    assert image_tensor.shape == (1, 1, 28, 28)


def test_preprocess_image_cifar(mock_image_path):
    image_tensor = preprocess_image(mock_image_path, model_type="cifar10")

    assert image_tensor.shape == (1, 3, 32, 32)


def test_predict_mnist(mock_mnist_model, mock_image_path, mock_device):
    mock_image_tensor = torch.randn(1, 1, 28, 28)

    mock_mnist_model.return_value = torch.randn(1, 10)

    predicted_class, confidence = predict(mock_mnist_model, mock_image_tensor, mock_device)

    assert isinstance(predicted_class, int)
    assert 0 <= confidence <= 100


def test_predict_cifar(mock_cifar_model, mock_image_path, mock_device):
    mock_image_tensor = torch.randn(1, 3, 32, 32)

    mock_cifar_model.return_value = torch.randn(1, 10)

    predicted_class, confidence = predict(mock_cifar_model, mock_image_tensor, mock_device)

    assert isinstance(predicted_class, int)
    assert 0 <= confidence <= 100


def test_preprocess_image_invalid_model(mock_image_path):
    with pytest.raises(ValueError):
        preprocess_image(mock_image_path, model_type="invalid_model")
