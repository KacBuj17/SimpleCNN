import pytest
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
from scripts.predict import preprocess_image, predict
from models.models import MnistCNN, Cifar10CNN


@pytest.fixture
def mock_image_path():
    # Ścieżka do obrazu, który będziemy przetwarzać
    return "tests/mock_image.png"


@pytest.fixture
def mock_image():
    # Mock obrazu, który będzie używany w testach
    return MagicMock(spec=Image.Image)


@pytest.fixture
def mock_model():
    # Mock modelu, który będzie używany w testach
    return MagicMock(spec=MnistCNN)


@pytest.fixture
def mock_device():
    # Mock urządzenia (np. CPU lub GPU)
    return torch.device("cpu")


def test_preprocess_image_mnist(mock_image_path):
    # Testowanie preprocessora dla modelu MNIST
    image_tensor = preprocess_image(mock_image_path, model_type="mnist")

    # Sprawdzanie kształtu tensoru, który powinien być 1x1x28x28 dla MNIST
    assert image_tensor.shape == (1, 1, 28, 28)


def test_preprocess_image_cifar(mock_image_path):
    # Testowanie preprocessora dla modelu CIFAR
    image_tensor = preprocess_image(mock_image_path, model_type="cifar10")

    # Sprawdzanie kształtu tensoru, który powinien być 1x3x32x32 dla CIFAR
    assert image_tensor.shape == (1, 3, 32, 32)


def test_predict_mnist(mock_model, mock_image_path, mock_device):
    # Testowanie funkcji predict dla MNIST
    mock_image_tensor = torch.randn(1, 1, 28, 28)  # Mockowany tensor obrazu

    # Ustawiamy mock na zwracanie predefiniowanych wyników
    mock_model.return_value = torch.randn(1, 10)  # Przykładowe wyjście modelu (10 klas)

    predicted_class, confidence = predict(mock_model, mock_image_tensor, mock_device)

    # Sprawdzanie, czy funkcja zwróciła poprawną klasę i pewność
    assert isinstance(predicted_class, int)
    assert 0 <= confidence <= 100


def test_predict_cifar(mock_model, mock_image_path, mock_device):
    # Testowanie funkcji predict dla CIFAR
    mock_image_tensor = torch.randn(1, 3, 32, 32)  # Mockowany tensor obrazu

    # Ustawiamy mock na zwracanie predefiniowanych wyników
    mock_model.return_value = torch.randn(1, 10)  # Przykładowe wyjście modelu (10 klas)

    predicted_class, confidence = predict(mock_model, mock_image_tensor, mock_device)

    # Sprawdzanie, czy funkcja zwróciła poprawną klasę i pewność
    assert isinstance(predicted_class, int)
    assert 0 <= confidence <= 100


def test_preprocess_image_invalid_model(mock_image_path):
    # Testowanie przypadku, gdy nieznany model jest podany
    with pytest.raises(ValueError):
        preprocess_image(mock_image_path, model_type="invalid_model")
