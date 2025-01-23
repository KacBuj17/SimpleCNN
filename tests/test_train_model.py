import pytest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from scripts.train import train_model


@pytest.fixture
def mock_data():
    # Tworzymy przykładowe dane treningowe
    train_data = [(torch.randn(1, 3, 32, 32), torch.tensor(0)) for _ in range(10)]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=2)
    return train_loader


@pytest.fixture
def mock_model():
    # Tworzymy prosty model testowy
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 32 * 32, 10)
    )
    return model


def test_train_model(mock_model, mock_data):
    # Przygotowanie modelu, kryterium, optymalizatora i urządzenia
    model = mock_model
    train_loader = mock_data
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cpu")

    # Mockowanie metody optymalizatora, aby upewnić się, że jest wywoływana
    optimizer.step = MagicMock()

    # Wywołanie funkcji train_model
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=1)

    # Sprawdzanie, czy model jest w trybie treningowym
    assert model.training, "Model should be in training mode during training."

    # Sprawdzanie, czy optymalizator został wywołany
    assert optimizer.step.called, "Optimizer's step method should be called during training."
