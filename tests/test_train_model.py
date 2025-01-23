import pytest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from scripts.train import train_model


@pytest.fixture
def mock_data():
    train_data = [(torch.randn(3, 32, 32), torch.tensor(0)) for _ in range(10)]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=2)
    return train_loader


@pytest.fixture
def mock_model():
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 32 * 32, 10)
    )
    return model


def test_train_model(mock_model, mock_data):
    model = mock_model
    train_loader = mock_data
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cpu")

    optimizer.step = MagicMock()

    train_model(model, train_loader, criterion, optimizer, device, num_epochs=1)

    assert model.training, "Model should be in training mode during training."

    assert optimizer.step.called, "Optimizer's step method should be called during training."
