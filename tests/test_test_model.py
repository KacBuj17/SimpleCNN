import pytest
import torch
import torch.nn as nn
from scripts.test import test_model, compute_accuracy


@pytest.fixture
def mock_data():
    test_data = [(torch.randn(3, 32, 32), torch.tensor(0)) for _ in range(10)]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=2)
    return test_loader


@pytest.fixture
def mock_model():
    mock_model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 32 * 32, 10)
    )
    for param in mock_model.parameters():
        param.data.fill_(0.1)
    return mock_model


def test_compute_accuracy():
    correct = 75
    total = 100
    accuracy = compute_accuracy(correct, total)
    assert accuracy == 75.0, "Accuracy calculation is incorrect."


def test_test_model(mock_model, mock_data):
    device = torch.device("cpu")

    test_loader = mock_data

    accuracy = test_model(mock_model, test_loader, device)

    assert isinstance(accuracy, float), "Accuracy should be a float."
    assert 0 <= accuracy <= 100, "Accuracy should be between 0 and 100."



