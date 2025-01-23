import pytest
from unittest.mock import patch
import torch
import torch.nn as nn
from trainers.model_trainer import ModelTrainer


@pytest.fixture
def mock_data():
    train_data = torch.randn(10, 3, 32, 32)
    train_labels = torch.randint(0, 10, (10,))
    test_data = torch.randn(5, 3, 32, 32)
    test_labels = torch.randint(0, 10, (5,))

    train_loader = torch.utils.data.DataLoader(list(zip(train_data, train_labels)), batch_size=2)
    test_loader = torch.utils.data.DataLoader(list(zip(test_data, test_labels)), batch_size=2)
    return train_loader, test_loader


@pytest.fixture
def model_trainer(mock_data):
    train_loader, test_loader = mock_data
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 32 * 32, 10)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    device = torch.device("cpu")
    return ModelTrainer(model, train_loader, test_loader, criterion, optimizer, device)


def test_train_method(model_trainer):
    with patch("trainers.model_trainer.train_model") as mock_train_model:
        mock_train_model.return_value = None
        model_trainer.train(num_epochs=1)
        mock_train_model.assert_called_once()


def test_test_method(model_trainer):
    with patch("trainers.model_trainer.test_model") as mock_test_model:
        mock_test_model.return_value = 85.0
        model_trainer.test()
        mock_test_model.assert_called_once()


def test_save_method(model_trainer):
    with patch("trainers.model_trainer.save_model") as mock_save_model:
        mock_save_model.return_value = None
        model_trainer.save("dummy_path.pth")
        mock_save_model.assert_called_once_with(model_trainer.model, "dummy_path.pth")

