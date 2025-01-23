import pytest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from trainers.model_trainer import ModelTrainer


@pytest.fixture
def mock_data():
    # Tworzymy przykładowe dane treningowe i testowe
    train_data = [(torch.randn(1, 3, 32, 32), torch.tensor(0)) for _ in range(10)]
    test_data = [(torch.randn(1, 3, 32, 32), torch.tensor(0)) for _ in range(5)]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=2)

    return train_loader, test_loader


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


@pytest.fixture
def model_trainer(mock_model, mock_data):
    train_loader, test_loader = mock_data
    model = mock_model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    device = torch.device("cpu")
    return ModelTrainer(model, train_loader, test_loader, criterion, optimizer, device)


def test_train_method(model_trainer):
    with patch("scripts.train.train_model") as mock_train_model:
        # Mockujemy funkcję train_model
        mock_train_model.return_value = None

        # Wywołujemy metodę train
        model_trainer.train(num_epochs=1)

        # Sprawdzamy, czy funkcja train_model została wywołana
        mock_train_model.assert_called_once_with(
            model_trainer.model,
            model_trainer.train_loader,
            model_trainer.criterion,
            model_trainer.optimizer,
            model_trainer.device,
            1  # num_epochs
        )


def test_test_method(model_trainer):
    with patch("scripts.test.test_model") as mock_test_model:
        # Mockujemy funkcję test_model
        mock_test_model.return_value = 85.0

        # Wywołujemy metodę test
        model_trainer.test()

        # Sprawdzamy, czy funkcja test_model została wywołana
        mock_test_model.assert_called_once_with(
            model_trainer.model,
            model_trainer.test_loader,
            model_trainer.device
        )


def test_save_method(model_trainer):
    with patch("models.model_utils.save_model") as mock_save_model:
        # Mockujemy funkcję save_model
        mock_save_model.return_value = None

        # Wywołujemy metodę save
        model_trainer.save("dummy_path.pth")

        # Sprawdzamy, czy funkcja save_model została wywołana
        mock_save_model.assert_called_once_with(model_trainer.model, "dummy_path.pth")
