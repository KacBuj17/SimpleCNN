import pytest
import torch
from models.models import MnistCNN, Cifar10CNN, Cifar100CNN


@pytest.mark.parametrize("model_class, input_shape, output_size", [
    (MnistCNN, (1, 1, 28, 28), 10),     # MNIST: 1 kanał, 28x28, 10 klas
    (Cifar10CNN, (1, 3, 32, 32), 10),  # CIFAR-10: 3 kanały, 32x32, 10 klas
    (Cifar100CNN, (1, 3, 32, 32), 100) # CIFAR-100: 3 kanały, 32x32, 100 klas
])
def test_model_forward_pass(model_class, input_shape, output_size):
    model = model_class()  # Inicjalizacja modelu
    model.eval()           # Wyłączenie dropout dla testów

    # Dane wejściowe
    input_data = torch.randn(*input_shape)

    # Przepuszczenie danych przez model
    output = model(input_data)

    # Sprawdzenie wymiaru wyjścia
    assert output.shape == (input_shape[0], output_size), \
        f"Expected output shape {(input_shape[0], output_size)}, got {output.shape}"


@pytest.mark.parametrize("model_class, input_shape", [
    (MnistCNN, (1, 1, 28, 28)),    # MNIST
    (Cifar10CNN, (1, 3, 32, 32)), # CIFAR-10
    (Cifar100CNN, (1, 3, 32, 32)) # CIFAR-100
])
def test_model_backward_pass(model_class, input_shape):
    model = model_class()  # Inicjalizacja modelu
    model.train()          # Włączenie dropout dla testów

    # Dane wejściowe i cele (targety)
    input_data = torch.randn(*input_shape, requires_grad=True)
    target = torch.randint(0, model.fc3.out_features, (input_shape[0],))

    # Funkcja strat
    criterion = torch.nn.CrossEntropyLoss()

    # Forward pass
    output = model(input_data)

    # Oblicz stratę
    loss = criterion(output, target)

    # Backward pass
    loss.backward()

    # Sprawdzenie, czy gradienty są obliczane dla parametrów
    for param in model.parameters():
        assert param.grad is not None, "Gradient not computed for some parameters"
