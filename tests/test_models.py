import pytest
import torch
from models.models import MnistCNN, Cifar10CNN, Cifar100CNN


@pytest.mark.parametrize("model_class, input_shape, output_size", [
    (MnistCNN, (1, 1, 28, 28), 10),
    (Cifar10CNN, (1, 3, 32, 32), 10),
    (Cifar100CNN, (1, 3, 32, 32), 100),
])
def test_model_forward_pass(model_class, input_shape, output_size):
    model = model_class()
    model.eval()

    input_data = torch.randn(*input_shape)

    output = model(input_data)

    assert output.shape == (input_shape[0], output_size), \
        f"Expected output shape {(input_shape[0], output_size)}, got {output.shape}"


@pytest.mark.parametrize("model_class, input_shape", [
    (MnistCNN, (1, 1, 28, 28)),
    (Cifar10CNN, (1, 3, 32, 32)),
    (Cifar100CNN, (1, 3, 32, 32)),
])
def test_model_backward_pass(model_class, input_shape):
    model = model_class()
    model.train()

    input_data = torch.randn(*input_shape, requires_grad=True)
    target = torch.randint(0, model.fc3.out_features, (input_shape[0],))

    criterion = torch.nn.CrossEntropyLoss()

    output = model(input_data)

    loss = criterion(output, target)

    loss.backward()

    for param in model.parameters():
        assert param.grad is not None, "Gradient not computed for some parameters"
