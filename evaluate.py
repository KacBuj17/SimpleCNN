import torch
from save_load import load_model
from data_loader import get_data_loaders
from test import compute_accuracy


def evaluate_model(model, data_loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = compute_accuracy(correct, total)

    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.__version__)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    batch_size = 64
    model_path = './models/SimpleCNN_model.pth'

    model = load_model(model_path, device)

    _, test_loader = get_data_loaders(batch_size=batch_size)

    accuracy = evaluate_model(model, test_loader, device)

    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
