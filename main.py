import torch
import torch.optim as optim
import torch.nn as nn
from model import SimpleCNN
from data_loader import get_data_loaders
from train import train_model
from test import test_model
from save_load import save_model
import time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.__version__)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    batch_size = 64
    num_epochs = 5
    learning_rate = 0.001

    model_name = 'SimpleCNN_model.pth'

    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    model = SimpleCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training model...")
    start = time.time()
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)
    stop = time.time()

    print(f"Time of training: {stop - start} seconds.")

    print("Testing model...")
    test_model(model, test_loader, device)

    print("Saving model...")
    save_model(model, f'./models/{model_name}')


if __name__ == "__main__":
    main()
