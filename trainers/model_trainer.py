import time

from scripts.test import test_model
from scripts.train import train_model
from models.model_utils import save_model


class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs: int):
        print("Training model...")
        start = time.time()
        train_model(self.model, self.train_loader, self.criterion, self.optimizer, self.device, num_epochs)
        stop = time.time()
        print(f"Training completed in {stop - start:.2f} seconds.")

    def test(self):
        print("Testing model...")
        accuracy = test_model(self.model, self.test_loader, self.device)
        print(f"Test Accuracy: {accuracy:.2f}%")

    def save(self, file_path: str):
        print("Saving model...")
        save_model(self.model, file_path)
