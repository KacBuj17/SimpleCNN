import torch
import argparse

from test import test_model
from models.model_factory import ModelFactory
from data.data_loader import get_data_loaders


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate a neural network model.")
    parser.add_argument("--model_type", type=str, default="mnist", choices=["mnist", "cifar10", "cifar100"],
                        help="Type of the model to train: mnist, cifar10, cifar100 (default: mnist)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for data loaders (default: 64 for CIFAR trained_models, 64 for MNIST)")
    parser.add_argument("--model_path", type=str, default="./trained_models/MNISTCNN.pth",
                        help="Path to save the trained model (default: ./trained_models/MNISTCNN.pth)")
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    print(torch.__version__)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    try:
        model = ModelFactory.create_eval_model(args.model_path, device, args.model_type)
        _, test_loader = get_data_loaders(model, batch_size=args.batch_size)
    except ValueError as e:
        print(e)
        return

    accuracy = test_model(model, test_loader, device)

    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
