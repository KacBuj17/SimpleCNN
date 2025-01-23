import torch
import argparse
from models.model_factory import ModelFactory
from data.data_loader import get_data_loaders
from trainers.model_trainer import ModelTrainer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate a neural network model.")
    parser.add_argument("--model_type", type=str, default="cifar100", choices=["mnist", "cifar10", "cifar100"],
                        help="Type of the model to train: mnist, cifar10, cifar100 (default: cifar100)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer (default: 0.001)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for data loaders (default: 4 for CIFAR trained_models, 64 for MNIST)")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--model_save_path", type=str, default="./trained_models/Cifar100CNN_model.pth",
                        help="Path to save the trained model (default: ./trained_models/Cifar100CNN_model.pth)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.__version__)
    print(f"Device used: {device}")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    try:
        model, criterion, optimizer = ModelFactory.create_train_model(args.model_type, args.learning_rate)
        train_loader, test_loader = get_data_loaders(model=model, batch_size=args.batch_size)
    except ValueError as e:
        print(e)
        return

    trainer = ModelTrainer(model, train_loader, test_loader, criterion, optimizer, device)
    trainer.train(args.num_epochs)
    trainer.test()
    trainer.save(args.model_save_path)


if __name__ == "__main__":
    main()
