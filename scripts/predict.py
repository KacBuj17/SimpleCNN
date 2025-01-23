import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image


def preprocess_image(image_path, model_type):
    if model_type == "mnist":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    image = Image.open(image_path)
    image = transform(image)
    return image.unsqueeze(0)

def predict(model, image_tensor, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor.to(device))

        probabilities = F.softmax(outputs, dim=1)

        predicted_class = torch.argmax(probabilities, 1).item()

        confidence = probabilities[0, predicted_class].item() * 100

    return predicted_class, confidence
