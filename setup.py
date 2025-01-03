from setuptools import setup, find_packages

setup(
    name="Simple_CNN",
    version="0.1",
    description="Simple CNN project with MNIST, CIFAR10 and CIFAR100",
    author="Kacper Bujak",
    author_email="kacbuj@student.agh.edu.pl",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "argparse",
    ],
    python_requires=">=3.9",
)
