Dev env install:
```commandline
pip3 install -e .
```

Requirements
```commandline
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu121
```

Main run example:
```commandline
python3 scripts/main.py --model_type cifar100 --learning_rate 0.001 --batch_size 32 --num_epochs 5 --model_save_path "./trained_models/Cifar100CNN.pth"
```

Eval run example:
```commandline
python3 scripts/evaluate.py --model_type mnist --batch_size 32 --model_path ./trained_models/MNIST_model.pth
```