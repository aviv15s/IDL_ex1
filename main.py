import torch
import tarfile
import os

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
model = torch.load('ex1 data.tar', map_location='cpu')