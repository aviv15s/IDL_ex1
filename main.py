import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
# Example usage:
file_path = 'neg_A0201.txt'
with open(file_path, 'r') as file:
    f = file.readlines()
data_rows = [line.strip().split(' ') for line in f]
tensor_y = torch.Tensor(data_rows)
my_dataset = TensorDataset(tensor_y) # create your datset
dataloader = DataLoader(my_dataset) #
# Iterate through the DataLoader
for batch in dataloader:
    print(batch)