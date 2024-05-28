import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torchvision import datasets
import pandas as pd

def split_data(dataset, test_ratio):
    test_size = test_ratio * len(dataset)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


if __name__ == "__main__":

    print("hello world")