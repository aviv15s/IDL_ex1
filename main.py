import torch
from torch.utils.data import TensorDataset, DataLoader

MAX_NUM_LETTERS = 20
import torch.nn.functional as Fun
dict ={}
def one_hot(tensor: torch.tensor):
    return Fun.one_hot(tensor, num_classes = MAX_NUM_LETTERS)
def create_dataloader(file_path:str):

    data_rows = [line.strip().split(' ') for line in f]
    tensor = torch.Tensor(data_rows)
    my_dataset = TensorDataset(tensor)  # create your datset
    dataloader = DataLoader(my_dataset)  #
    # Iterate through the DataLoader
    for batch in dataloader:
        print(batch)
if __name__ == "__main__":
    file_path = 'neg_A0201.txt'
    with open(file_path, 'r') as file:
        f = file.readlines()
    [for line in f]
    create_dataloader('neg_A0201.txt')

