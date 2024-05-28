import torch
from scipy import datasets
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import main

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MAX_NUM_LETTERS = 20
LEN_WORD = 9

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20*9, 20*9),
            nn.ReLU(),
            nn.Linear(20*9, 20*9),
            nn.ReLU(),
            nn.Linear(20*9, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_module(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_function(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_module(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def parse_name_files_from_tar(name_tar):
    import tarfile

    # Open the tar file (use 'r:gz' for gzip, 'r:bz2' for bzip2, or 'r:xz' for xz compressed tar files)
    with tarfile.open('example.tar.gz', 'r:gz') as tar:
        # List all members in the tar file
        for member in tar.getmembers():
            print(member.name)

        # Extract a specific file
        file_to_extract = 'path/inside/tar/file.txt'
        extracted_file = tar.extractfile(file_to_extract)
if __name__ == "__main__":
    # training_data = datasets.FashionMNIST(
    #     root="data",
    #     train=True,
    #     download=True,
    #     transform=ToTensor(),
    # )
    #
    # # Download test data from open datasets.
    # test_data = datasets.FashionMNIST(
    #     root="data",
    #     train=False,
    #     download=True,
    #     transform=ToTensor(),
    # )

    batch_size = 64
    train_dataloader, test_dataloader = main.get_dataloaders()
    # Create data loaders.
    # train_dataloader = DataLoader(training_data, batch_size=batch_size)
    # test_dataloader = DataLoader(test_data, batch_size=batch_size)


    # train_dataloader, test_dataloader = main.get_dataloaders()
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)

    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_module(train_dataloader, model, loss_fn, optimizer)
        test_module(test_dataloader, model, loss_fn)
    print("Done!")
