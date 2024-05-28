import matplotlib.pyplot as plt
import torch
from torch import nn

import dataloader_creator

MAX_NUM_LETTERS = 20
LEN_WORD = 9
BATCH_SIZE = 64

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class OverfittingModel(nn.Module):
    """
     This model consists of a flattening layer followed by a sequence of linear layers
    with ReLU activations. The number of neurons in the hidden layers is the same as the
    input size, which can lead to overfitting on small datasets.

    Attributes:
    -----------
    flatten : nn.Flatten
        A layer that flattens the input tensor.
    linear_relu_stack : nn.Sequential
        A sequence of linear layers with ReLU activations.

    Methods:
    --------
    forward(x):
        Defines the forward pass of the model.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(MAX_NUM_LETTERS * LEN_WORD, MAX_NUM_LETTERS * LEN_WORD),
            nn.ReLU(),
            nn.Linear(MAX_NUM_LETTERS * LEN_WORD, MAX_NUM_LETTERS * LEN_WORD),
            nn.ReLU(),
            nn.Linear(MAX_NUM_LETTERS * LEN_WORD, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NonOverfittingModel(nn.Module):
    """
    A neural network model designed to avoid overfitting on a classification task.

    This model consists of a flattening layer followed by a sequence of linear layers
    with ReLU activations. The architecture is simple to prevent overfitting on small datasets.

    Attributes:
    -----------
    flatten : nn.Flatten
        A layer that flattens the input tensor.
    linear_relu_stack : nn.Sequential
        A sequence of linear layers with ReLU activations.

    Methods:
    --------
    forward(x):
        Defines the forward pass of the model.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(MAX_NUM_LETTERS * LEN_WORD, 9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Linear(9, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_module(dataloader, model, loss_function, optimizer):
    """
    Train the model on the given dataset for one epoch.

    :param dataloader: torch.utils.data.DataLoader
        DataLoader for the training dataset.
    :param model: torch.nn.Module
        The neural network model to be trained.
    :param loss_function: torch.nn.Module
        Loss function used to compute the loss.
    :param optimizer: torch.optim.Optimizer
        Optimizer used to update the model parameters.
    :return: float
        The average training loss over all batches.
    """
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_function(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def test_module(dataloader, model, loss_fn):
    """
    Evaluate the model on the test dataset and return the average loss.

    :param dataloader: torch.utils.data.DataLoader
        DataLoader for the test dataset.
    :param model: torch.nn.Module
        The neural network model to be evaluated.
    :param loss_fn: torch.nn.Module
        Loss function used to compute the loss.
    :return: float
        The average test loss over all batches.
    """
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
    return test_loss


def plot_epochs_loss(train_loss_list, test_loss_list):
    """
    Plot the training and test loss per epoch.

    :param train_loss_list: list
        List of training loss values per epoch.
    :param test_loss_list: list
        List of test loss values per epoch.
    :return: None
    """
    plt.plot(list(range(1, len(train_loss_list) + 1)), train_loss_list, label='train loss')
    plt.plot(list(range(1, len(test_loss_list) + 1)), test_loss_list, label='test loss')
    plt.xlabel("# epoch")
    plt.ylabel('loss [arb]')
    plt.title("Train and Test loss per epoch")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_dataloader, test_dataloader = dataloader_creator.get_dataloaders('neg_A0201.txt', 'pos_A0201.txt')
    model = NonOverfittingModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)

    train_loss_list, test_loss_list = [], []
    epochs = 500
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train_module(train_dataloader, model, loss_fn, optimizer)
        test_loss = test_module(test_dataloader, model, loss_fn)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    plot_epochs_loss(train_loss_list, test_loss_list)
