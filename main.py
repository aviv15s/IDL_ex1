import matplotlib.pyplot as plt
import torch
import dataloader_creator
from torch import nn
from dataloader_creator import MAX_NUM_LETTERS,LEN_WORD,BATCH_SIZE


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
            nn.Linear(MAX_NUM_LETTERS * LEN_WORD, 12),
            nn.ReLU(),
            nn.Linear(12, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NonOverfittingLinearModel(nn.Module):
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
            nn.Linear(MAX_NUM_LETTERS * LEN_WORD, 12),
            nn.Linear(12, 2)
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
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(dataloader)


def module_test(dataloader, model, loss_fn):
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
    print(f"accuracy: {correct}")
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

def spike_protein_test(model):
    file_path = "spike.txt"
    with open(file_path, 'r') as file:
        f = file.readlines()
        protein_data = "".join(["".join(line.split()) for line in f])

    list_words = [protein_data[i:i + 9] for i in range(len(protein_data) - 9)]
    dataset = dataloader_creator.word_list_to_tensor(list_words)

    model.eval()
    prediction = model(dataset)
    prediction = nn.Softmax(dim=1)(prediction)

    peptides_with_probs = [(list_words[i], prediction[i][0].item()) for i in range(len(list_words))]
    sorted_peptides = sorted(peptides_with_probs, key=lambda x: x[1], reverse=True)
    top3 = sorted_peptides[:3]

    for index, (peptide, prob) in enumerate(top3):
        print(f"{index + 1}. peptide {peptide} has probability {prob} of being positive")


# def run_model(model, train_dataloader, test_dataloader)

def plot_model_loss_graph(train_dataloader, test_dataloader, model, loss_fn, optimizer):
    train_loss_list, test_loss_list = [], []
    epochs = 500
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train_module(train_dataloader, model, loss_fn, optimizer)
        test_loss = module_test(test_dataloader, model, loss_fn)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print(f'Train loss: {train_loss}, Test loss: {test_loss}')

    plot_epochs_loss(train_loss_list, test_loss_list)


if __name__ == "__main__":
    train_dataloader, test_dataloader = dataloader_creator.get_dataloaders_after_split('neg_A0201.txt', 'pos_A0201.txt')
    loss_fn = nn.CrossEntropyLoss()
    print("Uncomment the code to run the different sections")

    # # 2b
    # overfitting_model = OverfittingModel().to(device)
    # optimizer = torch.optim.SGD(overfitting_model.parameters(), lr=5e-3)
    # plot_model_loss_graph(train_dataloader, test_dataloader, overfitting_model, loss_fn, optimizer)
    # torch.save(overfitting_model.state_dict(), 'overfitting_model_weights.pth')

    # 2c
    # nonOverfitting_model = NonOverfittingModel().to(device)
    # optimizer = torch.optim.SGD(nonOverfitting_model.parameters(), lr=5e-3)
    # plot_model_loss_graph(train_dataloader, test_dataloader, nonOverfitting_model, loss_fn, optimizer)
    # torch.save(nonOverfitting_model.state_dict(), 'nonOverfitting_model_weights.pth')

    # 2d
    # nonOverfittingLinear_model = NonOverfittingLinearModel().to(device)
    # optimizer = torch.optim.SGD(nonOverfittingLinear_model.parameters(), lr=5e-3)
    # plot_model_loss_graph(train_dataloader, test_dataloader, nonOverfittingLinear_model, loss_fn, optimizer)
    # torch.save(nonOverfittingLinear_model.state_dict(), 'nonOverfittingLinear_model_weights.pth')

    # 2e
    # model = NonOverfittingModel().to(device)
    # model.eval()
    # try:
    #     model.load_state_dict(torch.load('nonOverfitting_model_weights.pth'))
    # except:
    #     print("Need to run code of 2c before running this code section to train model!")
    #     exit()
    # spike_protein_test(model)
