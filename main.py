import numpy
import torch
import torch.cuda
import torch.nn.functional as Fun
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset

MAX_NUM_LETTERS = 20
LEN_WORD = 9
unique_letters = ['A', 'R', 'D', 'N', 'C', 'Q', 'E', 'G', 'H', 'I',
                  'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def create_tensor(file_path: str):
    with open(file_path, 'r') as file:
        f = file.readlines()
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(unique_letters)

    list_words = []
    for row in f:
        list_words.append(le.transform(list(row.strip())))
    list_words = numpy.array(list_words)
    tensor = Fun.one_hot(torch.Tensor(list_words).to(torch.int64), num_classes=MAX_NUM_LETTERS)
    return tensor


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(LEN_WORD * MAX_NUM_LETTERS, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item()
            print(f"loss: {loss:>7f}")


if __name__ == "__main__":
    device = ("cuda"
              if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().to(device)
    neg_path = 'neg_A0201.txt'
    pos_path = 'pos_A0201.txt'

    neg_tensor = create_tensor(neg_path)
    neg_labels = torch.zeros(len(neg_tensor))
    pos_tensor = create_tensor(pos_path)
    pos_labels = torch.ones(len(pos_tensor))

    # TensorDataset
    # dataset = torch.utils.data.ConcatDataset([neg_tensor, pos_tensor])
    # labels = torch.utils.data.ConcatDataset([neg_labels, pos_labels])

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1, shuffle=True)



    # TensorDataset(train, labeltessor)
    # DataLoader(TEnsordats)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())

    epochs =5
    for t in range(epochs):
        print(f"Epoch {t+1}\n________")
        train(train_dataloader,model,loss_fn,optimizer)
        test(test_dataloader, model, loss_fn)


