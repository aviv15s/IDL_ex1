import numpy
import torch
import torch.cuda
import torch.nn.functional as Fun
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


unique_letters = ['A', 'R', 'D', 'N', 'C', 'Q', 'E', 'G', 'H', 'I',
                  'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
MAX_NUM_LETTERS = 20
LEN_WORD = 9

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

def get_dataloaders():
    device = ("cuda"
              if torch.cuda.is_available() else "cpu")
    neg_path = 'neg_A0201.txt'
    pos_path = 'pos_A0201.txt'

    neg_tensor = create_tensor(neg_path)
    neg_labels = torch.zeros(len(neg_tensor))
    pos_tensor = create_tensor(pos_path)
    pos_labels = torch.ones(len(pos_tensor))

    # to make the same amount of samples
    pos_tensor_extended = pos_tensor * 6
    pos_labels_extended = pos_labels * 6

    dataset = torch.cat(pos_tensor_extended , neg_tensor)
    labels = pos_labels_extended + neg_labels
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1, shuffle=True)
    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test, y_test)
    batch = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=True)


    # TensorDataset(train, labeltessor)
    # DataLoader(TEnsordats)

    return train_dataloader, test_dataloader

