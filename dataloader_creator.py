import numpy
import torch
import torch.cuda
import torch.nn.functional as Fun
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from typing import List

MAX_NUM_LETTERS = 20
LEN_WORD = 9
BATCH_SIZE = 64
unique_letters = ['A', 'R', 'D', 'N', 'C', 'Q', 'E', 'G', 'H', 'I',
                  'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def word_list_to_tensor(word_list: List[str]):
    """

    :param word_list: list of words which need to be made to dataloader
    :return:
    """
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(unique_letters)

    word_list = [le.transform(list(row.strip())) for row in word_list]
    word_list = numpy.array(word_list)
    tensor = Fun.one_hot(torch.Tensor(word_list).to(torch.int64), num_classes=MAX_NUM_LETTERS)
    tensor = tensor.to(torch.float32)
    return tensor

def create_tensor(file_path: str):
    """

    :param file_path: to open and cretae tensor of
    :return: tensor one hot encoded
    """
    with open(file_path, 'r') as file:
        f = file.readlines()
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(unique_letters)

    return word_list_to_tensor(f)


def duplicate_ones(ones_x, ones_y, len):
    indices = torch.randint(low=0, high=ones_x.size(0), size=(len,))
    new_tensor_x = ones_x[indices]
    new_tensor_y = ones_y[indices]

    return new_tensor_x, new_tensor_y


def get_dataloaders_after_split(neg_path, pos_path):
    """
    :param neg_path: file path in with the negative patience are
    :param pos_path: file path in with the positive patience are
    Create Dataloaders from the specified files and put it with the needed labels.
    :return: dataloaders for train and test
    """

    # creation of base tensors
    neg_tensor = create_tensor(neg_path)
    neg_labels = torch.cat((torch.zeros(len(neg_tensor), 1), torch.ones(len(neg_tensor), 1)), dim=1)
    pos_tensor = create_tensor(pos_path)
    pos_labels = torch.cat((torch.ones(len(pos_tensor), 1), torch.zeros(len(pos_tensor), 1)), dim=1)

    # to make around the same amount of positive samples and splitting data
    dataset = torch.cat([pos_tensor, neg_tensor], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1, shuffle=True)

    X_one_train_duplicate, y_one_train_duplicate = duplicate_ones(X_train[y_train[:, 0] == 1.0],
                                                                  y_train[y_train[:, 0] == 1.0],
                                                                  len(X_train[y_train[:, 0] == 0.0]))

    final_X_train = torch.cat([X_train[y_train[:, 0] == 0.0], X_one_train_duplicate], dim=0)
    final_y_train = torch.cat([y_train[y_train[:, 0] == 0.0], y_one_train_duplicate], dim=0)

    # creation of the dataloader itself
    train_dataset = TensorDataset(final_X_train, final_y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, test_dataloader
