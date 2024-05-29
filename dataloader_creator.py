import numpy
import torch
import torch.cuda
import torch.nn.functional as Fun
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

MAX_NUM_LETTERS = 20
LEN_WORD = 9
BATCH_SIZE = 64
unique_letters = ['A', 'R', 'D', 'N', 'C', 'Q', 'E', 'G', 'H', 'I',
                  'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def create_tensor(file_path: str):
    """
    :param file_path: to open and cretae tensor of
    :return: tensor one hot encoded
    """
    with open(file_path, 'r') as file:
        f = file.readlines()
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(unique_letters)

    list_words = []
    for row in f:
        list_words.append(le.transform(list(row.strip())))
    list_words = numpy.array(list_words)
    tensor = Fun.one_hot(torch.Tensor(list_words).to(torch.int64), num_classes=MAX_NUM_LETTERS)
    tensor=tensor.to(torch.float32)
    return tensor

def get_dataloaders(neg_path, pos_path):
    """
    :param neg_path: file path in with the negative patience are
    :param pos_path: file path in with the positive patience are
    Create Dataloaders from the specified files and put it with the needed labels.
    :return: dataloaders for train and test
    """

    # creation of base tensors
    neg_tensor = create_tensor(neg_path)
    neg_labels = torch.cat((torch.zeros(len(neg_tensor),1),torch.ones(len(neg_tensor),1)), dim=1)
    pos_tensor = create_tensor(pos_path)
    pos_labels = torch.cat((torch.ones(len(pos_tensor), 1), torch.zeros(len(pos_tensor),1)),dim=1)

    # to make around the same amount of positive samples and splitting data
    pos_tensor_extended = pos_tensor.repeat(6,1,1)
    pos_labels_extended = pos_labels.repeat(6,1)
    dataset = torch.cat([pos_tensor_extended, neg_tensor],dim=0)
    labels = torch.cat([pos_labels_extended, neg_labels],dim=0)
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1, shuffle=True)

    # creation of the dataloader itself
    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, test_dataloader
