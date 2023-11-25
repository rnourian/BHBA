import numpy as np
import pandas as pd
from numpy import log2
from scipy.io import arff
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict


class Database(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_data_split(x, y, t_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size, random_state=40,
                                                        shuffle=True)
    return x_train, x_test, y_train, y_test


def separate_label(data):
    y = data[data.columns[-1]]
    x = data.drop(data.columns[-1], axis=1)
    return x, y


def load_mendeley_data():
    data = pd.read_csv('Data.csv', header=None)
    x, y = separate_label(data)
    y = y.replace([5], 0)
    classes = ('UCEC', 'BRCA', 'KIRC', 'LUAD', 'LUSC')
    return classes, x, y

def load_UCI_data():
    data = pd.read_csv('UCI_Data.csv', low_memory=False, header=None)
    df_label = pd.read_csv('UCI_Label.csv')
    y = pd.read_csv('labels_deep.csv', header=None)
    data = data.drop(0, axis=1)
    data = data.drop(0, axis=0)
    y = y[y.columns[-1]]

    label = LabelEncoder()
    x = data
    classes = list(OrderedDict.fromkeys(df_label['Class']))
    return classes, x, y

def load_GSE_data():
    data = pd.read_csv('GSE211692_dataset.csv', low_memory=False, header=None)
    y = pd.read_csv('labels.csv')
    data = data.drop(0, axis=1)
    data = data.drop(0, axis=0)
    y = y[y.columns[-1]]

    label = LabelEncoder()
    x = data
    classes = set(y)
    return classes, x, y

def scale_reduction(elem):
    return log2(elem.astype('double') + 1)


def get_data_types():
    data = pd.read_csv('mendeley/Type.csv', header=None)
    classes = np.unique(data.values)
    return classes



