from math import sqrt
import numpy as np
import torch

def convert_to_tensor2(x_train, y_train, x_test, y_test):
    x_train_torch = torch.FloatTensor(x_train.to_numpy())
    y_train_torch = torch.LongTensor(y_train.to_numpy())
    x_test_torch = torch.FloatTensor(x_test.to_numpy())
    y_test_torch = torch.LongTensor(y_test.to_numpy())
    return x_train_torch, y_train_torch, x_test_torch, y_test_torch


def convert_to_tensor(x, y):
    x = torch.FloatTensor(x.to_numpy())
    y = torch.LongTensor(y.to_numpy())
    return x, y

def scale_value_data(x):
    mx_val = torch.max(x)
    return (x*255/mx_val)

def sequence_to_image(x):
    # x = torch.reshape(X_train[:, 0:961], [32, 31, 31])
    nChannels = x.shape[0]
    return torch.reshape(x[:, 0:961], [nChannels, 31, 31])