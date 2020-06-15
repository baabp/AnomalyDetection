import os

import torch
import torch.nn as nn
import torchvision
from torch import optim
import numpy as np

import tensorflow as tf
import pandas as pd

from zope.dottedname.resolve import resolve

from src.datasets.MNIST import MNIST
from src.models.DeepADoTS.autoencoder import AutoEncoder
from src.models.DeepADoTS.dagmm import DAGMM
from src.models.DeepADoTS.rnn_ebm import RecurrentEBM
from src.models.DeepADoTS.lstm_ad import LSTMAD



def prepare_dataset(class_name='MNIST.MNIST', train_num=None, test_num=None):
    class_name_head = "src.datasets"
    class_name = class_name_head + '.' + class_name
    x_train, y_train, x_test, y_test = resolve(class_name)(seed=0).data()
    # Use fewer instances for demonstration purposes
    if train_num is not None:
        x_train, y_train = x_train[:train_num], y_train[:train_num]
    if test_num is not None:
        x_test, y_test = x_test[:test_num], y_test[:test_num]
    return x_train, y_train, x_test, y_test

def exec_autoencoder(x_train, x_test):
    model = AutoEncoder(sequence_length=1, num_epochs=40, hidden_size=10, lr=1e-4, gpu=0)
    model.fit(x_train)
    output = model.predict_val(x_test)
    return model, output

def exec_dagmm(x_train, x_test):
    model = DAGMM(sequence_length=1, num_epochs=40, hidden_size=10, lr=1e-4, gpu=0)
    model.fit(x_train)
    print('end_fit')
    output = model.predict(x_test)
    return output

def exec_rnn(x_train, x_test):
    model = RecurrentEBM(gpu=0)
    model.fit(x_train)
    print('end_fit')
    output = model.predict(x_test)
    return output

    # error = model.predict(x_train_rs)
    # print(roc_auc_score(y_test, error))

def exec_lstm_ad(x_train, x_test):
    model = LSTMAD(gpu=0)
    model.fit(x_train)
    print('end_fit')
    output = model.predict(x_test)
    return output


if __name__ == '__main__':
    print(torch.cuda.is_available())
    x_train, y_train, x_test, y_test = prepare_dataset(train_num=1000, test_num=100)
    # model, output = exec_autoencoder(x_train, x_test)
    # output=exec_dagmm(x_train, x_test)
    # output = exec_rnn(x_train, x_test)
    output = exec_lstm_ad(x_train, x_test)
    print(output)