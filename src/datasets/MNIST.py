import os
import pickle

import numpy as np
import pandas as pd

from .dataset import Dataset


class MNIST(Dataset):
    """0 is the outlier class. The training set is free of outliers."""

    def __init__(self, seed):
        super().__init__(name="MNIST", file_name='')  # We do not need to load data from a file
        self.seed = seed

    def load(self):
        # 0 is the outlier, all other digits are normal
        OUTLIER_CLASS = 0
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Label outliers with 1 and normal digits with 0
        y_train, y_test = (y_train == OUTLIER_CLASS), (y_test == OUTLIER_CLASS)
        x_train = x_train[~y_train]  # Remove outliers from the training set
        x_train, x_test = x_train / 255, x_test / 255
        x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)
        self._data = tuple(pd.DataFrame(data=data) for data in [x_train, y_train, x_test, y_test])
