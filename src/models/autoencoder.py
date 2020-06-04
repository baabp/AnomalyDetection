from .algorithm_utils import PyTorchUtils
import numpy as np
import torch.nn as nn
from skorch import NeuralNetRegressor


class Encoder(nn.Module, PyTorchUtils):
    def __init__(self, n_features=5, hidden_size=10, seed=None, gpu=None):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(self.n_features))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [self.n_features]])
        enc_setup = dec_setup[::-1]

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encode = nn.Sequential(*layers)
        self.to_device(self._encode)

    def forward(self, X):
        encoded = self._encode(X)
        return encoded


class Decoder(nn.Module, PyTorchUtils):
    def __init__(self, n_features=5, hidden_size=10, seed=None, gpu=None):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(self.n_features))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [self.n_features]])

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self._decode = nn.Sequential(*layers)
        self.to_device(self._decode)

    def forward(self, X):
        decoded = self._decode(X)
        return decoded


class AutoEncoder(nn.Module):
    def __init__(self, n_features=5, hidden_size=10, seed=None, gpu=None):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seed = seed
        self.gpu = gpu

        self.encoder = Encoder(n_features=self.n_features, hidden_size=self.hidden_size, seed=self.seed,
                               gpu=self.gpu)
        self.decoder = Decoder(n_features=self.n_features, hidden_size=self.hidden_size, seed=self.seed,
                               gpu=self.gpu)

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded, encoded



class AutoEncoderModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, sequence_length: int, hidden_size: int, seed: int, gpu: int):
        # Each point is a flattened window and thus has as many features as sequence_length * features
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        input_length = n_features * sequence_length

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encoder = nn.Sequential(*layers)
        self.to_device(self._encoder)

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self._decoder = nn.Sequential(*layers)
        self.to_device(self._decoder)

    def forward(self, ts_batch):
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self._encoder(flattened_sequence.float())
        dec = self._decoder(enc)
        return dec, enc
