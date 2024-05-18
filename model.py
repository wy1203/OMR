import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


class LeakyReLU(nn.Module):
    def forward(self, x):
        return F.leaky_relu(x, negative_slope=0.2)


class CRNN(nn.Module):
    def __init__(self, params):
        super(CRNN, self).__init__()

        self.rnn_layers = params['rnn_layers']
        self.rnn_units = params['rnn_units']
        self.input_dim = params['input_dim']

        self.vocabulary_size = params['vocabulary_size']
        self.linear = nn.Linear(self.input_dim, 2*self.rnn_units)
        self.rnn = nn.ModuleList()
        for _ in range(self.rnn_layers):
            self.rnn.append(nn.LSTM(2*self.rnn_units, self.rnn_units,
                            bidirectional=True, batch_first=True))
            self.rnn.append(nn.Dropout(0.25))

        self.output = nn.Linear(self.rnn_units * 2, self.vocabulary_size)

    def forward(self, x):
        x = self.linear(x)
        for layer in self.rnn:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            else:
                x = layer(x)
        x = self.output(x)
        return x


def ctc_loss_func(preds, targets, input_lengths, target_lengths):
    preds = preds.log_softmax(1)
    return F.ctc_loss(preds, targets, input_lengths, target_lengths)


def initialize_model_params(input_dim, vocabulary_size):
    path = os.getcwd()
    params = load_config(path + '/config/ctc_model.yaml')
    params.update({
        'input_dim': input_dim,
        'vocabulary_size': vocabulary_size,
        'rnn_units': params['rnn_units'],
        'rnn_layers': params['rnn_layers']
    })
    return params
