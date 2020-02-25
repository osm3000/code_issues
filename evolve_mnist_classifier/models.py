import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

class Simple_MLP(nn.Module):
    def __init__(self, **hyperparam):
        super(Simple_MLP, self).__init__()
        self.input_size = hyperparam.get("input_size", 3) # technically, this is the size of the slice
        self.hidden_size = hyperparam.get("hidden_size", 50)
        self.nb_layers = hyperparam.get("nb_layers", 1)
        self.output_size = hyperparam.get("output_size", 10)

        self.model = nn.ModuleList()
        if self.nb_layers > 0:
            for layer_index in range(self.nb_layers):
                if layer_index == 0:
                    self.model.append(nn.Linear(self.input_size, self.hidden_size))
                else:
                    self.model.append(nn.Linear(self.hidden_size, self.hidden_size))
                self.model.append(nn.Tanh())

            self.model.append(nn.Linear(self.hidden_size, self.output_size))
        else:
            self.model.append(nn.Linear(self.input_size, self.output_size))
        self.model.append(nn.LogSoftmax(dim=1))

    def forward(self, data_input):
        data_input = data_input.squeeze(dim=1)
        out = None
        for layer_index, layer in enumerate(self.model):
            if layer_index == 0:
                out = layer(data_input)
            else:
                out = layer(out)
        return out

