import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: list=[64,64],
                 out_act: str='Identity',
                 hidden_act: str="ReLU"):

        super(MLP, self).__init__()

        self.out_act = getattr(nn, out_act)()
        self.hidden_act = getattr(nn, hidden_act)()
        self.layers = nn.ModuleList()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims)-1 else False
            layer = nn.Linear(in_dim, out_dim)
            #nn.init.normal_(layer.weight, mean=0., std=0.1)
            #nn.init.constant_(layer.bias, 0.)

            self.layers.append(layer)
            if is_last:
                self.layers.append(self.out_act)
            else:
                self.layers.append(self.hidden_act)


    def forward(self, xs):

        for layer in self.layers:
            xs = layer(xs)
        
        return xs