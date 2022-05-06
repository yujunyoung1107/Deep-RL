import torch
import numpy as np

def ToTensor(x, f=True):

    x = torch.tensor(x).view(1,-1)
    return x.float() if f else x