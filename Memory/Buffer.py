import numpy as np
import torch
import torch.nn as nn

class Buffer:

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def push(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)


    def get_sample(self):
        return self.states, self.actions, self.rewards

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def __len__(self):
        return len(self.states)