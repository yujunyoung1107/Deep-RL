import numpy as np
import torch
import torch.nn as nn

class Buffer:

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def push(self, s, a, r, ns, done):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(ns)
        self.dones.append(done)

    def get_sample(self):
        return self.states, self.actions, self.rewards, self.next_states, self.dones

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def __len__(self):
        return len(self.states)