import numpy as np
import torch
import torch.nn as nn
from random import sample

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

class ReplayBuffer:

    def __init__(self, max_size):

        self.max_size = max_size
        self.states = [None]*max_size
        self.actions = [None]*max_size
        self.rewards = [None]*max_size
        self.next_states = [None]*max_size
        self.dones = [None]*max_size
        self.size = 0

    def push(self, s, a, r, ns, done):
        
        index = self.size % self.max_size
        
        self.states[index] = s
        self.actions[index] = a
        self.rewards[index] = r
        self.next_states[index] = ns
        self.dones[index] = done
        self.size = min((self.size + 1), self.max_size)

    def get_sample(self, batch_size):

        indices = sample(range(0, self.size), batch_size)

        s = [self.states[indice] for indice in indices]
        a = [self.actions[indice] for indice in indices]
        r = [self.rewards[indice] for indice in indices]
        ns = [self.next_states[indice] for indice in indices]
        done = [self.dones[indice] for indice in indices]

        return s, a, r, ns, done

    def __len__(self):

        return self.size

