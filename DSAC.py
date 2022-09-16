import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gym
import time
from Network.MLP import MLP
from Memory.Memory import ReplayBuffer
from Utils.utils import *


class Qnet(nn.Module):
    
    def __init__(self, s_dim, a_dim , lr):
        super(Qnet, self).__init__()

        self.network = MLP(input_dim=s_dim, output_dim=a_dim, num_neurons=[256,256])
        self.opt = torch.optim.Adam(params=self.network.parameters(), lr=lr)
        #self.apply(init_weights)
    
    def forward(self, state):
        return self.network(state)


class Actor(nn.Module):

    def __init__(self, s_dim, a_dim, lr):
        super(Actor, self).__init__()

        self.network = MLP(input_dim=s_dim, output_dim=a_dim, num_neurons=[256,256])
        self.opt = torch.optim.Adam(params=self.network.parameters(), lr=lr)
        self.softmax = nn.Softmax(dim=-1)
        #self.apply(init_weights)

    def forward(self, state):

        action_prob = self.softmax(self.network(state))
        dist = Categorical(action_prob)
        action = dist.sample()

        z = (action_prob == 0.0).float() * 1e-8
        log_prob = torch.log(action_prob + z)
        

        return action, action_prob, log_prob


class DSAC(nn.Module):

    def __init__(self, s_dim, a_dim, device):
        super(DSAC, self).__init__()

        self.actor = Actor(s_dim, a_dim, 5e-4).to(device)
        self.Q1 = Qnet(s_dim, a_dim, 5e-4).to(device)
        self.Q2 = Qnet(s_dim, a_dim, 5e-4).to(device)
        self.Q1_target = Qnet(s_dim, a_dim, 5e-4).to(device)
        self.Q2_target = Qnet(s_dim, a_dim, 5e-4).to(device)
        self.log_alpha = torch.zeros(1).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=5e-4) 

        self.Q1_target.load_state_dict(self.Q1.state_dict())    
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        self.memory = ReplayBuffer(1000000, device)
        self.gamma = 0.99
        self.criterion = nn.MSELoss()
        self.target_entropy = -a_dim
        #self.target_entropy = -np.log(1./a_dim) * 0.98
        self.tau = 1e-2
        self.device = device

    def get_action(self, state):

        state = ToTensor(state).to(self.device)
        action, _, _ = self.actor(state)
        return action.view(-1).detach().cpu().numpy() 

    def get_sample(self, s, a, r, ns, done):
        
        s = ToTensor(s)
        a = ToTensor(a, False)
        r = ToTensor(r)
        ns = ToTensor(ns)
        done = ToTensor(done)

        self.memory.push(s, a, r, ns, done)

    def update(self):

        s, a, r, ns, done = self.memory.get_sample(256)

        _, action_prob, log_prob = self.actor(s)    
        Q_value = torch.min(self.Q1(s), self.Q2(s)) 
        actor_loss = (action_prob * (self.log_alpha.exp() * log_prob - Q_value)).sum(1).mean()
        entropies = torch.sum(action_prob*log_prob, dim=1)

        self.actor.opt.zero_grad()
        actor_loss.backward()
        self.actor.opt.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha * (self.target_entropy + entropies).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        with torch.no_grad():
            _, action_prob, log_prob = self.actor(ns)
            Q1 = self.Q1_target(ns)
            Q2 = self.Q2_target(ns)
            Q_value = (action_prob * (torch.min(Q1,Q2) - self.log_alpha.exp()*log_prob)).sum(dim=1).unsqueeze(-1)
            Q_target = r + self.gamma * (1-done) * Q_value
        
        new_Q1 = self.Q1(s).gather(1,a)
        new_Q2 = self.Q2(s).gather(1,a)
        Q1_loss = 0.5 * self.criterion(new_Q1, Q_target)
        Q2_loss = 0.5 * self.criterion(new_Q2, Q_target)
        Q_loss = Q1_loss + Q2_loss

        self.Q1.opt.zero_grad()
        self.Q2.opt.zero_grad()
        Q_loss.backward()

        self.Q1.opt.step()
        self.Q2.opt.step()

        #grad?
        soft_update(self.Q1, self.Q1_target, self.tau)
        soft_update(self.Q2, self.Q2_target, self.tau)

    def get_memory(self, env, num_samples):
        s = env.reset()[0]
        for _ in range(num_samples):
            a = env.action_space.sample()
            ns, r, done, trunc, _ = env.step(a)
            new_done = False if (done==False and trunc==False) else True

            self.get_sample(s,a,r,ns,new_done)
            s = ns
            if new_done:
                s = env.reset()[0]
        
    
    def run_episode(self, env, num_episode):

        reward_sum = []

        self.get_memory(env, 10000)

        for ep in range(num_episode):
            s = env.reset()[0]
            cum_r = 0
            while True:
                a = self.get_action(s)
                ns, r, done, trunc, _ = env.step(a.item())
                new_done = False if (done==False and trunc==False) else True

                self.get_sample(s,a,r,ns,new_done)

                self.update()

                s = ns
                cum_r += r 
                if new_done:
                    reward_sum.append(cum_r)
                    print('reward : {}'.format(cum_r))
                    break

            if ep % 10 == 0 and ep != 0:
                print('ep : {} | reward_avg : {}'.format(ep, np.mean(reward_sum)))
                reward_sum = []




def main():

    env = gym.make('LunarLander-v2')
    s_dim = env.observation_space.shape[0]  
    a_dim = env.action_space.n

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    agent = DSAC(s_dim, a_dim, device)
    start_time = time.time()
    agent.run_episode(env, 100)
    print(time.time() - start_time)


if __name__ == '__main__':
    main()