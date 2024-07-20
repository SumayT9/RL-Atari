import random
import torch
import numpy as np
from collections import namedtuple, deque
import torch.nn as nn
import torch.optim as optim
from dqn_model import DQN
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
  

class Agent():
    def __init__(self, input_size, action_size):
        self.action_size = action_size
        self.discount_factor = 0.99
        self.epsilon_start = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 6000
        self.lr = 1e-4
        self.n_steps = 0
        self.tau = 0.005

        self.memory = ReplayMemory(1000000)
        self.policy_net = DQN(input_size, action_size)
        self.policy_net.to(device)

        self.target_net = DQN(input_size, action_size)
        self.target_net.to(device)
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

    def load_policy_net(self, path):
        self.policy_net = torch.load(path).to(device)
    
    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def get_action(self, state, train=True):
        state = state.to(device)
        eps = self.epsilon_min + (self.epsilon_min + self.epsilon_start) * math.exp(-1. * self.n_steps / self.epsilon_decay)
        self.n_steps += 1
        if np.random.rand() <= eps and train:
            return torch.randint(self.action_size, size=(1, 1), device=device)
        with torch.no_grad():
            policy_out = self.policy_net(state)
            a = policy_out.max(1).indices.view(1, 1)
        return a

    def train_policy_net(self, batch_size=32):        
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)

        q = self.policy_net(states)
        state_action_values = q.gather(1, actions)
        next_state_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_qs = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = next_qs.max(1).values
        
        expected_state_action_values = (next_state_values * self.discount_factor) + rewards
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()