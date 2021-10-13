import torch.nn as nn
import torch 
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import namedtuple
from Enva import Envairoment
from itertools import count


Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

def extract_tensor(expr):
    expr = Experience(*zip(*expr))
    t1 = torch.cat(expr.state)
    t2 = torch.cat(expr.action)
    t3 = torch.cat(expr.next_state)
    t4 = torch.cat(expr.reward)
    return t1, t2, t3, t4



class QDN(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.out = nn.Linear(8, 4)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

class Qvalue:

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states.float()).gather(dim = 1, index = actions.unsqueeze(-1))
    
    
    @staticmethod
    def get_next(target_net, next_states):
        finl_state_loc = next_states.flatten(start_dim = 1).max(dim=1)[0].eq(0)\
            .type(torch.bool)
        non_finl_state_loc = (finl_state_loc == False)
        non_finl_state = next_states[non_finl_state_loc]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size)
        values[non_finl_state_loc] = target_net(non_finl_state.float()).max(dim = 1)[0].detach()
        return values



class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, expr):
        if len(self.memory) < self.capacity:
            self.memory.append(expr)
        else:
            self.memory[self.push_count % self.capacity] = expr
        self.push_count += 1
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide(self, batch_size):
        return len(self.memory) >= batch_size

class EpsilonStratgy:

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end)*math.exp(-current_step*self.decay)


class Agent:

    def __init__(self, strategy, num_action):

        self.current_step = 0
        self.strategy = strategy
        self.num_action = num_action

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            return random.randrange(self.num_action)
        else:
            with torch.no_grad():
                return policy_net(state.float()).argmax(dim = 1).item()


#main   
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 1e-3
num_episodes = 1000
batch_size = 256
gamma = 0.999

em = Envairoment()
strategy = EpsilonStratgy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_action_avilable())
memory = ReplayMemory(memory_size)

policy_net = QDN()
target_net = QDN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr = lr)

episodeDuration = []
import time
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()

    for timestep in count():
        time.sleep(0.1)
        em.rander()
        action = agent.select_action(state, policy_net)
        action = torch.tensor(action).reshape(1,)
        reward = em.take_action(action)
        reward = torch.tensor(reward).reshape(1,).float()
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide(batch_size):
            experiences = memory.sample(batch_size)

            states, actions, next_states, rewards = extract_tensor(experiences)
            
            current_qvalue = Qvalue.get_current(policy_net, states, actions)
            next_qvalue = Qvalue.get_next(target_net, next_states)
            target_qvalue = (next_qvalue*gamma) + rewards

            loss = F.mse_loss(current_qvalue.reshape(-1,1), target_qvalue.reshape(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if em.done:
            episodeDuration.append(timestep)
            break
    
    if episode%target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
