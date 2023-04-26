# Spring 2022, IOC 5259 Reinforcement Learning
# HW2: DDPG

import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable


class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OUNoise:

    def __init__(
            self,
            action_dimension,
            scale=0.1,
            mu=0,
            theta=0.15,
            sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]  # 2

        # TODO (5~10 lines)
        # Construct your own actor network
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 300),
            nn.ReLU(),
            nn.Linear(300, num_outputs),
            nn.Tanh()
        )

    def forward(self, inputs):
        # TODO (5~10 lines)
        # Define the forward pass your actor network
        return self.actor(inputs) * 2


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]  # 2

        # TODO (5~10 lines)
        # Construct your own critic network
        self.critic_head = nn.Sequential(
            nn.Linear(num_inputs + num_outputs, hidden_size),
            nn.ReLU()
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, inputs, actions):
        # TODO (5~10 lines)
        # Define the forward pass your critic network
        x = self.critic_head(torch.cat([inputs, actions], dim=1))
        return self.critic(x)


class DDPG(object):
    def __init__(
            self,
            num_inputs,  # 8
            action_space,  # Box([-1. -1.], [1. 1.], (2,), float32)
            gamma=0.995,
            tau=0.0005,
            hidden_size=128,
            lr_a=1e-3,
            lr_c=1e-3):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor.apply(init_weights)
        self.actor_target = Actor(
            hidden_size,
            self.num_inputs,
            self.action_space)
        # self.actor_perturbed = Actor(
        #     hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic.apply(init_weights)
        self.critic_target = Critic(
            hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state)))
        mu = mu.data

        # TODO (3~5 lines)
        # Add noise to your action for exploration
        # Clipping might be needed
        a = mu if action_noise is None else mu + action_noise
        a = torch.clip(a, self.action_space.low[0], self.action_space.high[0])
        return a

    def update_parameters(self, batch):
        state_batch = Variable(batch.state)
        action_batch = Variable(batch.action)
        reward_batch = Variable(batch.reward)
        mask_batch = Variable(batch.mask)
        next_state_batch = Variable(batch.next_state)

        # TODO (10~20 lines)
        # Calculate policy loss and value loss
        # Update the actor and the critic
        q_val = self.critic(state_batch, action_batch)
        with torch.no_grad():
            a_next = self.actor_target(next_state_batch)
            q_next = self.critic_target(next_state_batch, a_next)
            q_target = reward_batch + self.gamma * q_next * (1 - mask_batch)
        self.critic.zero_grad()
        criterion = nn.MSELoss()
        critic_loss = criterion(q_val, q_target)
        critic_loss.backward()
        self.critic_optim.step()

        action = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, action).mean()
        self.actor.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item()

    def save_model(
            self,
            env_name,
            suffix="",
            actor_path=None,
            critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = "preTrained/ddpg_actor_{}_{}_{}".format(
                env_name, timestamp, suffix)
        if critic_path is None:
            critic_path = "preTrained/ddpg_critic_{}_{}_{}".format(
                env_name, timestamp, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))


def train():
    num_episodes = 300
    gamma = 0.99
    tau = 0.002
    hidden_size = 400
    noise_scale = 0.3
    replay_size = 500000
    batch_size = 64
    updates_per_step = 1
    print_freq = 1
    total_numsteps = 0

    train_ewma_reward = 0

    test_ewma_reward = 0
    # updates = 0

    agent = DDPG(env.observation_space.shape[0],
                 env.action_space,
                 gamma, tau, hidden_size)
    # ounoise = OUNoise(env.action_space.shape[0])
    noise = GaussianNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)

    for i_episode in range(num_episodes):

        # ounoise.scale = noise_scale
        # ounoise.reset()

        train_ep_reward = 0
        t = 0
        state = torch.Tensor([env.reset()])  # (1, 8)

        while True:
            # TODO (15~25 lines)
            # 1. Interact with the env to get new (s,a,r,s') samples
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic
            if total_numsteps < 10000:
                a = env.action_space.sample()
            else:
                a = agent.select_action(state, noise.sample())  # tensor (1, 2)
                a = a.numpy().flatten()
            s_next, reward, done, _ = env.step(a)  # numpy (2, )
            s_next = torch.tensor([s_next], dtype=torch.float32)
            # episode_reward += reward
            memory.push(state, torch.tensor([a]), torch.tensor(
                [[int(done)]]), s_next, torch.tensor([[reward/100]]))
            if total_numsteps >= 10000 and total_numsteps % updates_per_step == 0:
                batch = Transition(*(torch.cat(x).float()
                                   for x in zip(*memory.sample(batch_size))))
                agent.update_parameters(batch)
            total_numsteps += 1
            t += 1
            train_ep_reward += reward
            state = s_next
            if done:
                # exponential weight moving average reward
                train_ewma_reward = 0.05 * train_ep_reward + (1 - 0.05) * train_ewma_reward
                print(
                    f'Step: {total_numsteps}\t'
                    f'Episode: {i_episode}\t'
                    f'Train R: {train_ep_reward:.2f}\t'
                    f'Train ER: {train_ewma_reward:.2f}\t', end=''
                )
                break

        # testing
        t = 0
        if i_episode % print_freq == 0:
            state = torch.Tensor([env.reset()])
            test_ep_reward = 0
            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action.numpy().flatten())
                env.render()
                test_ep_reward += reward
                next_state = torch.Tensor([next_state])
                state = next_state
                t += 1
                if done:
                    break
            # update EWMA reward and log the results
            test_ewma_reward = 0.05 * test_ep_reward + (1 - 0.05) * test_ewma_reward
            print(
                f'Test R: {test_ep_reward:.2f}\t'
                f'Test ER: {test_ewma_reward:.2f}\t'
            )
    agent.save_model(env.unwrapped.spec.id, '.pth')


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10
    # env = gym.make('LunarLanderContinuous-v2')
    env = gym.make('Pendulum-v0')
    random.seed(random_seed)
    env.seed(random_seed)
    env.action_space.np_random.seed(random_seed)
    torch.manual_seed(random_seed)
    train()
