# Spring 2022, IOC 5259 Reinforcement Learning
# HW1-partII: REINFORCE and baseline
import os

import gym
from itertools import count
from collections import namedtuple
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the shared layer(s), the action layer(s), and the value layer(s)
            2. Random weight initialization of each layer
    """

    def __init__(self):
        super(Policy, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128

        # YOUR CODE HERE (5~10 lines) #
        self.l1 = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)
        self.value_layer = nn.Linear(self.hidden_size, 1)

        # END OF YOUR CODE #

        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding
              action probability distribution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        # YOUR CODE HERE (3~5 lines) #
        x = self.l1(state)
        action_prob = F.softmax(self.action_layer(x), dim=1)
        state_value = self.value_layer(x)

        # END OF YOUR CODE #

        return action_prob, state_value

    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        # YOUR CODE HERE (3~5 lines) #
        state = torch.from_numpy(state).float().unsqueeze(0)
        act_probs, state_value = self(state)
        m = Categorical(act_probs)
        action = m.sample()

        # END OF YOUR CODE #

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def calculate_loss(self, gamma=0.99):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """

        # Initialize the lists and variables
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []

        # YOUR CODE HERE (8-15 lines) #
        discounted_rewards = scipy.signal.lfilter([1], [1, float(-gamma)], self.rewards[::-1], axis=0)[::-1]
        discounted_rewards = torch.from_numpy(discounted_rewards.copy()).float()
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)  # normalize discounted rewards

        for (log_prob, state_value), Gt in zip(saved_actions, discounted_rewards):
            policy_losses.append(-log_prob * (Gt - state_value))
            value_losses.append(F.mse_loss(state_value[0][0], Gt))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        # END OF YOUR CODE #

        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


def train(lr=0.01):
    """
        Train the model using SGD (via backpropagation)
        TODO: In each episode,
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    """

    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=3000, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # run infinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0
        # Uncomment the following line to use learning rate scheduler
        scheduler.step()

        # For each episode, only run 9999 steps so that we don't
        # infinite loop while learning

        # YOUR CODE HERE (10-15 lines) #

        for steps in range(9999):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)

            if done:
                optimizer.zero_grad()
                loss = model.calculate_loss()
                loss.backward()
                optimizer.step()
                ep_reward = sum(model.rewards)
                t = steps + 1
                model.clear_memory()
                break

        # END OF YOUR CODE #

        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        # check if we have "solved" the LunarLander-v2 problem
        if ewma_reward > env.spec.reward_threshold:
            os.makedirs('./preTrained', exist_ok=True)
            torch.save(model.state_dict(), './preTrained/LunarLander-v2_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """
    model = Policy()

    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))

    render = True
    max_episode_len = 10000

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len + 1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 20
    lr = 0.001
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test(f'LunarLander-v2_{lr}.pth')
