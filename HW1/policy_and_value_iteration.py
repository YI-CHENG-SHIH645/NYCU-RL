# Spring 2022, IOC 5259 Reinforcement Learning
# HW1, partI: Policy Iteration and Value iteration for MDPs

import numpy as np
import gym


def get_rewards_and_transitions_from_env(env):
    # Get state and action space sizes
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize matrices
    R = np.zeros((num_states, num_actions, num_states))
    P = np.zeros((num_states, num_actions, num_states))

    # Get rewards and transition probabilities for all transitions from an OpenAI gym environment
    for s in range(num_states):
        for a in range(num_actions):
            for transition in env.P[s][a]:
                prob, s_, r, done = transition
                R[s, a, s_] = r
                P[s, a, s_] = prob

    return R, P


def value_iteration(env, gamma=0.9, max_iterations=10 ** 6, eps=10 ** -3):
    """        
        Run value iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for value iteration
            eps: float
                for the termination criterion of value iteration 
        ----------
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function V(s)
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve V(s) using the Bellman optimality operator
            4. Derive the optimal policy using V(s)
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])

    # FINISH TODOS HERE #
    V = {s: 0 for s in range(num_spaces)}
    P = env.P
    ith_iter = 0
    while True:
        biggest_change = 0
        if ith_iter == max_iterations:
            break
        ith_iter += 1
        for s in V:
            v_candidates = []
            for a in range(num_actions):
                transitions = P[s][a]
                v_candidate = 0
                for transition in transitions:
                    v_candidate += transition[0] * (transition[2] + gamma * V[transition[1]])
                v_candidates.append(v_candidate)
            biggest_change = max(biggest_change, np.abs(V[s]-max(v_candidates)))
            V[s] = max(v_candidates)
        if biggest_change < eps:
            break
    problem_dist = None
    for s in V:
        v_candidates = []
        for a in range(num_actions):
            v_candidate = 0
            for transition in P[s][a]:
                v_candidate += transition[0] * (transition[2] + gamma * V[transition[1]])
            v_candidates.append(v_candidate)
        if s == 147:
            problem_dist = v_candidates
        policy[s] = np.argmax(v_candidates)
    #####################
    print(ith_iter, problem_dist)
    # Return optimal policy    
    return policy


def policy_iteration(env, gamma=0.9, max_iterations=10 ** 6, eps=10 ** -3):
    """ 
        Run policy iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for the policy evalaution in policy iteration
            eps: float
                for the termination criterion of policy evaluation 
        ----------  
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize with a random policy and initial value function
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve the policy
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])

    # FINISH TODOS HERE #
    V = {s: 0 for s in range(num_spaces)}
    P = env.P
    ith_iter = 0
    problem_dist = None
    while True:
        if ith_iter == max_iterations:
            break
        ith_iter += 1

        # Policy Evaluation
        while True:
            biggest_change = 0
            for s in V:
                transitions = P[s][policy[s]]
                v = 0
                for transition in transitions:
                    v += transition[0] * (transition[2] + gamma * V[transition[1]])
                biggest_change = max(biggest_change, np.abs(V[s] - v))
                V[s] = v
            if biggest_change < eps:
                break

        # Policy Improvement
        policy_is_converged = True
        for s in V:
            s_a_values = []
            old_a = policy[s]
            for a in range(num_actions):
                v = 0
                for transition in P[s][a]:
                    v += transition[0] * (transition[2] + gamma * V[transition[1]])
                s_a_values.append(v)
            policy[s] = np.argmax(s_a_values)
            if s == 147:
                problem_dist = s_a_values
            if old_a != policy[s]:
                policy_is_converged = False
        if policy_is_converged:
            break
    #############################
    print(ith_iter, problem_dist)
    # Return optimal policy
    return policy


def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))


def run_pi_and_vi(env_name):
    """ 
        Enforce policy iteration and value iteration
    """
    env = gym.make(env_name)
    env.action_space.seed(6)
    print('== {} =='.format(env_name))
    print('# of actions:', env.action_space.n)
    print('# of states:', env.observation_space.n)
    # print(env.desc)

    vi_pi = value_iteration(env)
    pi_pi = policy_iteration(env)

    return pi_pi, vi_pi


if __name__ == '__main__':
    # OpenAI gym environment: Taxi-v2 or Taxi-v3
    pi_policy, vi_policy = run_pi_and_vi('Taxi-v3')

    # For debugging
    action_map = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
    print_policy(pi_policy, action_map, shape=None)
    print_policy(vi_policy, action_map, shape=None)

    # Compare the policies obtained via policy iteration and value iteration
    diff = sum([abs(x - y) for x, y in zip(pi_policy.flatten(), vi_policy.flatten())])
    print('Discrepancy:', diff)
