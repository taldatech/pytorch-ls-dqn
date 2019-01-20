"""
This file implements utility functions.
Most of the function are taken from the PyTorch Agent Net (PTAN) by Shmuma.
PTAN: https://github.com/Shmuma/ptan
"""

# imports

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import os


def test_agent(env, agent, num_rollouts=20):
    """
    This function runs `num_rollouts` using the current agent's policy.
    :param env: environment to test the agent in (gym.Env)
    :param agent: Agent to predict actions (DQNAgent)
    :param num_rollouts: number of episodes to play (int)
    :return: average_reward: average reward from all the rollouts
    """
    total_reward = 0.0
    for i in range(num_rollouts):
        state = env.reset()
        game_reward = 0.0
        while True:
            action, _ = agent([state])
            state, reward, done, _ = env.step(action)
            total_reward += reward
            game_reward += reward
            if done:
                # print("dqn-game ", i, "reward: ", game_reward)
                break
    return 1.0 * total_reward / num_rollouts


def save_agent_state(net, optimizer, frame, games, epsilon, save_replay=False, replay_buffer=None, name='', path=None):
    """
    This function saves the current state of the DQN (the weights) to a local file.
    :param net: the current DQN (nn.Module)
    :param optimizer: the network's optimizer (torch.optim)
    :param frame: current frame number (int)
    :param games: total number of games seen (int)
    :param epsilon: current exploration value (float)
    :param save_replay: whether or not to save the replay buffer (bool)
    :param replay_buffer: the replay buffer (list)
    :param name: specific name for the checkpoint (str)
    :param path: path to specific location where to save (str)
    """
    if path:
        full_path = path
    else:
        if name:
            filename = "agent_ls_dqn_" + name + ".pth"
        else:
            filename = "agent_ls_dqn.pth"
        dir_name = './agent_ckpt'
        full_path = os.path.join(dir_name, filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if save_replay and replay_buffer is not None:
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'frame_count': frame,
            'games': games,
            'epsilon': epsilon,
            'replay_buffer': replay_buffer
        }, full_path)
    else:
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'frame_count': frame,
            'games': games,
            'epsilon': epsilon
        }, full_path)
    print("Saved Agent checkpoint @ ", full_path)


def load_agent_state(net, optimizer, selector, path=None, copy_to_target_network=False, load_optimizer=True,
                     target_net=None, buffer=None, load_buffer=False, env_name='pong'):
    """
    This function loads a state of the DQN (the weights) from a local file.
    :param net: the current DQN (nn.Module)
    :param optimizer: the network's optimizer (torch.optim)
    :param selector: action selector instance
    :param path: full path to checkpoint file (.pth) (str)
    :param copy_to_target_network: whether or not to copy the weights to target network (bool)
    :param load_optimizer: whether or not to load the optimizer state (bool)
    :param load_buffer: whether or not to load the replay buffer (bool)
    :param buffer: the replay buffer
    :param target_net: the target DQN
    :param env_name: environment name (str)
    """
    if path is None:
        if env_name == 'pong':
            filename = "pong_agent_ls_dqn.pth"
            dir_name = './pong_agent_ckpt'
            full_path = os.path.join(dir_name, filename)
        else:
            filename = "agent_ls_dqn.pth"
            dir_name = './agent_ckpt'
            full_path = os.path.join(dir_name, filename)
    else:
        full_path = path
    exists = os.path.isfile(full_path)
    if exists:
        if not torch.cuda.is_available():
            checkpoint = torch.load(full_path, map_location='cpu')
        else:
            checkpoint = torch.load(full_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.steps_count = checkpoint['steps_count']
        # self.episodes_seen = checkpoint['episodes_seen']
        selector.epsilon = checkpoint['epsilon']
        # self.num_param_update = checkpoint['num_param_updates']
        print("Checkpoint loaded successfully from ", full_path)
        # # for manual loading a checkpoint
        if copy_to_target_network:
            target_net.sync()
        if load_buffer and buffer is not None:
            buffer.buffer = checkpoint['replay_buffer']
    else:
        print("No checkpoint found...")


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu", double_dqn=False):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    if double_dqn:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)