#!/usr/bin/env python3
import gym
# import argparse
import torch
import torch.optim as optim
import os

from tensorboardX import SummaryWriter

# import dqn_model, common
import ls_dqn_model.dqn_model as dqn_model
import ls_dqn_model.common as common
import ls_dqn_model.ptan as ptan
import numpy as np
import random


def save_agent_state(net, optimizer, frame, games, epsilon, save_replay=False, replay_buffer=None, name=''):
    """
    This function saves the current state of the DQN (the weights) to a local file.
    :param
    """
    if model_name:
        filename = "pong_agent_ls_dqn_" + name + ".pth"
    else:
        filename = "pong_agent_ls_dqn.pth"
    dir_name = './pong_agent_ckpt'
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
    print("Saved Pong Agent checkpoint @ ", full_path)


def load_agent_state(net, optimizer, selector,  path=None, copy_to_target_network=False, load_optimizer=True):
    """
    This function loads an agent checkpoint.
    Parameters:
        path: path to a checkpoint, e.g `/path/to/dir/ckpt.pth` (str)
        copy_to_target_network: whether or not to copy the loaded training
            DQN parameters to the target DQN, for manual loading (bool)
        load_optimizer: whether or not to restore the optimizer state
    """
    # TODO: add frame_idx, maybe save replay buffer
    if path is None:
        filename = "pong_agent_ls_dqn.pth"
        dir_name = './pong_agent_ckpt'
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
        # if copy_to_target_network:
        #     self.Q_target.load_state_dict(self.Q_train.state_dict())
    else:
        print("No checkpoint found...")


def calc_fqi_matrices(net, tgt_net, batch, gamma, n_srl, m_batch_size=512, device='cpu', use_dueling=False,
                      use_boosting=False):
    """
    This function calculates A and b tensors for the FQI solution.
    :param batch: batch of samples to extract features from (list)
    :param net: network to extract features from (nn.Module)
    :param tgt_net: target netwrok from which Q values of next states are calculated (nn.Module)
    :param gamma: discount factor (float)
    :param n_srl: number of samples to include in the FQI solution
    :param m_batch_size: number of samples to calculate simultaneously (int)
    :param device: on which device to perform the calculation (cpu/gpu)
    :param use_dueling: whether or not to use Dueling DQN architecture
    :param use_boosting: whether or not to use Boosted FQI
    :return: A, A_bias, b, b_bias parameters for calculating the LS (np.arrays)
    """
    num_batches = n_srl // m_batch_size
    if use_dueling:
        dim = net.fc1_adv.out_features
        num_actions = net.fc2_adv.out_features
    else:
        dim = net.fc1.out_features
        num_actions = net.fc2.out_features
    A = torch.zeros([dim * num_actions, dim * num_actions], dtype=torch.float32).to(device)
    A_bias = torch.zeros([1 * num_actions, 1 * num_actions], dtype=torch.float32).to(device)
    b = torch.zeros([dim * num_actions, 1], dtype=torch.float32).to(device)
    b_bias = torch.zeros([1 * num_actions, 1], dtype=torch.float32).to(device)
    for i in range(num_batches):
        idx = i * m_batch_size
        if i == num_batches - 1:
            states, actions, rewards, dones, next_states = common.unpack_batch(batch[idx:])
        else:
            states, actions, rewards, dones, next_states = common.unpack_batch(batch[idx: idx + m_batch_size])
        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

        states_features = net.forward_to_last_hidden(states_v)
        # augmentation
        states_features_aug = torch.zeros([states_features.shape[0], dim * num_actions], dtype=torch.float32).to(device)
        states_features_bias_aug = torch.zeros([states_features.shape[0], 1 * num_actions],
                                               dtype=torch.float32).to(device)
        for j in range(states_features.shape[0]):
            position = actions_v[j] * dim
            states_features_aug[j, position:position + dim] = states_features.detach()[j, :]
            states_features_bias_aug[j, actions_v[j]] = 1
        states_features_mat = torch.mm(torch.t(states_features_aug), states_features_aug)
        states_features_bias_mat = torch.mm(torch.t(states_features_bias_aug), states_features_bias_aug)
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0

        expected_state_action_values = next_state_values.detach() * gamma + rewards_v  # y_i
        if use_boosting:
            state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            # calculate truncated bellman error
            bellman_error = expected_state_action_values.detach() - state_action_values.detach()
            truncated_bellman_error = bellman_error.clamp(-1, 1)  # TODO: is this correct?
            b += torch.mm(torch.t(states_features_aug.detach()), truncated_bellman_error.detach().view(-1, 1))
            b_bias += torch.mm(torch.t(states_features_bias_aug), truncated_bellman_error.detach().view(-1, 1))
        else:
            b += torch.mm(torch.t(states_features_aug.detach()), expected_state_action_values.detach().view(-1, 1))
            b_bias += torch.mm(torch.t(states_features_bias_aug), expected_state_action_values.detach().view(-1, 1))
        A += states_features_mat.detach()
        A_bias += states_features_bias_mat
    A = (1.0 / n_srl) * A
    A_bias = (1.0 / n_srl) * A_bias
    b = (1.0 / n_srl) * b
    b_bias = (1.0 / n_srl) * b_bias
    return A, A_bias, b, b_bias


def calc_fqi_w_srl(a, a_bias, b, b_bias, w, w_b, lam=1.0, device='cpu', use_regularization=True):
    """
    This function calculates the closed-form solution of the DQI algorithm.
    :param a: A matrix built from features (np.array)
    :param a_bias: same, but for bias
    :param b: b vector built from features and rewards (np.array)
    :param b_bias: same, but for bias
    :param w: weights of the last hidden layer in the DQN (np.array)
    :param w_b: bias weights
    :param lam: regularization parameter for the Least-Square (float)
    :param device: on which device to perform the calculation (cpu/gpu)
    :param use_regularization: whether or not to use regularization
    :return: w_srl: retrained weights using FQI closed-form solution (np.array)
    """
    num_actions = w.shape[0]
    dim = w.shape[1]
    w = w.view(-1, 1)
    w_b = w_b.view(-1, 1)
    if not use_regularization:
        lam = 0
    w_srl = torch.mm(torch.inverse(a + lam * torch.eye(num_actions * dim).to(device)), b + lam * w.detach())
    w_b_srl = torch.mm(torch.inverse(a_bias + lam * torch.eye(num_actions * 1).to(device)), b_bias + lam * w_b.detach())
    return w_srl.view(num_actions, dim), w_b_srl.squeeze()


def ls_step(net, tgt_net, batch, gamma, n_srl, lam=1.0, m_batch_size=256, device='cpu', use_dueling=False,
            use_boosting=False, use_regularization=True):
    """
    This function performs the least-squares update on the last hidden layer weights.
    :param batch: batch of samples to extract features from (list)
    :param net: network to extract features from (nn.Module)
    :param tgt_net: target netwrok from which Q values of next states are calculated (nn.Module)
    :param gamma: discount factor (float)
    :param n_srl: number of samples to include in the FQI solution
    :param lam: regularization parameter for the Least-Square (float)
    :param m_batch_size: number of samples to calculate simultaneously (int)
    :param device: on which device to perform the calculation (cpu/gpu)
    :param use_dueling: whether or not to use Dueling DQN architecture
    :param use_regularization: whether or not to use regularization
    :param use_boosting: whether or not to use Boosted FQI
    :return:
    """
    a, a_bias, b, b_bias = calc_fqi_matrices(net, tgt_net, batch, gamma,
                                             n_srl, m_batch_size=m_batch_size, device=device,
                                             use_dueling=use_dueling, use_boosting=use_boosting)
    if use_dueling:
        w_last_dict = net.fc2_adv.state_dict()
    else:
        w_last_dict = net.fc2.state_dict()
    w_srl, w_b_srl = calc_fqi_w_srl(a.detach(), a_bias.detach(), b.detach(), b_bias.detach(),
                                    w_last_dict['weight'], w_last_dict['bias'], lam=lam, device=device,
                                    use_regularization=use_regularization)
    w_last_dict['weight'] = w_srl.detach()
    w_last_dict['bias'] = w_b_srl.detach()
    if use_dueling:
        net.fc2_adv.load_state_dict(w_last_dict)
    else:
        net.fc2.load_state_dict(w_last_dict)
    print("least-squares step done.")


if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']
#    params['epsilon_frames'] = 200000
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
#     args = parser.parse_args()
#     device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    training_random_seed = 10
    save_freq = 50000
    n_drl = 100000  # steps of DRL between SRL
    n_srl = params['replay_size']  # size of batch in SRL step
    use_double_dqn = False
    use_dueling_dqn = False
    use_boosting = False
    use_ls_dqn = True
    use_constant_seed = True  # to compare performance independently of the randomness
    save_for_analysis = False  # save also the replay buffer for later analysis

    lam = 1.0  # regularization parameter
    params['batch_size'] = 64
    if use_ls_dqn:
        print("using ls-dqn with lambda:", str(lam))
        model_name = "-LSDQN-LAM-" + str(lam) + "-" + str(int(1.0 * n_drl / 1000)) + "K"
    else:
        model_name = "-DQN"
    model_name += "-BATCH-" + str(params['batch_size'])
    if use_double_dqn:
        print("using double-dqn")
        model_name += "-DOUBLE"
    if use_dueling_dqn:
        print("using dueling-dqn")
        model_name += "-DUELING"
    if use_boosting:
        print("using boosting")
        model_name += "-BOOSTING"
    if use_constant_seed:
        model_name += "-SEED-" + str(training_random_seed)
        np.random.seed(training_random_seed)
        random.seed(training_random_seed)
        env.seed(training_random_seed)
        torch.manual_seed(training_random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_random_seed)
        print("training using constant seed of ", training_random_seed)

    writer = SummaryWriter(comment="-" + params['run_name'] + model_name)
    if use_dueling_dqn:
        net = dqn_model.DuelingLSDQN(env.observation_space.shape, env.action_space.n).to(device)
    else:
        net = dqn_model.LSDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)

    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])  # TODO: change to RMSprop

    load_agent_state(net, optimizer, selector, load_optimizer=False)

    frame_idx = 0
    drl_updates = 0

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    if save_for_analysis:
                        temp_model_name = model_name + "_" + str(frame_idx)
                        save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards), selector.epsilon,
                                         save_replay=True, replay_buffer=buffer.buffer, name=temp_model_name)
                    else:
                        save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards), selector.epsilon)
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'],
                                          device=device, double_dqn=use_double_dqn)
            loss_v.backward()
            optimizer.step()
            drl_updates += 1

            # LS-UPDATE STEP
            if use_ls_dqn and (drl_updates % n_drl == 0) and (len(buffer) >= n_srl):
            # if (len(buffer) > 1):
                print("performing ls step...")
                batch = buffer.sample(n_srl)
                ls_step(net, tgt_net.target_model, batch, params['gamma'], len(batch), lam=lam,
                        m_batch_size=256, device=device, use_dueling=use_dueling_dqn, use_boosting=use_boosting)
                # a, a_bias, b, b_bias = calc_fqi_matrices(net, tgt_net.target_model, batch, params['gamma'],
                #                                          len(batch), m_batch_size=256, device=device)
                # w_last_dict = net.fc2.state_dict()
                # w_srl, w_b_srl = calc_fqi_w_srl(a.detach(), a_bias.detach(), b.detach(), b_bias.detach(),
                #                                 w_last_dict['weight'], w_last_dict['bias'], lam=1.0, device=device)
                # w_last_dict['weight'] = w_srl.detach()
                # w_last_dict['bias'] = w_b_srl.detach()
                # net.fc2.load_state_dict(w_last_dict)

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
            if frame_idx % save_freq == 0:
                if save_for_analysis and frame_idx % n_drl == 0:
                    temp_model_name = model_name + "_" + str(frame_idx)
                    save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards), selector.epsilon,
                                     save_replay=True, replay_buffer=buffer.buffer, name=temp_model_name)
                else:
                    save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards), selector.epsilon)
