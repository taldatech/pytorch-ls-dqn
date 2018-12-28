"""
This file implements the SRL algorithms.
Author: Tal Daniel
"""

#imports
import torch
import ls_dqn_model.utils.utils as utils
import copy


def calc_fqi_matrices(net, tgt_net, batch, gamma, n_srl, m_batch_size=512, device='cpu', use_dueling=False,
                      use_boosting=False, use_double_dqn=False):
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
    :param use_double_dqn: whether or not to use Double DQN
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
            states, actions, rewards, dones, next_states = utils.unpack_batch(batch[idx:])
        else:
            states, actions, rewards, dones, next_states = utils.unpack_batch(batch[idx: idx + m_batch_size])
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
        if use_double_dqn:
            next_state_actions = net(next_states_v).max(1)[1]
            next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
        else:
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
            use_boosting=False, use_regularization=True, use_double_dqn=False):
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
    :param use_double_dqn: whether or not to use Double DQN
    :return:
    """
    a, a_bias, b, b_bias = calc_fqi_matrices(net, tgt_net, batch, gamma,
                                             n_srl, m_batch_size=m_batch_size, device=device,
                                             use_dueling=use_dueling, use_boosting=use_boosting,
                                             use_double_dqn=use_double_dqn)
    if use_dueling:
        w_last_dict = copy.deepcopy(net.fc2_adv.state_dict())
    else:
        w_last_dict = copy.deepcopy(net.fc2.state_dict())
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

