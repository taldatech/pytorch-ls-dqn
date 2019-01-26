"""
This file implements the SRL algorithms.
Author: Tal Daniel
"""

# imports
import torch
import ls_dqn_model.utils.utils as utils
import copy


def ls_step_dueling(net, tgt_net, batch, gamma, n_srl, lam=1.0, m_batch_size=256, device='cpu',
                    use_boosting=False, use_double_dqn=False):
    """
    This function performs the least-squares update on the last hidden layer weights for Dueling architecture.
    :param batch: batch of samples to extract features from (list)
    :param net: network to extract features from (nn.Module)
    :param tgt_net: target netwrok from which Q values of next states are calculated (nn.Module)
    :param gamma: discount factor (float)
    :param n_srl: number of samples to include in the FQI solution
    :param lam: regularization parameter for the Least-Square (float)
    :param m_batch_size: number of samples to calculate simultaneously (int)
    :param device: on which device to perform the calculation (cpu/gpu)
    :param use_boosting: whether or not to use Boosted FQI
    :param use_double_dqn: whether or not to use Double DQN
    :return:
    """
    w_adv_last_dict_before = copy.deepcopy(net.fc2_adv.state_dict())
    w_adv_last_dict = copy.deepcopy(net.fc2_adv.state_dict())
    w_adv_last = w_adv_last_dict['weight'].view(-1, 1)
    w_adv_bias_last = w_adv_last_dict['bias'].view(-1, 1)

    w_val_last_dict_before = copy.deepcopy(net.fc2_val.state_dict())
    w_val_last_dict = copy.deepcopy(net.fc2_val.state_dict())
    w_val_last = w_val_last_dict['weight'].view(-1, 1)
    w_val_bias_last = w_val_last_dict['bias'].view(-1, 1)

    num_batches = n_srl // m_batch_size
    dim = net.fc1_adv.out_features
    num_actions = net.fc2_adv.out_features

    # basic building blocks
    phi_vt_phi_a_full = torch.zeros([dim, dim * num_actions], dtype=torch.float32).to(device)
    phi_vt_phi_a_bias_full = torch.zeros([1, 1 * num_actions], dtype=torch.float32).to(device)
    # phi_at_phi_v_full = torch.t(phi_vt_phi_a_full)
    phi_vt_phi_v_full = torch.zeros([dim, dim], dtype=torch.float32).to(device)
    phi_vt_phi_v_bias_full = torch.zeros([1, 1], dtype=torch.float32).to(device)
    phi_at_phi_a_full = torch.zeros([dim * num_actions, dim * num_actions], dtype=torch.float32).to(device)
    phi_at_phi_a_bias_full = torch.zeros([1 * num_actions, 1 * num_actions], dtype=torch.float32).to(device)
    phi_at_yi_full = torch.zeros([dim * num_actions, 1], dtype=torch.float32).to(device)
    phi_at_yi_bias_full = torch.zeros([1 * num_actions, 1], dtype=torch.float32).to(device)
    phi_vt_yi_full = torch.zeros([dim, 1], dtype=torch.float32).to(device)
    phi_vt_yi_bias_full = torch.zeros([1, 1], dtype=torch.float32).to(device)
    # if using boosting: y_i = rho_i (the bellman residuals)

    ## Method B
    # A_adv_full = torch.zeros([dim * num_actions, dim * num_actions], dtype=torch.float32).to(device)
    # A_adv_bias_full = torch.zeros([num_actions, num_actions], dtype=torch.float32).to(device)
    # b_adv_full = torch.zeros([dim * num_actions, 1], dtype=torch.float32).to(device)
    # b_adv_bias_full = torch.zeros([num_actions, 1], dtype=torch.float32).to(device)
    #
    # A_val_full = torch.zeros([dim, dim], dtype=torch.float32).to(device)
    # A_val_bias_full = torch.zeros([1, 1], dtype=torch.float32).to(device)
    # b_val_full = torch.zeros([dim, 1], dtype=torch.float32).to(device)
    # b_val_bias_full = torch.zeros([1, 1], dtype=torch.float32).to(device)

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

        states_features_adv, states_features_val = net.forward_to_last_hidden(states_v)
        # augmentation
        states_features_adv_aug = torch.zeros([states_features_adv.shape[0], dim * num_actions],
                                              dtype=torch.float32).to(device)
        states_features_adv_bias_aug = torch.zeros([states_features_adv.shape[0], 1 * num_actions],
                                                   dtype=torch.float32).to(device)
        states_features_val_bias = torch.ones([states_features_val.shape[0], 1 * 1],
                                              dtype=torch.float32).to(device)
        for j in range(states_features_adv.shape[0]):
            position = actions_v[j] * dim
            states_features_adv_aug[j, position:position + dim] = states_features_adv.detach()[j, :]
            states_features_adv_bias_aug[j, actions_v[j]] = 1
        states_features_adv_aug = states_features_adv_aug - states_features_adv_aug.mean(dim=1).unsqueeze(-1)
        states_features_adv_bias_aug = states_features_adv_bias_aug - states_features_adv_bias_aug.mean(
            dim=1).unsqueeze(-1)
        states_features_adv_mat = torch.mm(torch.t(states_features_adv_aug), states_features_adv_aug)
        states_features_val_mat = torch.mm(torch.t(states_features_val.detach()), states_features_val.detach())
        states_features_adv_bias_mat = torch.mm(torch.t(states_features_adv_bias_aug), states_features_adv_bias_aug)
        states_features_val_bias_mat = torch.mm(torch.t(states_features_val_bias), states_features_val_bias)
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
            truncated_bellman_error = bellman_error.clamp(-1, 1)  # rho_i

        # # The following is the most accurate approximation, but takes too long! (Method B)
        # phi_vt_phi_a_tmp = torch.mm(torch.t(states_features_val.detach()), states_features_adv_aug)
        # phi_vt_phi_a_bias_tmp = torch.mm(torch.t(states_features_val_bias.detach()), states_features_adv_bias_aug)
        # phi_at_phi_a_tmp = states_features_adv_mat
        # phi_at_phi_a_bias_tmp = states_features_adv_bias_mat
        # phi_vt_phi_v_tmp = states_features_val_mat
        # phi_vt_phi_v_bias_tmp = states_features_val_bias_mat
        # phi_at_phi_v_tmp = torch.t(phi_vt_phi_a_tmp)
        # phi_at_phi_v_bias_tmp = torch.t(phi_vt_phi_a_bias_tmp)
        # if use_boosting:
        #     phi_at_yi_tmp = torch.mm(torch.t(states_features_adv_aug),
        #                              truncated_bellman_error.detach().view(-1, 1))
        #     phi_at_yi_bias_tmp = torch.mm(torch.t(states_features_adv_bias_aug),
        #                                   truncated_bellman_error.detach().view(-1, 1))
        #     phi_vt_yi_tmp = torch.mm(torch.t(states_features_val.detach()),
        #                              truncated_bellman_error.detach().view(-1, 1))
        #     phi_vt_yi_bias_tmp = torch.mm(torch.t(states_features_val_bias.detach()),
        #                                   truncated_bellman_error.detach().view(-1, 1))
        # else:
        #     phi_at_yi_tmp = torch.mm(torch.t(states_features_adv_aug),
        #                              expected_state_action_values.detach().view(-1, 1))
        #     phi_at_yi_bias_tmp = torch.mm(torch.t(states_features_adv_bias_aug),
        #                                   expected_state_action_values.detach().view(-1, 1))
        #     phi_vt_yi_tmp = torch.mm(torch.t(states_features_val.detach()),
        #                              expected_state_action_values.detach().view(-1, 1))
        #     phi_vt_yi_bias_tmp = torch.mm(torch.t(states_features_val_bias.detach()),
        #                                   expected_state_action_values.detach().view(-1, 1))
        # # calc A_v, A_a, b_A, b_B
        # A_adv_full += torch.mm(phi_at_phi_a_tmp, torch.eye(dim * num_actions).to(device) - torch.mm(
        #     torch.mm(phi_at_phi_v_tmp, torch.inverse(
        #         phi_vt_phi_v_tmp + lam * torch.eye(phi_vt_phi_v_tmp.shape[0]).to(device))),
        #     phi_vt_phi_a_tmp))
        #
        # A_adv_bias_full += torch.mm(phi_at_phi_a_bias_tmp, torch.eye(1 * num_actions).to(device) - torch.mm(
        #     torch.mm(phi_at_phi_v_bias_tmp, torch.inverse(
        #         phi_vt_phi_v_bias_tmp + lam * torch.eye(phi_vt_phi_v_bias_tmp.shape[0]).to(device))),
        #     phi_vt_phi_a_bias_tmp))
        #
        # b_adv_full += phi_at_yi_tmp - torch.mm(torch.mm(phi_at_phi_v_tmp, torch.inverse(
        #     phi_vt_phi_v_tmp + lam * torch.eye(phi_vt_phi_v_tmp.shape[0]).to(device))),
        #                                      phi_vt_yi_tmp)
        # b_adv_bias_full += phi_at_yi_bias_tmp - torch.mm(torch.mm(phi_at_phi_v_bias_tmp, torch.inverse(
        #     phi_vt_phi_v_bias_tmp + lam * torch.eye(phi_vt_phi_v_bias_tmp.shape[0]).to(device))),
        #                                            phi_vt_yi_bias_tmp)
        #
        # A_val_full += torch.mm(phi_vt_phi_v_tmp, torch.eye(dim).to(device) - torch.mm(
        #     torch.mm(phi_vt_phi_a_tmp, torch.inverse(
        #         phi_at_phi_a_tmp + lam * torch.eye(phi_at_phi_a_tmp.shape[0]).to(device))),
        #     phi_at_phi_v_tmp))
        #
        # A_val_bias_full += torch.mm(phi_vt_phi_v_bias_tmp, torch.eye(1 * 1).to(device) - torch.mm(
        #     torch.mm(phi_vt_phi_a_bias_tmp, torch.inverse(
        #         phi_at_phi_a_bias_tmp + lam * torch.eye(phi_at_phi_a_bias_tmp.shape[0]).to(device))),
        #     phi_at_phi_v_bias_tmp))
        #
        # b_val_full += phi_vt_yi_tmp - torch.mm(torch.mm(phi_vt_phi_a_tmp, torch.inverse(
        #     phi_at_phi_a_tmp + lam * torch.eye(phi_at_phi_a_tmp.shape[0]).to(device))),
        #                                      phi_at_yi_tmp)
        # b_val_bias_full += phi_vt_yi_bias_tmp - torch.mm(torch.mm(phi_vt_phi_a_bias_tmp, torch.inverse(
        #     phi_at_phi_a_bias_tmp + lam * torch.eye(phi_at_phi_a_bias_tmp.shape[0]).to(device))),
        #                                            phi_at_yi_bias_tmp)

        phi_vt_phi_a_full += torch.mm(torch.t(states_features_val.detach()), states_features_adv_aug)
        phi_vt_phi_a_bias_full += torch.mm(torch.t(states_features_val_bias.detach()), states_features_adv_bias_aug)
        phi_at_phi_a_full += states_features_adv_mat
        phi_at_phi_a_bias_full += states_features_adv_bias_mat
        phi_vt_phi_v_full += states_features_val_mat
        phi_vt_phi_v_bias_full += states_features_val_bias_mat
        if use_boosting:
            phi_at_yi_full += torch.mm(torch.t(states_features_adv_aug),
                                       truncated_bellman_error.detach().view(-1, 1))
            phi_at_yi_bias_full += torch.mm(torch.t(states_features_adv_bias_aug),
                                            truncated_bellman_error.detach().view(-1, 1))
            phi_vt_yi_full += torch.mm(torch.t(states_features_val.detach()),
                                       truncated_bellman_error.detach().view(-1, 1))
            phi_vt_yi_bias_full += torch.mm(torch.t(states_features_val_bias.detach()),
                                            truncated_bellman_error.detach().view(-1, 1))
        else:
            phi_at_yi_full += torch.mm(torch.t(states_features_adv_aug),
                                       expected_state_action_values.detach().view(-1, 1))
            phi_at_yi_bias_full += torch.mm(torch.t(states_features_adv_bias_aug),
                                            expected_state_action_values.detach().view(-1, 1))
            phi_vt_yi_full += torch.mm(torch.t(states_features_val.detach()),
                                       expected_state_action_values.detach().view(-1, 1))
            phi_vt_yi_bias_full += torch.mm(torch.t(states_features_val_bias.detach()),
                                            expected_state_action_values.detach().view(-1, 1))

    # Method A / C
    phi_vt_phi_a_full = (1.0 / n_srl) * phi_vt_phi_a_full
    phi_vt_phi_a_bias_full = (1.0 / n_srl) * phi_vt_phi_a_bias_full
    phi_at_phi_v_full = torch.t(phi_vt_phi_a_full)
    phi_at_phi_v_bias_full = torch.t(phi_vt_phi_a_bias_full)
    phi_vt_phi_v_full = (1.0 / n_srl) * phi_vt_phi_v_full
    phi_vt_phi_v_bias_full = (1.0 / n_srl) * phi_vt_phi_v_bias_full
    phi_at_phi_a_full = (1.0 / n_srl) * phi_at_phi_a_full
    phi_at_phi_a_bias_full = (1.0 / n_srl) * phi_at_phi_a_bias_full
    phi_at_yi_full = (1.0 / n_srl) * phi_at_yi_full
    phi_at_yi_bias_full = (1.0 / n_srl) * phi_at_yi_bias_full
    phi_vt_yi_full = (1.0 / n_srl) * phi_vt_yi_full
    phi_vt_yi_bias_full = (1.0 / n_srl) * phi_vt_yi_bias_full

    A_adv = phi_at_phi_a_full - torch.mm(torch.mm(phi_at_phi_v_full, torch.inverse(
        phi_vt_phi_v_full + lam * torch.eye(phi_vt_phi_v_full.shape[0]).to(device))),
                                         phi_vt_phi_a_full) + lam * torch.eye(phi_at_phi_a_full.shape[0]).to(device)
    A_adv_bias = phi_at_phi_a_bias_full - torch.mm(torch.mm(phi_at_phi_v_bias_full, torch.inverse(
        phi_vt_phi_v_bias_full + lam * torch.eye(phi_vt_phi_v_bias_full.shape[0]).to(device))),
                                                   phi_vt_phi_a_bias_full) + lam * torch.eye(
        phi_at_phi_a_bias_full.shape[0]).to(device)

    b_adv = phi_at_yi_full - torch.mm(torch.mm(phi_at_phi_v_full, torch.inverse(
        phi_vt_phi_v_full + lam * torch.eye(phi_vt_phi_v_full.shape[0]).to(device))),
                                      phi_vt_yi_full) + lam * w_adv_last
    b_adv_bias = phi_at_yi_bias_full - torch.mm(torch.mm(phi_at_phi_v_bias_full, torch.inverse(
        phi_vt_phi_v_bias_full + lam * torch.eye(phi_vt_phi_v_bias_full.shape[0]).to(device))),
                                                phi_vt_yi_bias_full) + lam * w_adv_bias_last

    A_val = phi_vt_phi_v_full - torch.mm(torch.mm(phi_vt_phi_a_full, torch.inverse(
        phi_at_phi_a_full + lam * torch.eye(phi_at_phi_a_full.shape[0]).to(device))),
                                         phi_at_phi_v_full) + lam * torch.eye(phi_vt_phi_v_full.shape[0]).to(device)
    A_val_bias = phi_vt_phi_v_bias_full - torch.mm(torch.mm(phi_vt_phi_a_bias_full, torch.inverse(
        phi_at_phi_a_bias_full + lam * torch.eye(phi_at_phi_a_bias_full.shape[0]).to(device))),
                                                   phi_at_phi_v_bias_full) + lam * torch.eye(
        phi_vt_phi_v_bias_full.shape[0]).to(device)

    b_val = phi_vt_yi_full - torch.mm(torch.mm(phi_vt_phi_a_full, torch.inverse(
        phi_at_phi_a_full + lam * torch.eye(phi_at_phi_a_full.shape[0]).to(device))),
                                      phi_at_yi_full) + lam * w_val_last
    b_val_bias = phi_vt_yi_bias_full - torch.mm(torch.mm(phi_vt_phi_a_bias_full, torch.inverse(
        phi_at_phi_a_bias_full + lam * torch.eye(phi_at_phi_a_bias_full.shape[0]).to(device))),
                                                phi_at_yi_bias_full) + lam * w_val_bias_last

    # A_adv = torch.eye(dim * num_actions).to(device) - torch.mm(torch.mm(phi_at_phi_v_full, torch.inverse(
    #     phi_vt_phi_v_full + lam * torch.eye(phi_vt_phi_v_full.shape[0]).to(device))), phi_vt_phi_a_full)
    # A_adv_bias = torch.eye(1 * num_actions).to(device) - torch.mm(torch.mm(phi_at_phi_v_bias_full, torch.inverse(
    #     phi_vt_phi_v_bias_full + lam * torch.eye(phi_vt_phi_v_bias_full.shape[0]).to(device))), phi_vt_phi_a_bias_full)
    #
    # b_adv = torch.mm(torch.inverse(phi_at_phi_a_full + lam * torch.eye(phi_at_phi_a_full.shape[0]).to(device)),
    #                  phi_at_yi_full - torch.mm(torch.mm(phi_at_phi_v_full, torch.inverse(
    #                      phi_vt_phi_v_full + lam * torch.eye(phi_vt_phi_v_full.shape[0]).to(device))),
    #                                            phi_vt_yi_full) + lam * w_adv_last)
    # b_adv_bias = torch.mm(
    #     torch.inverse(phi_at_phi_a_bias_full + lam * torch.eye(phi_at_phi_a_bias_full.shape[0]).to(device)),
    #     phi_at_yi_bias_full - torch.mm(torch.mm(phi_at_phi_v_bias_full, torch.inverse(
    #         phi_vt_phi_v_bias_full + lam * torch.eye(phi_vt_phi_v_bias_full.shape[0]).to(device))),
    #                                    phi_vt_yi_bias_full) + lam * w_adv_bias_last)
    #
    # A_val = torch.eye(dim).to(device) - torch.mm(torch.mm(phi_vt_phi_a_full, torch.inverse(
    #     phi_at_phi_a_full + lam * torch.eye(phi_at_phi_a_full.shape[0]).to(device))), phi_at_phi_v_full)
    # A_val_bias = torch.eye(1 * 1).to(device) - torch.mm(torch.mm(phi_vt_phi_a_bias_full, torch.inverse(
    #     phi_at_phi_a_bias_full + lam * torch.eye(phi_at_phi_a_bias_full.shape[0]).to(device))), phi_at_phi_v_bias_full)
    #
    # b_val = torch.mm(torch.inverse(phi_vt_phi_v_full + lam * torch.eye(phi_vt_phi_v_full.shape[0]).to(device)),
    #                  phi_vt_yi_full - torch.mm(torch.mm(phi_vt_phi_a_full, torch.inverse(
    #                      phi_at_phi_a_full + lam * torch.eye(phi_at_phi_a_full.shape[0]).to(device))),
    #                                            phi_at_yi_full) + lam * w_val_last)
    # b_val_bias = torch.mm(
    #     torch.inverse(phi_vt_phi_v_bias_full + lam * torch.eye(phi_vt_phi_v_bias_full.shape[0]).to(device)),
    #     phi_vt_yi_bias_full - torch.mm(torch.mm(phi_vt_phi_a_bias_full, torch.inverse(
    #         phi_at_phi_a_bias_full + lam * torch.eye(phi_at_phi_a_bias_full.shape[0]).to(device))),
    #                                    phi_at_yi_bias_full) + lam * w_val_bias_last)

    # # Method C:
    # A_adv = phi_at_phi_a_full + lam * torch.eye(phi_at_phi_a_full.shape[0]).to(device)
    # A_adv_bias = phi_at_phi_a_bias_full + lam * torch.eye(phi_at_phi_a_bias_full.shape[0]).to(device)
    # b_adv = phi_at_yi_full - torch.mm(phi_at_phi_v_full, w_val_last) + lam * w_adv_last
    # b_adv_bias = phi_at_yi_bias_full - torch.mm(phi_at_phi_v_bias_full, w_val_bias_last) + lam * w_adv_bias_last
    #
    # A_val = phi_vt_phi_v_full + lam * torch.eye(phi_vt_phi_v_full.shape[0]).to(device)
    # A_val_bias = phi_vt_phi_v_bias_full + lam * torch.eye(phi_vt_phi_v_bias_full.shape[0]).to(device)
    # b_val = phi_vt_yi_full - torch.mm(phi_vt_phi_a_full, w_adv_last) + lam * w_val_last
    # b_val_bias = phi_vt_yi_bias_full - torch.mm(phi_vt_phi_a_bias_full, w_adv_bias_last) + lam * w_val_bias_last

    ## Method B:
    # A_adv = (1.0 / n_srl) * A_adv_full + lam * torch.eye(A_adv_full.shape[0]).to(device)
    # A_adv_bias = (1.0 / n_srl) * A_adv_bias_full + lam * torch.eye(A_adv_bias_full.shape[0]).to(device)
    # b_adv = (1.0 / n_srl) * b_adv_full + lam * w_adv_last
    # b_adv_bias = (1.0 / n_srl) * b_adv_bias_full + lam * w_adv_bias_last
    #
    # A_val = (1.0 / n_srl) * A_val_full + lam * torch.eye(A_val_full.shape[0]).to(device)
    # A_val_bias = (1.0 / n_srl) * A_val_bias_full + lam * torch.eye(A_val_bias_full.shape[0]).to(device)
    # b_val = (1.0 / n_srl) * b_val_full + lam * w_val_last
    # b_val_bias = (1.0 / n_srl) * b_val_bias_full + lam * w_val_bias_last

    w_adv_srl = torch.mm(torch.inverse(A_adv), b_adv).view(num_actions, dim)
    w_bias_adv_srl = torch.mm(torch.inverse(A_adv_bias), b_adv_bias).squeeze()

    w_val_srl = torch.mm(torch.inverse(A_val), b_val).view(1, dim)
    w_bias_val_srl = torch.mm(torch.inverse(A_val_bias), b_val_bias).view(1)

    w_adv_last_dict['weight'] = w_adv_srl
    w_adv_last_dict['bias'] = w_bias_adv_srl

    w_val_last_dict['weight'] = w_val_srl
    w_val_last_dict['bias'] = w_bias_val_srl

    net.fc2_adv.load_state_dict(w_adv_last_dict)
    net.fc2_val.load_state_dict(w_val_last_dict)

    adv_weight_diff = torch.sum((w_adv_last_dict['weight'] - w_adv_last_dict_before['weight']) ** 2)
    adv_bias_diff = torch.sum((w_adv_last_dict['bias'] - w_adv_last_dict_before['bias']) ** 2)
    adv_total_weight_diff = torch.sqrt(adv_weight_diff + adv_bias_diff)

    val_weight_diff = torch.sum((w_val_last_dict['weight'] - w_val_last_dict_before['weight']) ** 2)
    val_bias_diff = torch.sum((w_val_last_dict['bias'] - w_val_last_dict_before['bias']) ** 2)
    val_total_weight_diff = torch.sqrt(val_weight_diff + val_bias_diff)

    print("total weight difference of ls-update:: advantage: %.4f" % adv_total_weight_diff.item(),
          " value: %.4f" % val_total_weight_diff.item())
    print("least-squares step done.")


def calc_fqi_matrices(net, tgt_net, batch, gamma, n_srl, m_batch_size=512, device='cpu',
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
    :param use_boosting: whether or not to use Boosted FQI
    :param use_double_dqn: whether or not to use Double DQN
    :return: A, A_bias, b, b_bias parameters for calculating the LS (np.arrays)
    """
    num_batches = n_srl // m_batch_size
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
            truncated_bellman_error = bellman_error.clamp(-1, 1)
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


def ls_step(net, tgt_net, batch, gamma, n_srl, lam=1.0, m_batch_size=256, device='cpu',
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
    :param use_regularization: whether or not to use regularization
    :param use_boosting: whether or not to use Boosted FQI
    :param use_double_dqn: whether or not to use Double DQN
    :return:
    """
    a, a_bias, b, b_bias = calc_fqi_matrices(net, tgt_net, batch, gamma,
                                             n_srl, m_batch_size=m_batch_size, device=device,
                                             use_boosting=use_boosting,
                                             use_double_dqn=use_double_dqn)
    w_last_before = copy.deepcopy(net.fc2.state_dict())
    w_last_dict = copy.deepcopy(net.fc2.state_dict())
    w_srl, w_b_srl = calc_fqi_w_srl(a.detach(), a_bias.detach(), b.detach(), b_bias.detach(),
                                    w_last_dict['weight'], w_last_dict['bias'], lam=lam, device=device,
                                    use_regularization=use_regularization)
    w_last_dict['weight'] = w_srl.detach()
    w_last_dict['bias'] = w_b_srl.detach()
    net.fc2.load_state_dict(w_last_dict)

    weight_diff = torch.sum((w_last_dict['weight'] - w_last_before['weight']) ** 2)
    bias_diff = torch.sum((w_last_dict['bias'] - w_last_before['bias']) ** 2)
    total_weight_diff = torch.sqrt(weight_diff + bias_diff)
    print("total weight difference of ls-update: ", total_weight_diff.item())
    print("least-squares step done.")
