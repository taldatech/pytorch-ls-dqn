"""
The LS-DQN experiment:
For each checkpoint of the vanilla DQN, calculate the mean of 20 rollouts, then retrain
the last layer using SRL algorithm and calculate the mean of another 20 rollouts.
"""

import gym
import torch
import torch.optim as optim

import ls_dqn_model.utils.dqn_model as dqn_model
from ls_dqn_model.utils.hyperparameters import HYPERPARAMS
from ls_dqn_model.utils.agent import DQNAgent, TargetNet
from ls_dqn_model.utils.actions import EpsilonGreedyActionSelector
import ls_dqn_model.utils.experience as experience
import ls_dqn_model.utils.utils as utils
from ls_dqn_model.utils.srl_algorithms import ls_step, ls_step_dueling
import ls_dqn_model.utils.wrappers as wrappers
import numpy as np
import random
import pickle

"""
DQN
"""
# ckpt_list = ["./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-SEED-10_100000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-SEED-10_200000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-SEED-10_300000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-SEED-10_400000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-SEED-10_500000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-SEED-10_510229.pth"]

"""
DDQN
"""
# ckpt_list = ["./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-SEED-10_100000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-SEED-10_200000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-SEED-10_300000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-SEED-10_400000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-SEED-10_500000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-SEED-10_600000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-SEED-10_700000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-SEED-10_800000.pth"]

"""
Dueling DQN
"""

ckpt_list = ["./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DUELING-SEED-10_100000.pth",
             "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DUELING-SEED-10_200000.pth",
             "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DUELING-SEED-10_300000.pth",
             "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DUELING-SEED-10_400000.pth",
             "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DUELING-SEED-10_500000.pth",
             "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DUELING-SEED-10_517405.pth"]

"""
Dueling DDQN
"""

# ckpt_list = ["./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-DUELING-SEED-10_100000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-DUELING-SEED-10_200000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-DUELING-SEED-10_300000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-DUELING-SEED-10_400000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-DUELING-SEED-10_500000.pth",
#              "./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-DOUBLE-DUELING-SEED-10_561258.pth"]


if __name__ == "__main__":
    params = HYPERPARAMS['pong']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = gym.make(params['env_name'])
    env = wrappers.wrap_dqn(env)

    # ckpt_list = ["./pong_agent_ckpt/pong_agent_ls_dqn_-DQN-BATCH-64-SEED-10_300000.pth"]

    training_random_seed = 10
    n_srl = params['replay_size']  # size of batch in SRL step
    use_double_dqn = False
    use_dueling_dqn = True
    use_boosting = True
    use_ls_dqn = True
    use_constant_seed = True  # to compare performance independently of the randomness

    # lam = 100.0  # regularization parameter
    if use_boosting:
        lams = [10000, 1000, 100, 10, 1]
        # lams = [10000]
    else:
        lams = [100, 10, 1, 0.1, 0.01]
        # lams = [0.1]
    num_rollouts = 20  # number of episodes to evaluate the weights
    params['batch_size'] = 64

    for lam in lams:
        if use_ls_dqn:
            print("using ls-dqn with lambda:", str(lam))
            model_name = "LSDQN-LAM-" + str(lam)
        else:
            model_name = "DQN"
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
            print("using constant seed of ", training_random_seed)

        if use_dueling_dqn:
            net = dqn_model.DuelingLSDQN(env.observation_space.shape, env.action_space.n).to(device)
        else:
            net = dqn_model.LSDQN(env.observation_space.shape, env.action_space.n).to(device)
        tgt_net = TargetNet(net)

        selector = EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
        epsilon_tracker = utils.EpsilonTracker(selector, params)

        agent = DQNAgent(net, selector, device=device)

        exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
        buffer = experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
        optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])  # TODO: change to RMSprop
        ckpt_dqn_rewards = []
        ckpt_lsdqn_rewards = []
        for ckpt in ckpt_list:
            print("current checkpoint: ", ckpt)
            utils.load_agent_state(net, optimizer, selector, path=ckpt, copy_to_target_network=True,
                                   load_optimizer=False, target_net=tgt_net, buffer=buffer, load_buffer=True)
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
                        print("dqn-game ", i, "reward: ", game_reward)
                        break
            ckpt_dqn_rewards.append(1.0 * total_reward / num_rollouts)
            # retrain using ls-dqn
            print("performing ls step...")
            batch = buffer.sample(n_srl)
            if use_dueling_dqn:
                ls_step_dueling(net, tgt_net.target_model, batch, params['gamma'], len(buffer), lam=lam,
                                m_batch_size=256,
                                device=device,
                                use_boosting=use_boosting, use_double_dqn=use_double_dqn)
            else:
                ls_step(net, tgt_net.target_model, batch, params['gamma'], len(buffer), lam=lam,
                        m_batch_size=256, device=device, use_boosting=use_boosting,
                        use_double_dqn=use_double_dqn)
            total_reward = 0.0
            for i in range(num_rollouts):
                game_reward = 0.0
                state = env.reset()
                while True:
                    action, _ = agent([state])
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    game_reward += reward
                    if done:
                        print("lsdqn-game ", i, "reward: ", game_reward)
                        break
            ckpt_lsdqn_rewards.append(1.0 * total_reward / num_rollouts)
            print(ckpt.split('/')[-1], "- dqn mean rewards: {}, lsdqn mean rewards: {}"
                  .format(ckpt_dqn_rewards[-1], ckpt_lsdqn_rewards[-1]))
        path_to_results = "./exp_results/" + model_name + ".results"
        with open(path_to_results, 'wb') as fp:
            exp_results = {}
            exp_results['dqn'] = ckpt_dqn_rewards
            exp_results['lsdqn'] = ckpt_lsdqn_rewards
            pickle.dump(exp_results, fp)
        print("experiment done.")
