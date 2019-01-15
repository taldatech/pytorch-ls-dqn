"""
Hyperparameters for environments.
"""

HYPERPARAMS = {
    'pong': {
        'env_name': "PongNoFrameskip-v4",
        'stop_reward': 18.0,
        'run_name': 'pong',
        'replay_size': 100000,
        'replay_initial': 10000,
        'target_net_sync': 1000,
        'epsilon_frames': 10 ** 5,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 32
    },
    'boxing': {
        'env_name': "BoxingNoFrameskip-v4",
        'stop_reward': 90.0,
        'run_name': 'boxing',
        'replay_size': 250000,
        'replay_initial': 10000,
        'target_net_sync': 1000,
        'epsilon_frames': 10 ** 5,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 64
    },
    'breakout-small': {
        'env_name': "BreakoutNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout-small',
        'replay_size': 3 * 10 ** 5,
        'replay_initial': 20000,
        'target_net_sync': 1000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 64
    },
    'breakout': {
        'env_name': "BreakoutNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
    'invaders': {
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'invaders',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
    'upndown': {
        'env_name': "UpNDownNoFrameskip-v4",
        'stop_reward': 35000.0,
        'run_name': 'upndown',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
    'phoenix': {
        'env_name': "PhoenixFrameskip-v4",
        'stop_reward': 25000.0,
        'run_name': 'phoenix',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
}
