import argparse



def get_args():
    parser = argparse.ArgumentParser(description='RL',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--train',
        help='train or mini-train for test',
        action='store_true',
        default=False)
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=False,
        help='whether to use gpu')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--task', 
        help='experiment task', 
        type=str)
    parser.add_argument(
        '--seed', 
        help='RNG seed', 
        type=int, 
        default=0)
    parser.add_argument(
        '--frame-history-len',
        type=int,
        default=1,
        help='frame history length (default: 4)')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.00025,
        help='learning rate (default: 0.00025)')
    parser.add_argument(
        '--eps',
        type=float,
        default=0.01,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.95,
        help='RMSprop optimizer alpha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (dafault: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--demo',
        help='pre-train with human demonstration',
        action='store_true',
        default=False)
    parser.add_argument(
        '--pre_train_steps', 
        help='human demo pre-train steps',
        type=int,
        default=100000)
    parser.add_argument(
        '--algo',default='ppo',help='algorithm to use: a2c | ppo | dddqn')
    parser.add_argument(
        '--replay_buffer_size',
        type=int,
        default=1000000,
        help='replay buffer size')
    parser.add_argument(
        '--exploration_schedule',
        type=int,
        default=1000000,
        help='exploration schedule')
    parser.add_argument(
        '--learning_starts',
        type=int,
        default=60000,
        help='start learning after steps')
    parser.add_argument(
        '--prioritized_replay_beta_iters',
        type=int,
        default=5500000,
        help='prioritized_replay_beta_iters')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy tern coefficient (default:0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many traing CPU processes to use (default: 1)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=10,
        help='number of forward steps in A2C (default: 10)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default:4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=5,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-eps',
        type=int,
        default=1000,
        help='number of environment eps to train (default: 1000)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--save-model-dir',
        default='./models/',
        help='directory to save models (default: ./models/)')
    parser.add_argument(
        '--log-dir',
        default='./logs/',
        help='directory t0 save agent logs (default: ./logs/)')
    parser.add_argument(
        '--save-performance-dir',
        default='./perform_records/',
        help='directory to save performance records (default: ./perform_records/)')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    args = parser.parse_args()

    assert args.algo in ['a2c', 'ppo']
    
    return args
    

def parse_obs_space(obs_space):
    """
    input: obs_space : gym.space
    output:
        pixel_shape : (64,64,3)
        non_pixel_obs: name of different non pixel features
        non_pixel_shape: len of non_pixel_obs
    """
    pixel_shape = None
    non_pixel_shape = None
    non_pixel_obs = []
    for obs in obs_space.spaces:
        if obs == 'pov':
            pixel_shape = obs_space[obs].shape
        else:
            if obs == 'inventory':
                items = obs_space[obs]
                for item in items.spaces:
                    non_pixel_obs.append(item)
            elif obs == 'compassAngle':
                non_pixel_obs.append(obs)
            elif obs == 'equipped_items':
                equipped_items = obs_space[obs]
                for hand in equipped_items.spaces:
                    hand_items = equipped_items[hand]
                    for item in hand_items.spaces:
                        item_type = hand_items[item].__class__.__name__
                        if item_type == 'Box':
                            non_pixel_obs.append(item)
                        elif item_type == 'Enum':
                            vals = hand_items[item].values
                            for val in vals:
                                non_pixel_obs.append(val)

    non_pixel_shape = len(non_pixel_obs)
    return pixel_shape, non_pixel_obs, non_pixel_shape

def parse_action_space(action_space):
    """
    input: action_space: gym.space
    output:
        for ppo
        action_spaces: output size for different action dimension
        action_spaces_name: name of different dimension of action spaces
    """
    action_spaces = []
    action_spaces_name = []
    for act in action_space.spaces:
        if action_space[act].__class__.__name__ == 'Discrete':
            action_spaces_name.append(act)
            action_spaces.append(action_space[act].n)
        elif action_space[act].__class__.__name__ == 'Enum':
            action_spaces_name.append(act)
            action_spaces.append(len(action_space[act].values))
        elif act == 'camera':
            action_spaces_name.append('pitch')
            action_spaces_name.append('yaw')
            action_spaces.append(1)
            action_spaces.append(1)

    return action_spaces, action_spaces_name

def dddqn_parse_action_space(action_space):
    """
    input: action_space: gym.space
    output:
        for dddqn
        action_spaces: output size for different action dimension
        action_spaces_name: name of different dimension of action spaces
    """
    action_spaces = []
    action_spaces_name = []
    for act in action_space.spaces:
        if action_space[act].__class__.__name__ == 'Discrete':
            action_spaces_name.append(act)
            action_spaces.append(action_space[act].n)
        elif action_space[act].__class__.__name__ == 'Enum':
            action_spaces_name.append(act)
            action_spaces.append(len(action_space[act].values))
        elif act == 'camera':
            action_spaces_name.append('pitch')
            action_spaces_name.append('yaw')
            action_spaces.append(36)
            action_spaces.append(36)

    return action_spaces, action_spaces_name

