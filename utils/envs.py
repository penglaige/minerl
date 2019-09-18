import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from utils.parser import parse_obs_space

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        env = gym.make(env_id)

        env.seed(seed + rank)

        obs_space = env.observation_space

        pixel_shape, non_pixel_name, non_pixel_shape = parse_obs_space(obs_space)


        if log_dir is not None:
            env = bench.Monitro(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)


        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        # env = TransposeImage(env, op=[2,0,1])

        return env

    return _thunk

def make_vec_envs(env_name,
                seed,
                num_processes,
                gamma,
                log_dir,
                device,
                allow_early_resets,
                num_frame_stack=None):
    envs = [make_env(env_name, seed, i, log_dir, allow_early_resets)
            for i in range(num_processes)]
    
    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)
    
    envs = VecPyTorch(envs, device)

    """
    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    else:
        envs = VecPyTorchFrameStack(envs, 4, device)
    """
    return envs

class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose obervation space (base class)
        """
        super(TransposeObs, self).__init__(env)

class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0 ,1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space['pov'].shape
        self.observation_space['pov'] = Box(
            self.observation_space['pov'].low[0,0,0],
            self.observation_space['pov'].high[0,0,0],
            [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
            dtype=self.observation_space['pov'].dtype
        )

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every 'skip'-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        #obs = torch.from_numpy(obs).float().to(self.device)
        return obs
        
    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        #obs = torch.from_numpy(obs).float().to(self.device)
        #reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
"""
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        obs_space = venv.observation_space  #warpped ob space

        pixel_shape, non_pixel_name, non_pixel_shape = parse_obs_space(obs_space)

        wos = pixel_shape
        self.shape_dim0
"""
