import gym
import minerl
import numpy as np
import torch

from model import Policy
from ppo import PPO
from utils.schedules import *
from utils.parser import get_args, parse_obs_space
from utils.storage import RolloutStorage
from utils.utils import *
from utils.replay_buffer import ReplayBuffer
from utils.envs import make_vec_envs

from builtins import range
from past.utils import old_div

import os
import sys
import time
import copy
import glob
import json
import random
import logging
import struct
import socket
from collections import deque

task = 'MineRLNavigateDense-v0'
num_processes = 2
seed = 1
gamma = 0.95
log_dir = None
torch.set_num_threads(1)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

envs = make_vec_envs(task, seed, num_processes, gamma, log_dir, device, False)

print("------ envs test1 done! ---------")

#print("envs.observation_space:",envs.observation_space)
#print("envs.action_space: ",envs.action_space)

obs = envs.reset()
print("obs: ",obs)
print("pov the same?:", (obs['pov'][0]==obs['pov'][1]).all())

print("------ envs test2 done! ---------")

action1 = [0,1,-14,-13,0,0,1,0,0,1,0]
action2 = [1,0, 15,-1, 1,0,1,0,0,1,1]
actions = [action1, action2]
actions = torch.tensor(actions)
#actions = envs.venv.return_actions()
print("------ envs test3 done! ---------")
# input actions : torch.tensor (num_process, num_branches)
obs, reward, done, infos = envs.step(actions)
print("obs2: ",obs)
print("pov the same?:", (obs['pov'][0]==obs['pov'][1]).all())
print("------ envs test4 done! ---------")
