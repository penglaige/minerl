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
num_processes = 8
seed = 1
gamma = 0.95
log_dir = None
torch.set_num_threads(1)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda:0" if args.cuda else "cpu")

envs = make_vec_envs(task, seed, num_processes, gamma, log_dir, device, False)

print("------ envs test done! ---------")


