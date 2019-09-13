import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def _faltten_helper(T, N, _tensor):
    