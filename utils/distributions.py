# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
# sample is a function name
# 不改变的话 sample的output size 是 [batch]
# make the outout size to batch x 1
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
# ???
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, 
    actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1,keepdim=True)

# Normal /Gaussian distribution
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
# log_prob is function name!
# log_prob_normal(action) size: (batch, 1)?
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = lambda self, actions: log_prob_bernoulli(
    self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1,-1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        
        return x + bias

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        # parameter initialization
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
    
    def get_prob(self, x):
        x = self.linear(x)
        return x

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        # action_mean size: batch x 1

        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

    def get_prob(self, x):
        action_mean = self.fc_mean(x)
        return action_mean

class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

        def forward(self, x):
            x = self.linear(x)
            return FixedBernoulli(logits=x)