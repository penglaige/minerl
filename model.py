#https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.distributions import Bernoulli, Categorical, DiagGaussian, init
from utils.parser import parse_action_space, parse_obs_space, dddqn_parse_action_space

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class Policy(nn.Module):
    def __init__(self, obs_space, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()

        pixel_shape, non_pixel_obs, non_pixel_shape = parse_obs_space(obs_space)
        action_spaces, action_spaces_name = parse_action_space(action_space)
        # pixel:tuple (h,w,c),
        # non_pixel_obs = ['name']
        # non_pixel_shape :int = len(non_pixel_obs)
        # action_spaces: [2,2,1,1...]
        # action_spaces_name = ['attack',...]

        if base_kwargs is None:
            base_kwargs = {}
        base = Branch_CNNBase

        if non_pixel_shape == 0:
            add_non_pixel = False
        else:
            add_non_pixel = True

        # arguments
        non_pixel_layer = base_kwargs['non_pixel_layer']
        convs           = base_kwargs['convs']
        in_channels     = base_kwargs['frame_history_len'] * pixel_shape[2]
        in_feature      = base_kwargs['in_feature']
        hidden_actions  = base_kwargs['hidden_actions']
        hidden_value    = base_kwargs['hidden_value']
        aggregator      = base_kwargs['aggregator']

        self.num_branches = len(action_spaces)
        self.base = base(add_non_pixel, non_pixel_shape,
            non_pixel_layer, convs, in_channels,
            in_feature, hidden_actions, hidden_value,
            action_spaces,aggregator)

        self.dist_idxes = []
        dist_l = 1
        for i in range(self.num_branches):
            if(action_spaces[i] == 1):
                # continuous action space
                num_outputs = action_spaces[i]
                num_inputs = hidden_actions[-1]
                setattr(self, "dist" + str(dist_l), DiagGaussian(num_inputs, num_outputs))
                self.dist_idxes.append(dist_l)
                dist_l += 1
            else:
                # discrete action space
                num_inputs = hidden_actions[-1]
                num_outputs = action_spaces[i]
                setattr(self, "dist" + str(dist_l), Categorical(num_inputs, num_outputs))
                self.dist_idxes.append(dist_l)
                dist_l += 1

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, x, non_pixel_features, deterministic=False):
        value, actor_features = self.base(x, non_pixel_features)

        # calculate dist for different branches
        batch_size = x.size(0)
        dists = []
        actions = []
        action_log_probs = []

        for i in range(self.num_branches):
            idx = self.dist_idxes[i]
            # actor_feature: torch (batch, 128)
            actor_feature = actor_features[i]
            dist = self.__getattr__("dist" + str(idx))(actor_feature)
            dists.append(dist)

            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

            # use action() to get values.
            # action: size: batch x 1
            action = action.type(dtype)
            actions.append(action)
            
            action_log_prob = dist.log_probs(action)
            # action_log_prob size:  batch x 1
            action_log_probs.append(action_log_prob)

            dist_entropy = dist.entropy().mean()

        actions = torch.cat(actions,1).reshape(batch_size,-1)
        action_log_probs = torch.cat(action_log_probs, 1).reshape(batch_size, -1)

        # value size: batch x 1
        # actions size: list, list[0].size() == 32 x action_num for each dimension
        # action_log_probs : list, len = branches, 32 x hidden_feature_size
        return value, actions, action_log_probs

    def get_value(self, x, non_pixel_features):
        value, _ = self.base(x, non_pixel_features)
        return value

    def evaluate_actions(self, x, non_pixel_features, actions):
        """
        x: torch.tensor (batch, c,h,w)
        non_pixel_features: torch.tensor (batch, num_features)
        actions: torch.tensor (batch, num_branches)
        """
        batch_size = actions.size(0)
        value, actor_features = self.base(x, non_pixel_features)

        # calculate dist for different branches
        action_log_probs = []
        dist_entropys = []

        for i in range(self.num_branches):
            idx = self.dist_idxes[i]
            actor_feature = actor_features[i]
            dist = self.__getattr__("dist" + str(idx))(actor_feature)

            #
            action = actions[:,i].view(actions.size(0),-1)
            
            action_log_prob = dist.log_probs(action)
            action_log_probs.append(action_log_prob)

            dist_entropy = dist.entropy().mean()
            dist_entropys.append(dist_entropy)

        action_log_probs = torch.cat(action_log_probs, 1).reshape(batch_size, -1)
        #dist_entropys = torch.cat(dist_entropys, 1).reshape(batch_size, -1)
        # dist_entropys len() = branches, for each action space dimension
        # dist_entropys: [dist_entropy:  tensor(0.6931, grad_fn=<MeanBackward0>) torch.Size([]) ]
        return value, action_log_probs, dist_entropys


                  
class Branch_CNNBase(nn.Module):
    """
    PPO network.
    Similar with Branches_dueling_DQN2.
    Test finished.
    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)

    in_channels: int
        Channels number of the input

    in_feature: int
        Number of features input to branches

    hiddens_actions: [int]
        list of sizes of hidden layers in the action-value/advantage branches -- 
        currently assumed the same across all such branches 

    hiddens_value: [int]
        list of sizes of hidden layers for the state-value branch 

    non_pixel_layer: [int]
        list of sizes of hidden layers for non_pixel_features

    num_branches: int
        number of action branches (= num_action_dims in current implementation)

    action_spaces: [int]
        number of actions for each branches

    dueling: bool
        if using dueling, then the network structure becomes similar to that of 
        dueling (i.e. Q = f(V,A)), but with N advantage branches as opposed to only one, 
        and if not dueling, then there will be N branches of Q-values  

    aggregator: str
        aggregator method used for dueling architecture: {naive, reduceLocalMean, reduceLocalMax, reduceGlobalMean, reduceGlobalMax}

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    def __init__(self, add_non_pixel, non_pixel_shape, 
        non_pixel_layer, convs, in_channels, 
        in_feature, hidden_actions, hidden_value,
        action_spaces, aggregator="reduceLocalMean"):
        super(Branch_CNNBase, self).__init__()

        self.add_non_pixel = add_non_pixel
        self.num_branches = len(action_spaces)
        self.action_spaces = action_spaces
        self.aggregator = aggregator

        self.conv_idxes = []
        self.action_branches_idxes = []
        self.value_idxes = []
        self.dist_idxes = []

        if add_non_pixel:
            self.non_pixel_idxes = []
        layer = 1

        # shared conv layers
        init_ = lambda m: init(m, nn.init.orthogonal_, 
                                lambda x: nn.init.constant_(x, 0),
                                nn.init.calculate_gain('relu'))

        for i in range(len(convs)):
            out_channels, kernel_size, stride = convs[i]
            if i == 0:
                in_channel = in_channels
            else:
                in_channel = convs[i-1][0]
            setattr(self, "conv" + str(layer), init_(nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=kernel_size, stride=stride)))
            self.conv_idxes.append(layer)
            layer += 1

        fc_l = 1
        # action branches
        for i in range(self.num_branches):
            action_branch = []
            for l in range(len(hidden_actions)):
                out_features = hidden_actions[l]
                if l == 0:
                    in_features = in_feature
                    if self.add_non_pixel:
                        in_features += non_pixel_layer[-1]
                else:
                    in_features = hidden_actions[l - 1]
                setattr(self, "fc"+ str(fc_l), init_(nn.Linear(in_features=in_features,out_features=out_features)))
                action_branch.append(fc_l)
                fc_l += 1
            # need to add final layer (distribution layer)
            # return hidden_action_features for each action dimension
            self.action_branches_idxes.append(action_branch)

        # non pixel feature net
        if self.add_non_pixel:
            for i in range(len(non_pixel_layer)):
                out_features = non_pixel_layer[i]
                if i == 0:
                    in_features = non_pixel_shape
                else:
                    in_feature = non_pixel_layer[i - 1]
                setattr(self, "fc" + str(fc_l), init_(nn.Linear(in_features=in_features,out_features=out_features)))
                self.non_pixel_idxes.append(fc_l)
                fc_l += 1
        
        # critic value
        for i in range(len(hidden_value)):
            out_features = hidden_value[i]
            if i == 0:
                in_features = in_feature
                if self.add_non_pixel:
                    in_features += non_pixel_layer[-1]
            else:
                in_features = hidden_value[i - 1]
            setattr(self, "fc" + str(fc_l), init_(nn.Linear(in_features=in_features,out_features=out_features)))
            self.value_idxes.append(fc_l)
            fc_l += 1
        
        init_ = lambda m: init(m, nn.init.orthogonal_,
                        lambda x:nn.init.constant_(x, 0))

        setattr(self, "fc" + str(fc_l), init_(nn.Linear(in_features=hidden_value[-1],out_features=1)))
        self.value_idxes.append(fc_l)
        
        self.relu = nn.ReLU()

        self.train()
    
    def forward(self, x, non_pixel_feature=None):
        # x size: batch_size, c, h, w
        batch_size = x.size(0)
        total_num_actions = 0
        for i in range(self.num_branches):
            total_num_actions += self.action_spaces[i]

        # calculate non_pixel_feature
        if self.add_non_pixel:
            for idx in self.non_pixel_idxes:
                non_pixel_feature = self.relu(self.__getattr__("fc" + str(idx))(non_pixel_feature))

        # calculate conv nets
        for idx in self.conv_idxes:
            x = self.relu(self.__getattr__("conv" + str(idx))(x))

        # flatten
        x = x.view(batch_size, -1)

        # combine non pixel feature with pixel feature
        if self.add_non_pixel:
            x = torch.cat((non_pixel_feature, x), 1)
        

        # action branches
        actor_features = []
        for i in range(self.num_branches):
            actor_feature = x
            actor_layers = self.action_branches_idxes[i]
            for idx in actor_layers:
                actor_feature = self.relu(self.__getattr__("fc" + str(idx))(actor_feature))

            # reduce ? aggregator?
            actor_features.append(actor_feature)

        # value branch
        val = x
        for idx in self.value_idxes:
            if idx == self.value_idxes[-1]:
                val = self.__getattr__("fc" + str(idx))(val)
            else:
                val = self.relu(self.__getattr__("fc" + str(idx))(val))

        return val, actor_features

            



class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3,stride=1)

        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x


class Branch_DDDQN(nn.Module):
    """This model takes as input an observation and returns values of all actions.
    With non-pixel features.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)

    in_channels: int
        Channels number of the input

    in_feature: int
        Number of features input to branches

    hiddens_actions: [int]
        list of sizes of hidden layers in the action-value/advantage branches -- 
        currently assumed the same across all such branches 

    hiddens_value: [int]
        list of sizes of hidden layers for the state-value branch 

    num_branches: int
        number of action branches (= num_action_dims in current implementation)

    num_actions_for_branch: [int]
        number of actions for each branches

    dueling: bool
        if using dueling, then the network structure becomes similar to that of 
        dueling (i.e. Q = f(V,A)), but with N advantage branches as opposed to only one, 
        and if not dueling, then there will be N branches of Q-values  

    aggregator: str
        aggregator method used for dueling architecture: {naive, reduceLocalMean, reduceLocalMax, reduceGlobalMean, reduceGlobalMax}

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    def __init__(self, obs_space, act_space, 
            non_pixel_layer, convs, 
            in_channels, in_feature, 
            hidden_actions, hidden_value, 
            aggregator="reduceLocalMean"):
        super(Branch_DDDQN, self).__init__()
        
        pixel_shape, non_pixel_obs, non_pixel_shape = parse_obs_space(obs_space)
        action_spaces, action_spaces_name = dddqn_parse_action_space(act_space)

        if non_pixel_shape == 0:
            self.add_non_pixel = False
        else:
            self.add_non_pixel = True

        self.num_branches = len(action_spaces)
        self.action_spaces = action_spaces

        self.aggregator = aggregator

        self.conv_idxes = []
        self.action_branches_idxes = []
        self.value_idxes = []

        if self.add_non_pixel:
            self.non_pixel_idxes = []
        layer = 1

        # shared conv layers
        
        for i in range(len(convs)):
            out_channels, kernel_size, stride = convs[i]
            if i == 0:
                in_channel = in_channels
            else:
                in_channel = convs[i-1][0]
            setattr(self, "conv"+str(layer), nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=kernel_size, stride=stride))
            self.conv_idxes.append(layer)
            layer += 1
        
        fc_l = 1
        # action branches
        for i in range(self.num_branches):
            action_branch = []
            for l in range(len(hidden_actions)):
                out_features = hidden_actions[l]
                if l == 0:
                    in_features = in_feature
                    if self.add_non_pixel:
                        in_features += non_pixel_layer[-1]
                else:
                    in_features = hidden_actions[l-1]
                setattr(self, "fc"+ str(fc_l), nn.Linear(in_features=in_features,out_features=out_features))
                action_branch.append(fc_l)
                fc_l += 1
            # output layer
            setattr(self, "fc" + str(fc_l), nn.Linear(in_features=hidden_actions[-1],out_features=self.action_spaces[i]))
            action_branch.append(fc_l)
            fc_l += 1
            self.action_branches_idxes.append(action_branch)
        
        # value net
        for i in range(len(hidden_value)):
            out_features = hidden_value[i]
            if i == 0:
                in_features = in_feature
                if self.add_non_pixel:
                    in_features += non_pixel_layer[-1]
            else:
                in_features = hidden_value[i - 1]
            setattr(self, "fc" + str(fc_l), nn.Linear(in_features=in_features,out_features=out_features))
            self.value_idxes.append(fc_l)
            fc_l += 1
        setattr(self, "fc" + str(fc_l), nn.Linear(in_features=hidden_value[-1],out_features=1))
        self.value_idxes.append(fc_l)

        # non pixel feature net
        fc_l += 1
        if self.add_non_pixel:
            for i in range(len(non_pixel_layer)):
                out_features = non_pixel_layer[i]
                if i == 0:
                    in_features = non_pixel_shape
                else:
                    in_features = non_pixel_layer[i - 1]
                setattr(self, "fc" + str(fc_l), nn.Linear(in_features=in_features,out_features=out_features))
                self.non_pixel_idxes.append(fc_l)
                fc_l += 1

        self.relu = nn.ReLU()
        

    def forward(self,x,non_pixel_feature=None):
        batch_size = x.size(0)
        total_num_actions = 0
        for i in range(self.num_branches):
            total_num_actions += self.action_spaces[i]
        
        # calculate non_pixel_feature
        if self.add_non_pixel:
            for idx in self.non_pixel_idxes:
                non_pixel_feature = self.relu(self.__getattr__("fc" + str(idx))(non_pixel_feature))

        # through conv nets
        for idx in self.conv_idxes:
            x = self.relu(self.__getattr__("conv" + str(idx))(x))

        x = x.view(x.size(0), -1)
        # combine non pixel feature with pixel feature
        if self.add_non_pixel:
            x = torch.cat((non_pixel_feature,x),1)

        val = x
        
        # action branches
        for i in range(self.num_branches):
            adv = x
            adv_layers = self.action_branches_idxes[i]
            for idx in adv_layers:
                if idx == adv_layers[-1]:
                    adv = self.__getattr__("fc" + str(idx))(adv)
                else:
                    adv = self.relu(self.__getattr__("fc" + str(idx))(adv))

            # reduce 
            if (self.aggregator == "reduceLocalMean"):
                adv -= adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_spaces[i])
            elif (self.aggregator == "reduceLocalMax"):
                adv_max = adv.max(1)[0].view(adv.size(0),1)
                adv -= adv_max.expand(x.size(0), self.action_spaces[i])
            
            if i == 0:
                adv_out = adv
            else:
                adv_out = torch.cat((adv_out,adv),1)
        #print(adv_out.size())

        # value branch
        for idx in self.value_idxes:
            if idx == self.value_idxes[-1]:
                val = self.__getattr__("fc" + str(idx))(val).expand(x.size(0), total_num_actions)
            else:
                val = self.relu(self.__getattr__("fc" + str(idx))(val))

        if(self.aggregator == "reduceGlobalMean"):
            x = val + adv_out - adv_out.mean(1).unsqueeze(1).expand(x.size(0), total_num_actions)
        elif (self.aggregator == "reduceGlobalMax"):
            adv_out_max=adv_out.max(1)[0].view(adv_out.size(0),1)
            x = val + adv_out - adv_out_max.expand(x.size(0), total_num_actions)
        else:
            x = val + adv_out
        #print(x.size()) = [batch_size, total_actions]

        return x