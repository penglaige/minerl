import torch
import torch.nn as nn
import numpy as np

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

class Branches_dueling_DQN(nn.Module):
    """This model takes as input an observation and returns values of all actions.

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
    def __init__(self, convs, in_channels, in_feature, hidden_actions, hidden_value, num_branches, num_actions_for_branch,aggregator="reduceLocalMean"):
        super(Branches_dueling_DQN, self).__init__()

        self.num_branches = num_branches
        self.num_actions_for_branch = num_actions_for_branch
        self.aggregator = aggregator

        self.conv_idxes = []
        self.action_branches_idxes = []
        self.value_idxes = []
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
        for i in range(num_branches):
            action_branch = []
            for l in range(len(hidden_actions)):
                out_features = hidden_actions[l]
                if l == 0:
                    in_features = in_feature
                else:
                    in_features = hidden_actions[l-1]
                setattr(self, "fc"+ str(fc_l), nn.Linear(in_features=in_features,out_features=out_features))
                action_branch.append(fc_l)
                fc_l += 1
            setattr(self, "fc" + str(fc_l), nn.Linear(in_features=hidden_actions[-1],out_features=num_actions_for_branch[i]))
            action_branch.append(fc_l)
            fc_l += 1
            self.action_branches_idxes.append(action_branch)
        
        # value net
        for i in range(len(hidden_value)):
            out_features = hidden_value[i]
            if i == 0:
                in_features = in_feature
            else:
                in_features = hidden_value[i - 1]
            setattr(self, "fc" + str(fc_l), nn.Linear(in_features=in_features,out_features=out_features))
            self.value_idxes.append(fc_l)
            fc_l += 1
        setattr(self, "fc" + str(fc_l), nn.Linear(in_features=hidden_value[-1],out_features=1))
        self.value_idxes.append(fc_l)

        self.relu = nn.ReLU()
        

    def forward(self,x):
        batch_size = x.size(0)
        total_num_actions = 0
        for i in range(self.num_branches):
            total_num_actions += self.num_actions_for_branch[i]
        
        # through conv nets
        for idx in self.conv_idxes:
            x = self.relu(self.__getattr__("conv" + str(idx))(x))

        x = x.view(x.size(0), -1)
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
                adv -= adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions_for_branch[i])
            elif (self.aggregator == "reduceLocalMax"):
                adv_max = adv.max(1)[0].view(adv.size(0),1)
                adv -= adv_max.expand(x.size(0), self.num_actions_for_branch[i])
            
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

class Branches_dueling_DQN2(nn.Module):
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
    def __init__(self, non_pixel_input, non_pixel_layer,convs, in_channels, in_feature, hidden_actions, hidden_value, num_branches, num_actions_for_branch,aggregator="reduceLocalMean"):
        super(Branches_dueling_DQN2, self).__init__()

        self.num_branches = num_branches
        self.num_actions_for_branch = num_actions_for_branch
        self.aggregator = aggregator

        self.conv_idxes = []
        self.action_branches_idxes = []
        self.value_idxes = []
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
        for i in range(num_branches):
            action_branch = []
            for l in range(len(hidden_actions)):
                out_features = hidden_actions[l]
                if l == 0:
                    in_features = in_feature + non_pixel_layer[-1]
                else:
                    in_features = hidden_actions[l-1]
                setattr(self, "fc"+ str(fc_l), nn.Linear(in_features=in_features,out_features=out_features))
                action_branch.append(fc_l)
                fc_l += 1
            setattr(self, "fc" + str(fc_l), nn.Linear(in_features=hidden_actions[-1],out_features=num_actions_for_branch[i]))
            action_branch.append(fc_l)
            fc_l += 1
            self.action_branches_idxes.append(action_branch)
        
        # value net
        for i in range(len(hidden_value)):
            out_features = hidden_value[i]
            if i == 0:
                in_features = in_feature + non_pixel_layer[-1]
            else:
                in_features = hidden_value[i - 1]
            setattr(self, "fc" + str(fc_l), nn.Linear(in_features=in_features,out_features=out_features))
            self.value_idxes.append(fc_l)
            fc_l += 1
        setattr(self, "fc" + str(fc_l), nn.Linear(in_features=hidden_value[-1],out_features=1))
        self.value_idxes.append(fc_l)

        # non pixel feature net
        fc_l += 1
        for i in range(len(non_pixel_layer)):
            out_features = non_pixel_layer[i]
            if i == 0:
                in_features = non_pixel_input
            else:
                in_features = non_pixel_layer[i - 1]
            setattr(self, "fc" + str(fc_l), nn.Linear(in_features=in_features,out_features=out_features))
            self.non_pixel_idxes.append(fc_l)
            fc_l += 1

        self.relu = nn.ReLU()
        

    def forward(self,x,non_pixel_feature):
        batch_size = x.size(0)
        total_num_actions = 0
        for i in range(self.num_branches):
            total_num_actions += self.num_actions_for_branch[i]
        
        # calculate non_pixel_feature
        for idx in self.non_pixel_idxes:
            non_pixel_feature = self.relu(self.__getattr__("fc" + str(idx))(non_pixel_feature))

        # through conv nets
        for idx in self.conv_idxes:
            x = self.relu(self.__getattr__("conv" + str(idx))(x))

        x = x.view(x.size(0), -1)
        # combine non pixel feature with pixel feature
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
                adv -= adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions_for_branch[i])
            elif (self.aggregator == "reduceLocalMax"):
                adv_max = adv.max(1)[0].view(adv.size(0),1)
                adv -= adv_max.expand(x.size(0), self.num_actions_for_branch[i])
            
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