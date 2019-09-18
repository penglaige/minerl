import numpy as np
import os
import pickle

def get_obs_features(obs_space, last_obs):
    last_pov = None
    last_non_pixel_features = []
    non_pixel_obs = []
    for obs in obs_space.spaces:
        if obs == 'pov':
            last_pov = last_obs[obs]
        else:
            if obs == 'inventory':
                items = obs_space[obs]
                for item in items.spaces:
                    last_non_pixel_features.append(last_obs[obs][item])
            elif obs == 'compassAngle':
                last_non_pixel_features.append(last_obs[obs])
            elif obs == 'equipped_items':
                equipped_items = obs_space[obs]
                for hand in equipped_items.spaces:
                    hand_items = equipped_items[hand]
                    for item in hand_items.spaces:
                        item_type = hand_items[item].__class__.__name__
                        if item_type == 'Box':
                            last_non_pixel_features.append(last_obs[obs][hand][item])
                        elif item_type == 'Enum':
                            vals = hand_items[item].values
                            for val in vals:
                                if val == last_obs[obs][hand][item]:
                                    last_non_pixel_features.append(1)
                                else:
                                    last_non_pixel_features.append(0)

    return last_pov, last_non_pixel_features

def get_actions(acts, act_space, action):
    """
    transfer action [1,1,1....] to action_space in env 
    """
    action_spaces = []
    action_spaces_name = []
    idx = 0
    for act in act_space.spaces:
        if act_space[act].__class__.__name__ == 'Discrete':
            action_spaces_name.append(act)
            action_spaces.append(act_space[act].n)
            action[act] = int(acts[idx])
            idx += 1
        elif act_space[act].__class__.__name__ == 'Enum':
            action_spaces_name.append(act)
            action_spaces.append(len(act_space[act].values))
            action[act] = act_space[act].values[int(acts[idx])]
            idx += 1
        elif act == 'camera':
            action_spaces_name.append('pitch')
            action_spaces_name.append('yaw')
            action_spaces.append(36)
            action_spaces.append(36)
            action[act] = [acts[idx] * 10 - 180, acts[idx + 1] * 10 - 180]
            idx += 2

    return action

def get_actions_continuous(acts, act_space, action):
    """
    transfer action [1,1,1....] to action_space in env 
    """
    action_spaces = []
    action_spaces_name = []
    idx = 0
    for act in act_space.spaces:
        if act_space[act].__class__.__name__ == 'Discrete':
            action_spaces_name.append(act)
            action_spaces.append(act_space[act].n)
            action[act] = int(acts[idx])
            idx += 1
        elif act_space[act].__class__.__name__ == 'Enum':
            action_spaces_name.append(act)
            action_spaces.append(len(act_space[act].values))
            action[act] = act_space[act].values[int(acts[idx])]
            idx += 1
        elif act == 'camera':
            action_spaces_name.append('pitch')
            action_spaces_name.append('yaw')
            action_spaces.append(1)
            action_spaces.append(1)
            action[act] = [acts[idx], acts[idx + 1]]
            idx += 2

    return action

def camera_transform(x):
    x = x + 180
    if (x >= 350):
        y = int(x / 10)
    else:
        y = round(x / 10)
    return y

def transfer_actions(action, act_space):
    """
    transfer action_space in env to action [1,1,1....]
    """
    #print(action)
    action_spaces = []
    res = []
    for act in act_space.spaces:
        if act_space[act].__class__.__name__ == 'Discrete':
            action_spaces.append(act_space[act].n)
            res.append(action[act])
        elif act_space[act].__class__.__name__ == 'Enum':
            action_spaces.append(len(act_space[act].values))
            res.append(action[act])
        elif act == 'camera':
            res.append(camera_transform(action[act][0]))
            res.append(camera_transform(action[act][1]))
            action_spaces.append(36)
            action_spaces.append(36)

    return res

def log(j, ep, ep_rewards, best_mean_episode_reward):
    Save_Reward_Every_N_EPs = 10
    if len(ep_rewards) > 0:
        mean_episode_reward = np.mean(ep_rewards[-20:])
        best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
    
    print("---- Ep %d -------" %ep)
    print("num_param_updates:%d" % j)
    print("ep reward: %f" % ep_rewards[-1])
    print("mean reward (20 episodes) %f" % mean_episode_reward)
    print("best mean reward %f" % best_mean_episode_reward)
    print("---------------------------------------")

    #=================== TensorBoard logging =============#
    if ep % Save_Reward_Every_N_EPs == 0:

        data = {"total_rewards":ep_rewards}

        if not os.path.exists("perform_records"):
            os.makedirs("perform_records")

        save_path = f"perform_records/ep{ep}.pkl"
        f = open(save_path,"wb")
        pickle.dump(data,f)
        f.close()

    return mean_episode_reward


