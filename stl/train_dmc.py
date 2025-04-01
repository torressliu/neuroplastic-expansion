# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
sys.path.append(r'../')
import numpy as np
import torch
import gym
from dm_control import suite
import argparse
import os

from NE.utils import ReplayBuffer, show_sparsity
from TD3 import TD3
from torch.utils.tensorboard import SummaryWriter
import json
import torch.nn.functional as F
from collections import deque
import copy


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, task_name, seed=1, eval_episodes=10):
    eval_env = suite.load(env_name, task_name)
    #eval_env.seed(1 + 100)
    
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state = obs2state(state.observation)
        while not done:
            action = policy.select_action(np.array(state))
            timestep = eval_env.step(action)
            if timestep.reward != None:
                avg_reward += timestep.reward
            done = timestep.last()
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward
def obs2state(observation):
    """Converts observation dictionary to state tensor"""
    l1 = [val.tolist() for val in list(observation.values())]
    l2 = []
    for sublist in l1:
        try:
            l2.extend(sublist)
        except:
            l2.append(sublist)
    return torch.FloatTensor(l2).view(1, -1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", default='exptest')                              # Experiment name
    parser.add_argument("--env", default='dog')                          # Environment
    parser.add_argument("--task", default='run')
    parser.add_argument("--frameskip", default= 2, type=int)
    parser.add_argument("--max_episode_steps", default=1000, type=int)
    parser.add_argument("--seed", default=1, type=int)                              # Seed
    parser.add_argument("--start_timesteps", default=25e3, type=int)                # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)                       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=6e6, type=int)                   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)                      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                                 # Discount factor
    parser.add_argument("--tau", default=0.005)                                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)                              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)                       # Frequency of delayed policy updates
    parser.add_argument("--hidden_dim", default=256, type=int)                      # Num of hidden neurons in each layer
    parser.add_argument("--awaken", default=0.35, type=float)  # Initial recall/auto batch start point
    parser.add_argument("--Tamp", default=0.9)  # Target network update rate
    parser.add_argument("--auto_batch", action='store_true', default=False)
    parser.add_argument("--recall", action='store_true', default=False)  # Use random grow scheme
    parser.add_argument("--static_actor", action='store_true', default=False)       # Fix the topology of actor
    parser.add_argument("--static_critic", action='store_true', default=False)      # Fix the topology of critic
    parser.add_argument("--uniform", action='store_true', default=False)  # grow mode
    parser.add_argument("--actor_sparsity", default=0., type=float)                 # Sparsity of actor
    parser.add_argument("--critic_sparsity",default=0., type=float)                 # Sparsity of critic
    parser.add_argument("--delta", default=10000, type=int)                         # Mask update interval
    parser.add_argument("--zeta", default=0.5, type=float)                          # Initial mask update ratio
    parser.add_argument("--random_grow", action='store_true', default=False)        # Use random grow scheme
    parser.add_argument("--nstep", default=1, type=int)                             # N-step
    parser.add_argument("--delay_nstep", default=0, type=int)                       # Delay of using N-step
    parser.add_argument("--buffer_max_size", default=int(1e6), type=int)            # Upper bound of buffer capacity 
    parser.add_argument("--buffer_min_size", default=int(1e5),type=int)             # Lower bound of buffer capacity
    parser.add_argument("--use_dynamic_buffer", action='store_true', default=False) # Use dynamic buffer
    parser.add_argument("--stl_actor", action='store_true', default=False)  # Transfer to small to large mode
    parser.add_argument("--stl_critic", action='store_true', default=False)  # Transfer to small to large mode
    parser.add_argument("--buffer_threshold", default=0.2, type=float)              # Threshold of policy distance 
    parser.add_argument("--buffer_adjustment_interval", default=int(1e4),type=int)  # How often (time steps) we check the buffer

    args = parser.parse_args()
    max_episode_steps = args.max_episode_steps
    args.T_end = (args.max_timesteps - args.start_timesteps)
    the_dir = 'results_TD3' 
    root_dir = './'+the_dir+'/'+args.exp_id+'_'+args.env+'_'+args.task
    argsDict = copy.deepcopy(args.__dict__)
    del argsDict['seed']
    config_json=json.dumps(argsDict, indent=4)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    file_json=open(root_dir+'/config.json','w')
    file_json.write(config_json)
    file_json.close()
    if not os.path.exists("./"+the_dir):
        os.makedirs("./"+the_dir)

    print("---------------------------------------")
    print(f"Policy: TD3, Env: {args.env}_{args.task}, Seed: {args.seed}")
    print("---------------------------------------")

    exp_dir = root_dir+'/'+str(args.seed)
    tensorboard_dir = exp_dir+'/tensorboard/'
    model_dir = exp_dir+'/model/'

    
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.set_num_threads(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = suite.load(args.env, args.task)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #print("sss",env.reset().observation)
    state_dim = obs2state(env.reset().observation).size()[1]
    action_dim = env.action_spec().shape[0]
    spec = env.action_spec()
    max_action = spec.maximum[0]

    args.state_dim = state_dim
    args.action_dim = action_dim
    args.max_action = max_action

    args.policy_noise *= max_action
    args.noise_clip *= max_action

    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    writer = SummaryWriter(tensorboard_dir)
    policy = TD3(args, writer)

    if args.actor_sparsity > 0:
        print("Training a sparse actor network")
        show_sparsity(policy.actor.state_dict())
    if args.critic_sparsity > 0:
        print("Training a sparse critic network")
        show_sparsity(policy.critic.state_dict())

 
    replay_buffer = ReplayBuffer(state_dim, action_dim, args.buffer_max_size)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.task)]
    recent_eval = deque(maxlen=20)
    best_eval = np.mean(evaluations)

    state, done_bool = env.reset(), False
    state = obs2state(state.observation)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    eval_num=0

    torch.save(policy.actor.state_dict(), model_dir+'actor0')
    torch.save(policy.critic.state_dict(),model_dir+'critic0')

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
            #action = env.action_space.sample()
            action_mean = action
        else:
            action_mean = policy.select_action(np.array(state))
            action = (
                action_mean
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        timestep = env.step(action)
        next_state = obs2state(timestep.observation)#timestep.observation
        reward = timestep.reward
        done_bool = timestep.last() #float(done) if episode_timesteps < env._max_episode_steps else 0
        #if done_bool:
        #    reward=0.

        # Store data in replay buffer

        replay_buffer.add(state, action, next_state, reward, done_bool, action_mean, episode_timesteps >= max_episode_steps)
        if args.use_dynamic_buffer and (t+1) % args.buffer_adjustment_interval == 0:
            if replay_buffer.size == replay_buffer.max_size: 
                ind = (replay_buffer.ptr + np.arange(8*args.batch_size)) % replay_buffer.max_size
            else:
                ind = (replay_buffer.left_ptr + np.arange(8*args.batch_size)) % replay_buffer.max_size
            batch_state = torch.FloatTensor(replay_buffer.state[ind]).to(device)
            batch_action_mean = torch.FloatTensor(replay_buffer.action_mean[ind]/max_action).to(device)
            with torch.no_grad():
                current_action = policy.actor(batch_state)/max_action
                distance=F.mse_loss(current_action, batch_action_mean)/2
            writer.add_scalar('buffer_distance',distance, t)
            if distance > args.buffer_threshold and replay_buffer.size > args.buffer_min_size:
                replay_buffer.shrink()
            writer.add_scalar('buffer_size', replay_buffer.size, t)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            ac_fau, cr_fau = policy.train(replay_buffer=replay_buffer, batch_size=args.batch_size)
            if (t + 1) % args.eval_freq == 0:
                writer.add_scalar('actor_FAU', ac_fau, t)
                writer.add_scalar('critic_FAU', cr_fau, t)


        if done_bool:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done_bool = env.reset(), False
            state = obs2state(state.observation)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            eval_result = eval_policy(policy, args.env, args.task, args.seed)
            evaluations.append(eval_result)
            recent_eval.append(eval_result)
            ac_sparsity=show_sparsity(policy.actor.state_dict())
            cr_sparsity = show_sparsity(policy.critic.state_dict())
            writer.add_scalar('actor_sparsity', ac_sparsity, t)
            writer.add_scalar('critic_sparsity', cr_sparsity, t)
            writer.add_scalar('reward',eval_result, eval_num)
            eval_num+=1
            if np.mean(recent_eval) > best_eval:
                best_eval = np.mean(recent_eval)
                torch.save(policy.actor.state_dict(), model_dir+'actor')
                torch.save(policy.critic.state_dict(),model_dir+'critic')
                if args.actor_sparsity > 0: torch.save(policy.actor_pruner.backward_masks, model_dir+'actor_masks')
                if args.critic_sparsity > 0: torch.save(policy.critic_pruner.backward_masks, model_dir+'critic_masks')
    writer.close()

if __name__ == "__main__":
    main()

