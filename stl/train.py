# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
sys.path.append(r'../')
import numpy as np
import torch
import gym
#from dm_control import suite
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
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", default='exptest')                              # Experiment name
    parser.add_argument("--env", default='HalfCheetah')                             # Environment
    parser.add_argument("--seed", default=1, type=int)                              # Seed
    parser.add_argument("--start_timesteps", default=25e3, type=int)                # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)                       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=6e6, type=int)                   # Max time steps to run environment
    parser.add_argument("--reset_step", default=2e5, type=int)                      # Out of use
    parser.add_argument("--rej_step", default=1e6, type=int)                        # Out of use
    parser.add_argument("--expl_noise", default=0.1)                                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)                      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                                 # Discount factor
    parser.add_argument("--tau", default=0.005)                                     # Target network update rate
    parser.add_argument("--Tamp", default=0.9)                                      # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)                              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)                       # Frequency of delayed policy updates
    parser.add_argument("--hidden_dim", default=256, type=int)                      # Num of hidden neurons in each layer
    parser.add_argument("--static_actor", action='store_true', default=False)       # Fix the topology of actor
    parser.add_argument("--static_critic", action='store_true', default=False)      # Fix the topology of critic
    parser.add_argument("--actor_sparsity", default=0.25, type=float)              # Final sparsity of actor
    parser.add_argument("--critic_sparsity",default=0.25, type=float)              # Final sparsity of critic
    parser.add_argument("--initial_stl_sparsity", default=0.8, type=float)         # Initial sparsity for STL mode
    parser.add_argument("--delta", default=100, type=int)                        # Mask update interval
    parser.add_argument("--zeta", default=0.5, type=float)                         # Initial mask update ratio
    parser.add_argument("--awaken", default=0.4, type=float)                       # Initial recall/auto batch start point
    parser.add_argument("--recall", action='store_true', default=False)            # recall history samples for activate neurons
    parser.add_argument("--auto_batch", action='store_true', default=True)         # for stable study with dynamic topology
    parser.add_argument("--random_grow", action='store_true', default=False)       # random grow scheme
    parser.add_argument("--uni", action='store_true', default=False)              # uniformly grow scheme
    parser.add_argument("--elastic", action='store_true', default=False)          # EMA update
    parser.add_argument("--nstep", default=1, type=int)                          # N-step returns
    parser.add_argument("--delay_nstep", default=0, type=int)                    # Delay of using N-step
    parser.add_argument("--buffer_max_size", default=int(1e6), type=int)         # Upper bound of buffer capacity 
    parser.add_argument("--buffer_min_size", default=int(1e5),type=int)          # Lower bound of buffer capacity
    parser.add_argument("--use_dynamic_buffer", action='store_true', default=False) # Use dynamic buffer
    parser.add_argument("--stl_actor", action='store_true', default=False)       # Enable STL mode for actor
    parser.add_argument("--stl_critic", action='store_true', default=False)      # Enable STL mode for critic
    parser.add_argument("--buffer_threshold", default=0.2, type=float)           # Threshold for policy distance
    parser.add_argument("--buffer_adjustment_interval", default=int(1e4),type=int) # Buffer check interval
    parser.add_argument("--init_method", default='lecun', type=str, choices=['zero', 'lecun']) # Weight initialization method, lecun is recommended for CNN based VRLs
    parser.add_argument("--grad_accumulation_n", default=1, type=int)           # Number of steps to accumulate gradients, multi-step can make the evaluation more stable for complex tasks, especially for VRLs
    parser.add_argument("--use_simple_metric", action='store_true', default=False)  # Use fast dormant weight pruning, instead of ReDo-style pruning
    parser.add_argument("--complex_prune", action='store_true', default=False)  # Warning: this will make your progress run slowly. When using ReDo-style pruning, we suggest to set complex_prune to True for clean the gradient of dormant neurons following the Dormant neuron phenomenon github repo
    args = parser.parse_args()
    args.T_end = (args.max_timesteps - args.start_timesteps)
    the_dir = 'results' 
    root_dir = './'+the_dir+'/'+args.exp_id+'_'+args.env
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
    print(f"Env: {args.env}, Seed: {args.seed}")
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
    
    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    args.state_dim = state_dim
    args.action_dim = action_dim
    args.max_action = max_action

    args.policy_noise *= max_action
    args.noise_clip *= max_action

    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    writer = SummaryWriter(tensorboard_dir)
    policy = TD3(args, writer)

    # Load saved models and DST scheduler state if they exist
    if os.path.exists(model_dir+'actor') and os.path.exists(model_dir+'critic'):
        print("Loading saved models...")
        policy.actor.load_state_dict(torch.load(model_dir+'actor'))
        policy.critic.load_state_dict(torch.load(model_dir+'critic'))
        if args.actor_sparsity > 0 and os.path.exists(model_dir+'actor_pruner'):
            policy.actor_pruner.load_state_dict(torch.load(model_dir+'actor_pruner'))
            policy.actor_pruner.backward_masks = torch.load(model_dir+'actor_masks')
        if args.critic_sparsity > 0 and os.path.exists(model_dir+'critic_pruner'):
            policy.critic_pruner.load_state_dict(torch.load(model_dir+'critic_pruner'))
            policy.critic_pruner.backward_masks = torch.load(model_dir+'critic_masks')
        print("Models loaded successfully")

    if args.actor_sparsity > 0:
        print("Training a sparse actor network")
        show_sparsity(policy.actor.state_dict())
    if args.critic_sparsity > 0:
        print("Training a sparse critic network")
        show_sparsity(policy.critic.state_dict())

 
    replay_buffer = ReplayBuffer(state_dim, action_dim, args.buffer_max_size)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]
    recent_eval = deque(maxlen=20)
    best_eval = np.mean(evaluations)

    state, done = env.reset(), False
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
            action = env.action_space.sample()
            action_mean = action
        else:
            action_mean = policy.select_action(np.array(state))
            action = (
                action_mean
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer

        replay_buffer.add(state, action, next_state, reward, done_bool, action_mean, episode_timesteps >= env._max_episode_steps)
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


        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            eval_result = eval_policy(policy, args.env, args.seed)
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
                # Save STL scheduler state if using sparse networks
                if args.actor_sparsity > 0:
                    torch.save(policy.actor_pruner.state_dict, model_dir+'actor_pruner')
                    torch.save(policy.actor_pruner.backward_masks, model_dir+'actor_masks')
                if args.critic_sparsity > 0:
                    torch.save(policy.critic_pruner.state_dict, model_dir+'critic_pruner')
                    torch.save(policy.critic_pruner.backward_masks, model_dir+'critic_masks')
    writer.close()

if __name__ == "__main__":
    main()

