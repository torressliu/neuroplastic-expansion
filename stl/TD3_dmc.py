import copy
import numpy as np
import torch
import torch.nn.functional as F
from RLx2_TD3.modules_TD3_dmc import MLPActor, MLPCritic
from NE.STL_Scheduler import DST_Scheduler
#from DST.DST_Scheduler_v2 import DST_Scheduler_v2
from NE.utils import ReplayBuffer, get_W, show_sparsity
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3(object):
	def __init__(
		self,
		args,
		writer
	):
		# RL hyperparameters
		self.max_action = args.max_action
		self.discount = args.discount
		self.tau = args.tau
		self.policy_noise = args.policy_noise
		self.noise_clip = args.noise_clip
		self.policy_freq = args.policy_freq

		# Neural networks
		self.actor = MLPActor(args.state_dim, args.action_dim, args.max_action, args.hidden_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = MLPCritic(args.state_dim, args.action_dim, args.hidden_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.sparse_actor = (args.actor_sparsity > 0)
		self.sparse_critic = (args.critic_sparsity > 0)
		self.critic_sparsity = args.critic_sparsity
		self.actor_sparsity = args.actor_sparsity

		self.total_it = 0
		self.nstep = args.nstep
		self.delay_nstep = args.delay_nstep
		self.writer:SummaryWriter = writer
		self.tb_interval = int(args.T_end/1000)

		if self.sparse_actor: # Sparsify the actor at initialization
			self.actor_pruner = DST_Scheduler(model=self.actor, optimizer=self.actor_optimizer, sparsity=args.actor_sparsity, T_end=int(args.T_end/self.policy_freq), static_topo=args.static_actor, zeta=args.zeta, delta=args.delta, random_grow=args.random_grow,stl=args.stl_actor)
			self.targer_actor_W, _ = get_W(self.actor_target)
		else:
			self.actor_pruner = lambda: True
		if self.sparse_critic: # Sparsify the critic at initialization
			self.critic_pruner = DST_Scheduler(model=self.critic, optimizer=self.critic_optimizer, sparsity=args.critic_sparsity, T_end=args.T_end, static_topo=args.static_critic, zeta=args.zeta, delta=args.delta, random_grow=args.random_grow,stl=args.stl_critic)
			self.targer_critic_W, _ = get_W(self.critic_target)
		else:
			self.critic_pruner = lambda: True

	def select_action(self, state) -> np.ndarray:
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer:ReplayBuffer, batch_size=256):
		self.total_it += 1

		# Delay to use multi-step TD target
		current_nstep = self.nstep if self.total_it >= self.delay_nstep else 1

		if self.total_it % self.tb_interval == 0: self.writer.add_scalar('current_nstep', current_nstep, self.total_it)
		
		state, action, next_state, reward, not_done, _, reset_flag = replay_buffer.sample(batch_size, current_nstep)
		
		with torch.no_grad():
			ac_fau = self.actor.forward_with_FAU(state[:, 0], self.actor_pruner.backward_masks)
			cr_fau = self.critic.forward_with_FAU(state[:, 0], action[:, 0], self.critic_pruner.backward_masks)
			noise = (
				torch.randn_like(action[:,0]) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			accum_reward = torch.zeros(reward[:,0].shape).to(device)
			have_not_done = torch.ones(not_done[:,0].shape).to(device)
			have_not_reset = torch.ones(not_done[:,0].shape).to(device)
			modified_n = torch.zeros(not_done[:,0].shape).to(device)
			for k in range(current_nstep):
				accum_reward += have_not_reset*have_not_done*self.discount**k*reward[:,k]
				have_not_done *= torch.maximum(not_done[:,k], 1-have_not_reset)
				if k == current_nstep - 1:
					break
				have_not_reset *= (1-reset_flag[:,k])
				modified_n += have_not_reset
			modified_n = modified_n.type(torch.long)
			nstep_next_state = next_state[np.arange(batch_size), modified_n[:,0]]
			next_action = (
				self.actor_target(nstep_next_state) + noise
			).clamp(-self.max_action, self.max_action)
			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(nstep_next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			if current_nstep == 1:
				target_Q = accum_reward.reshape(target_Q.shape) + have_not_done.reshape(target_Q.shape) * self.discount * target_Q
			else:
				target_Q = accum_reward.reshape(target_Q.shape) + have_not_done.reshape(target_Q.shape) * self.discount**(modified_n + 1) * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state[:,0], action[:,0])
		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		if self.total_it % self.tb_interval == 0: self.writer.add_scalar('critic_loss',critic_loss.item(), self.total_it)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		end_grow = show_sparsity(self.critic.state_dict(), to_print=False) <= self.critic_sparsity
		if self.critic_pruner(end_grow):
			self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			actor_loss:torch.Tensor = -self.critic.Q1(state[:,0], self.actor(state[:,0])).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			end_grow = show_sparsity(self.actor.state_dict(), to_print=False) <= self.actor_sparsity
			if self.actor_pruner(end_grow):
				self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			# Note we need to also sparsify the target network. Here we simply use the same mask.
			if self.sparse_critic:
				for w, mask in zip(self.targer_critic_W, self.critic_pruner.backward_masks):
					w.data *= mask

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			if self.sparse_actor:
				for w, mask in zip(self.targer_actor_W, self.actor_pruner.backward_masks):
					w.data *= mask
		return ac_fau,cr_fau
		

		
