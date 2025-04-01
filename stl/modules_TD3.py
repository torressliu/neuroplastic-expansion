import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MLPActor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
		super().__init__()
		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, action_dim)
		
		self.max_action = max_action
		self.each_nums = [state_dim, hidden_dim, hidden_dim]

	def forward(self, state) -> torch.Tensor:
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))

	def forward_with_FAU(self, state, current_topo):
		with torch.no_grad():
			h_policy = F.relu(self.l1(state))
			total_1 = h_policy.numel()

			h_policy_2 = F.relu(self.l2(h_policy))
			total_2 = h_policy_2.numel()

			action = self.max_action * torch.tanh(self.l3(h_policy_2))
			act_1 = (h_policy > 0).sum().item()
			act_2 = (h_policy_2 > 0).sum().item()

		rate = (act_1 + act_2) / (total_1 + total_2)
		return rate
	def score_drop(self, state):
		score_drops = []
		with torch.no_grad():
			h_policy = F.relu(self.l1(state))
			score_drops.append(h_policy)

			h_policy_2 = F.relu(self.l2(h_policy))
			score_drops.append(h_policy_2)

			action = self.l3(h_policy_2)
			score_drops.append(action)
		return score_drops, self.each_nums
class MLPCritic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256):
		super().__init__()

		# Q1 architecture
		# scale up to any size here manually
		self.l = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l1 = nn.Linear(hidden_dim, hidden_dim)
		self.l2 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim)
		)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2 architecture
		# scale up to any size here manually
		self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim)
		)
		self.l7 = nn.Linear(hidden_dim, 1)
		self.each_nums = [state_dim, hidden_dim, hidden_dim, hidden_dim, state_dim, hidden_dim, hidden_dim, hidden_dim]


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l(sa))
		q1 = F.relu(self.l1(q1))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = F.relu(self.l6(q2))
		q2 = self.l7(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l(sa))
		q1 = F.relu(self.l1(q1))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

	def forward_with_FAU(self, state, action, current_topo):
		with torch.no_grad():
			h_action = torch.cat([state, action], 1)

			h_Q1 = F.relu(self.l(h_action))
			total_1 = h_Q1.numel()

			h_Q12 = F.relu(self.l1(h_Q1))
			total_2 = h_Q12.numel()

			h_Q13 = F.relu(self.l2(h_Q12))
			total_3 = h_Q13.numel()
			h_Q14 = self.l3(h_Q13)

			h_Q2 = F.relu(self.l4(h_action))
			total_5 = h_Q2.numel()

			h_Q22 = F.relu(self.l5(h_Q2))
			total_6 = h_Q22.numel()

			h_Q23 = F.relu(self.l6(h_Q22))
			total_7 = h_Q23.numel()

			q2 = self.l7(h_Q23)
			act_1 = (h_Q1 > 0).sum().item()
			act_2 = (h_Q12 > 0).sum().item()
			act_3 = (h_Q13 > 0).sum().item()
			act_4 = (h_Q2 > 0).sum().item()
			act_5 = (h_Q22 > 0).sum().item()
			act_6 = (h_Q23 > 0).sum().item()

		rate = (act_1 + act_2 + act_3 + act_4 + act_5 + act_6) / (total_1 + total_2 + total_3 + total_5 + total_6 + total_7)		

		return rate
	def score_drop(self, state, action):
		score_drops = []
		with torch.no_grad():
			h_action = torch.cat([state, action], 1)

			h_Q1 = F.relu(self.l(h_action))
			score_drops.append(h_Q1)

			h_Q12 = F.relu(self.l1(h_Q1))
			score_drops.append(h_Q12)

			h_Q13 = F.relu(self.l2(h_Q12))
			score_drops.append(h_Q13)

			q1 = self.l3(h_Q12)
			score_drops.append(q1)
			# act_3 = (h_Q1 > 0).sum().item()
			# total_3 = q1.numel()

			h_Q2 = F.relu(self.l4(h_action))
			score_drops.append(h_Q2)

			h_Q22 = F.relu(self.l5(h_Q2))
			score_drops.append(h_Q22)
			q2 = self.l6(h_Q22)
			score_drops.append(q2)
			q22 = self.l7(q2)
			score_drops.append(q22)
		return score_drops, self.each_nums