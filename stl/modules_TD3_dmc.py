import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
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
class MLPActor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
		super().__init__()
		self.norm0 = nn.BatchNorm1d(state_dim)
		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.bn1 = nn.BatchNorm1d(hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.bn2 = nn.BatchNorm1d(hidden_dim)
		self.l3 = nn.Linear(hidden_dim, action_dim)
		
		self.max_action = max_action
		self.each_nums = [state_dim, hidden_dim, hidden_dim]

	def forward(self, state) -> torch.Tensor:
		#state = self.norm0(state)
		a = F.relu(self.l1(state))
		#a = self.bn1(a)
		a = F.relu(self.l2(a))
		#a = self.bn2(a)
		return self.max_action * torch.tanh(self.l3(a))

	def forward_with_FAU(self, state, current_topo):
		with torch.no_grad():
			h_policy = F.relu(self.l1(state))
			#h_policy = self.bn1(h_policy)
			total_1 = h_policy.numel()

			h_policy_2 = F.relu(self.l2(h_policy))
			#h_policy_2 = self.bn2(h_policy_2)
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
		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		#self.bn1 = nn.BatchNorm1d(hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
		#self.bn2 = nn.BatchNorm1d(hidden_dim)
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)
		self.each_nums = [state_dim, hidden_dim, hidden_dim, state_dim, hidden_dim, hidden_dim]


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		#q1 = self.bn1(q1)
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		#q2 = self.bn2(q2)
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = F.relu(self.l1(sa))
		#q1 = self.bn1(q1)
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

	def forward_with_FAU(self, state, action, current_topo):
		with torch.no_grad():
			h_action = torch.cat([state, action], 1)

			h_Q1 = F.relu(self.l1(h_action))
			#h_Q1 = self.bn1(h_Q1)
			total_1 = h_Q1.numel()

			h_Q12 = F.relu(self.l2(h_Q1))
			total_2 = h_Q12.numel()

			q1 = self.l3(h_Q12)
			#act_3 = (h_Q1 > 0).sum().item()
			#total_3 = q1.numel()

			h_Q2 = F.relu(self.l4(h_action))
			#h_Q2 = self.bn2(h_Q2)
			total_3 = h_Q2.numel()

			h_Q22 = F.relu(self.l5(h_Q2))
			total_4 = h_Q22.numel()
			q2 = self.l6(h_Q22)
			act_1 = (h_Q1 > 0).sum().item()
			act_2 = (h_Q12 > 0).sum().item()
			act_3 = (h_Q2 > 0).sum().item()
			act_4 = (h_Q22 > 0).sum().item()

		rate = (act_1 + act_2 + act_3 + act_4) / (total_1 + total_2 + total_3 + total_4)

		return rate
	def score_drop(self, state, action):
		score_drops = []
		with torch.no_grad():
			h_action = torch.cat([state, action], 1)

			h_Q1 = F.relu(self.l1(h_action))
			score_drops.append(h_Q1)

			h_Q12 = F.relu(self.l2(h_Q1))
			score_drops.append(h_Q12)

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
		return score_drops, self.each_nums