import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from AC import Actor, Critic
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG_EXPLORER(object):
	def __init__(self,state_dim,action_dim,max_action,min_action,discount=0.99,tau=1e-4):

		self.actor = Actor(state_dim, action_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=1e-3)

		# 外部報酬のCritic
		self.critic_ext = Critic(state_dim, action_dim).to(device)
		self.critic_ext_target = copy.deepcopy(self.critic_ext)
		self.critic_ext_optimizer = torch.optim.Adam(self.critic_ext.parameters(),lr=1e-3)

		# 内部報酬のCritic
		self.critic_int = Critic(state_dim, action_dim).to(device)
		self.critic_int_target = copy.deepcopy(self.critic_int)
		self.critic_int_optimizer = torch.optim.Adam(self.critic_int.parameters(),lr=1e-3)

		self.discount = discount
		self.tau = tau

		self.max_p = torch.FloatTensor(max_action).to(device)
		self.min_p = torch.FloatTensor(min_action).to(device)
		self.rng = (self.max_p - self.min_p).detach()
	
	def invert_gradient(self,delta_a,current_a):
		index = delta_a>0
		delta_a[index] *=  (index.float() * (self.max_p - current_a)/self.rng)[index]
		delta_a[~index] *= ((~index).float() * (current_a- self.min_p)/self.rng)[~index]
		return delta_a	

	def select_action(self,state):
		state = torch.FloatTensor(state.reshape(1,-1)).to(device)
		p = self.actor(state)
		np_max = self.max_p.cpu().data.numpy()
		np_min = self.min_p.cpu().data.numpy()
		return np.clip(p.cpu().data.numpy().flatten(),np_min,np_max)


	def train(self,replay_buffer, batch_size=64):
		# TODO: actionにパラメータ入ってる？ → MP-DQNの実装と違って離散とパラメータが一気位に同じ層として出力される
		state,action, next_state, reward, ex_reward, n_step, ex_n_step, not_done = replay_buffer.sample(batch_size)

		# 外部報酬のQ値を出す
		qvals_ext = self.critic_ext_target.forward(next_state,self.actor_target(next_state))
		qvals_ext = ex_reward + (not_done * self.discount * qvals_ext).detach()
		current_qvals_ext = self.critic_ext.forward(state, action)

		# 内部報酬のQ値を出す
		qvals_int = self.critic_int_target.forward(next_state,self.actor_target(next_state))
		qvals_int = ex_reward + (not_done * self.discount * qvals_int).detach()
		current_qvals_int = self.critic_int.forward(state, action)

		beta = 0.2
		# TODO: on_policy_target = n_step+ex_n_stepなのでは? → 大幅に改善した
		# 外部報酬のcritic損失と更新
		mixed_q_ext = beta*(n_step) + (1-beta)*qvals_ext
		critic_ext_loss = F.mse_loss(current_qvals_ext, mixed_q_ext)
		self.critic_ext_optimizer.zero_grad()
		critic_ext_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic_ext.parameters(), 10)
		self.critic_ext_optimizer.step()

		# 内部報酬のcritic損失と更新
		mixed_q_int = beta*(ex_n_step) + (1-beta)*qvals_int
		critic_int_loss = F.mse_loss(current_qvals_int, mixed_q_int)
		self.critic_int_optimizer.zero_grad()
		critic_int_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic_int.parameters(), 10)
		self.critic_int_optimizer.step()

		# 行動を取得
		current_a = Variable(self.actor(state))
		current_a.requires_grad = True

		# Q値の結合 https://qiita.com/pocokhc/items/8684c6c96d3d2963e284?utm_campaign=popular_items&utm_medium=feed&utm_source=popular_items
		rescaling_beta = 0.1
		qvals = rescaling(rescaling_inverse(self.critic_ext(state, current_a).mean()) + rescaling_beta*rescaling_inverse(self.critic_int(state, current_a).mean()))

		# TODO: grad_fnが消える -> 動作するようにはなった。rescalingをtorchの関数をしようすることで勾配情報消えずに計算できる。
		# TODO: grad_fnが消える問題は解決したが、mean()の位置があってるか？
		# TODO: Deep Deterministic Policy Gradient でactorとcriticの実装を参考にする。
		actor_loss = qvals

		self.critic_ext.zero_grad()
		self.critic_int.zero_grad()
		actor_loss.backward()
		delta_a = copy.deepcopy(current_a.grad.data)
		delta_a = self.invert_gradient(delta_a,current_a)
		current_a = self.actor(state)
		out = -torch.mul(delta_a,current_a)
		self.actor.zero_grad()
		out.backward(torch.ones(out.shape).to(device))
		torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
		self.actor_optimizer.step()
		for param, target_param in zip(self.critic_ext.parameters(), self.critic_ext_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.critic_int.parameters(), self.critic_int_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return [current_qvals_ext.mean().item(), mixed_q_ext.mean().item(), critic_ext_loss.item(), current_qvals_int.mean().item(), mixed_q_int.mean().item(), critic_int_loss.mean().item(),qvals.item()]


	def save(self, filename):
		torch.save(self.critic_ext.state_dict(), filename + "_critic_ext")
		torch.save(self.critic_ext_optimizer.state_dict(), filename + "_critic_ext_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic_ext.load_state_dict(torch.load(filename + "_critic"))
		self.critic_ext_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_ext_target = copy.deepcopy(self.critic_ext)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)


def rescaling(x, epsilon=0.001):
	if x == 0:
		return 0
	n = torch.sqrt(torch.abs(x)+1) - 1
	return torch.sign(x)*n + epsilon*x


def rescaling_inverse(x, epsilon=0.001):
	if x == 0:
		return 0
	n = torch.sqrt(1 + 4*epsilon*(torch.abs(x)+1+epsilon)) - 1
	return torch.sign(x)*(n/(2*epsilon) - 1)