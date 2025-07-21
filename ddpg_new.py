from utils30_new import Actor, Q_Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy
from torch.nn.utils import clip_grad_norm_

class DDPG_agent():
	def __init__(self, **kwargs):
		   # Actor
	    self.actor = Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
	    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
	    self.actor_scheduler  = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.995)
	    self.actor_target = copy.deepcopy(self.actor)
	
	    # Critic
	    self.q_critic = Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
	    self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
	    self.critic_scheduler    = torch.optim.lr_scheduler.ExponentialLR(self.q_critic_optimizer, gamma=0.995)
	    self.q_critic_target = copy.deepcopy(self.q_critic)
	
	    # Replay
	    self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(500e3), dvc=self.dvc)
	    self.noise_proc    = OUNoise(self.action_dim)
	
	def select_action(self, state, deterministic=False):
	    state = torch.FloatTensor(np.array(state, dtype=np.float32)[None]).to(self.dvc)
	    a = self.actor(state).cpu().numpy()[0]
	    if deterministic:
	        return a
	    noise = self.noise_proc()
	    return np.clip(a + noise, 0, self.max_action)
	
	def train(self):
	    # Sample
	    s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
	    with torch.no_grad():
	        # Target smoothing
	        target_a = self.actor_target(s_next)
	        noise = (torch.randn_like(target_a) * 0.2).clamp(-0.5, 0.5)
	        target_a = (target_a + noise).clamp(0, self.max_action)
	        target_Q = self.q_critic_target(s_next, target_a)
	        target_Q = r + (~dw) * self.gamma * target_Q
	    # Critic update
	    current_Q = self.q_critic(s, a)
	    q_loss = F.mse_loss(current_Q, target_Q)
	    self.q_critic_optimizer.zero_grad()
	    q_loss.backward()
	    clip_grad_norm_(self.q_critic.parameters(), max_norm=0.5)
	    self.q_critic_optimizer.step()
	    # Actor update
	    for p in self.q_critic.parameters(): p.requires_grad_(False)
	    a_loss = -self.q_critic(s, self.actor(s)).mean()
	    self.actor_optimizer.zero_grad()
	    a_loss.backward()
	    clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
	    self.actor_optimizer.step()
	    for p in self.q_critic.parameters(): p.requires_grad_(True)
	    # Soft update
	    with torch.no_grad():
	        for p, tp in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
	            tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)
	        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
	            tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)
	    # Schedulers
	    self.actor_scheduler.step()
	    self.critic_scheduler.step()
	    return a_loss.item(), q_loss.item()
	
	def save(self, EnvName, timestep):
	    torch.save(self.actor.state_dict(), f"./model/{EnvName}_actor{timestep}.pth")
	    torch.save(self.q_critic.state_dict(), f"./model/{EnvName}_critic{timestep}.pth")
	
	def load(self, EnvName, timestep):
	    self.actor.load_state_dict(torch.load(f"./model/{EnvName}_actor{timestep}.pth", map_location=self.dvc))
	    self.q_critic.load_state_dict(torch.load(f"./model/{EnvName}_critic{timestep}.pth", map_location=self.dvc))
class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size, dvc):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		#每次只放入一个时刻的数据
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = torch.from_numpy(a).to(self.dvc) # Note that a is numpy.array
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		#print(ind)
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
