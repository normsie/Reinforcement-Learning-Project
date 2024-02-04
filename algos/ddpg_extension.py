from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer, PolicyExtension
from .ddpg_agent import DDPGAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
import math
from pathlib import Path

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()


class DDPGExtension(DDPGAgent):
    def __init__(self, config=None):
        super(DDPGExtension, self).__init__(config)
        self.device = self.cfg.device
        self.name = 'ddpg_extension'
        state_dim = self.observation_space_dim
        self.action_dim = self.action_space_dim
        self.max_aciton = self.cfg.max_action
        self.lr = self.cfg.lr
        
        self.pi = PolicyExtension(state_dim, self.action_dim, self.max_action).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=float(self.lr))
        
        self.q1 = Critic(state_dim, self.action_dim).to(self.device)
        self.q2 = Critic(state_dim, self.action_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=float(self.lr))
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=float(self.lr))
        self.alpha = 0.4
        
        self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(self.alpha), dtype=torch.float32))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=float(self.lr))
        
        state_shape = [state_dim,]
        self.target_entropy = -self.action_space_dim
        self.buffer = ReplayBuffer(state_shape, self.action_dim, max_size=int(float(1e6)))
        
        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        self.policy_noise = 0.5
        self.noise_clip = 1.0
        
        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = 7000 # collect 5k random data for better exploration
        self.max_episode_steps=self.cfg.max_episode_steps
        
        
    def update(self,):
        """ After collecting one trajectory, update the pi and q for #transition times: """
        info = {}
        update_iter = self.buffer_ptr - self.buffer_head # update the network once per transition

        if self.buffer_ptr > self.random_transition: # update once we have enough data
            for i in range(update_iter):
                info = self._update()
                # if i%2==0:
                #     self._update_policy()
        
        # update the buffer_head:
        self.buffer_head = self.buffer_ptr
        return info
    # def _update_policy(self,):
    #     batch = self.buffer.sample(self.batch_size, device=self.device)
    #     sampled_action, log_prob = self.pi.sample(batch.state)
    #     q1_actor = self.q1(batch.state, sampled_action)
    #     q2_actor = self.q2(batch.state, sampled_action)
    #     q_actor = torch.min(q1_actor, q2_actor)
    #     actor_loss = (self.alpha*log_prob - q_actor).mean()
        
    #     self.pi_optim.zero_grad()
    #     actor_loss.backward()
    #     self.pi_optim.step()
    #     cu.soft_update_params(self.pi, self.pi_target, self.tau)
    def _update(self,):
        batch = self.buffer.sample(self.batch_size, device=self.device)
        
        q1_current = self.q1(batch.state, batch.action)
        q2_current = self.q2(batch.state, batch.action)
        
        with torch.no_grad():
            noise = (torch.randn_like(batch.action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action, log_prob = self.pi_target.sample(batch.next_state)
            next_action += noise
            q1_tar = self.q1_target(batch.next_state, next_action)
            q2_tar = self.q2_target(batch.next_state, next_action)
            q_tar = torch.min(q1_tar, q2_tar)
            q_tar = batch.reward + self.gamma*batch.not_done * (q_tar - self.alpha*log_prob)
            q_tar = q_tar.detach()
            
        critic_loss1 = F.mse_loss(q1_current, q_tar)
        critic_loss2 = F.mse_loss(q2_current, q_tar)
        
        self.q1_optim.zero_grad()
        critic_loss1.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        critic_loss2.backward()
        self.q2_optim.step()
        
        sampled_action, log_prob = self.pi.sample(batch.state)
        q1_actor = self.q1(batch.state, sampled_action)
        q2_actor = self.q2(batch.state, sampled_action)
        q_actor = torch.min(q1_actor, q2_actor)
        actor_loss = (self.alpha*log_prob - q_actor).mean()
        
        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()
        
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        
        cu.soft_update_params(self.q1, self.q1_target, self.tau)
        cu.soft_update_params(self.q2, self.q2_target, self.tau)
        cu.soft_update_params(self.pi, self.pi_target, self.tau)
        
        return {}
    
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(self.device)

        action, _ = self.pi(x)

        if not evaluation:
            if self.buffer_ptr < self.random_transition: # collect random trajectories for better exploration.
                action = torch.rand(self.action_dim)
            else:
                #expl_noise = 0.1 * self.max_action # the stddev of the expl_noise if not evaluation
                expl_noise = 0.3 * self.max_action
                action = action + expl_noise * torch.randn_like(action)

        return action, {} # just return a positional value
    
    def overlap(self, agent_pos, sand_pos, nosand_pos, radius=10):
        def distance(point1, point2):
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1]-point2[1])**2)
        
        agent_x, agent_y = agent_pos
        sanding_overlap = sum(
            1 for i in range(0, len(sand_pos), 2) if distance(agent_pos, (sand_pos[i], sand_pos[i+1])) < 2*radius
        )
        nosanding_overlap = sum(
            1 for i in range(0, len(nosand_pos), 2) if distance(agent_pos, (nosand_pos[i], nosand_pos[i+1])) < 2*radius
        )
        
        if sanding_overlap > 0 and nosanding_overlap > 0:
            return 0.1*sanding_overlap - 0.15*nosanding_overlap
        elif sanding_overlap > 0:
            return 0.1*sanding_overlap
        elif nosanding_overlap > 0:
            return -0.15*nosanding_overlap
        else:
            return 0
    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        agent_pos = state[:2]
        idx = (len(state) - 2) // 2
        sand_pos = state[2:2+idx]
        nosand_pos = state[idx:]
        noise_reward = self.overlap(agent_pos, sand_pos, nosand_pos)
        reward += noise_reward
        self.buffer.add(state, action, next_state, reward, done)
        
    def train_iteration(self):
        #start = time.perf_counter()
        # Run actual training        
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()
        while not done:
            
            # Sample action from policy
            action, _ = self.get_action(obs) 

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # Store action's outcome (so that the agent can improve its policy)        
            
            done_bool = float(done) if timesteps < self.max_episode_steps else 0 
            self.record(obs, action, next_obs, reward, done_bool)
                
            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            
            if timesteps >= self.max_episode_steps:
                done = True
            # update observation
            obs = next_obs.copy()

        # update the policy after one episode
        #s = time.perf_counter()
        info = self.update()
        #e = time.perf_counter()
        
        # Return stats of training
        info.update({
                    'episode_length': timesteps,
                    'ep_reward': reward_sum,
                    })
        
        end = time.perf_counter()
        return info
        
    def train(self):
        if self.cfg.save_logging:
            L = cu.Logger() # create a simple logger to record stats
        start = time.perf_counter()
        total_step=0
        run_episode_reward=[]
        log_count=0
        
        for ep in range(self.cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = self.train_iteration()
            train_info.update({'episodes': ep})
            total_step+=train_info['episode_length']
            train_info.update({'total_step': total_step})
            run_episode_reward.append(train_info['ep_reward'])
            
            if total_step>self.cfg.log_interval*log_count:
                average_return=sum(run_episode_reward)/len(run_episode_reward)
                if not self.cfg.silent:
                    print(f"Episode {ep} Step {total_step} finished. Average episode return: {average_return}")
                if self.cfg.save_logging:
                    train_info.update({'average_return':average_return})
                    L.log(**train_info)
                run_episode_reward=[]
                log_count+=1

        if self.cfg.save_model:
            self.save_model()
            
        logging_path = str(self.logging_dir)+'/logs'   
        if self.cfg.save_logging:
            L.save(logging_path, self.seed)
        self.env.close()

        end = time.perf_counter()
        train_time = (end-start)/60
        print('------ Training Finished ------')
        print(f'Total traning time is {train_time}mins')
        
    def load_model(self):
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        d = torch.load(filepath)
        self.q.load_state_dict(d['q'])
        self.q_target.load_state_dict(d['q_target'])
        self.pi.load_state_dict(d['pi'])
        self.pi_target.load_state_dict(d['pi_target'])
    
    def save_model(self):   
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        torch.save({
            'q': self.q.state_dict(),
            'q_target': self.q_target.state_dict(),
            'pi': self.pi.state_dict(),
            'pi_target': self.pi_target.state_dict()
        }, filepath)
        print("Saved model to", filepath, "...")