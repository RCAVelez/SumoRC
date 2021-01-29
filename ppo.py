import torch
from neuralnet import ActorNet
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import numpy as np

#This will be the base for Agoge's PPO
class PPO:
    def __init__(self,env): #be able to pass in multiple actors and create critics for them
        self._init_hyperparameters()

        self.env = env
        #may need to modify these two
        #self.obs_dim = env.observation_space.shape[0]
        self.obs_dim = 208
        #self.act_dim = env.action_space.shape[0]
        self.act_dim = 12

        #Algorithm Step 1
        #Initialize actor and critic networks
        self.actor1 = ActorNet(self.obs_dim, self.act_dim)
        self.critic1 = ActorNet(self.obs_dim, 1)

        self.actor2 = ActorNet(self.obs_dim,self.act_dim)
        self.critic2 = ActorNet(self.obs_dim, 1)

        self.actor_optim1 = Adam(self.actor1.parameters(), lr=self.lr)
        self.critic_optim1 = Adam(self.critic1.parameters(), lr=self.lr)

        self.actor_optim2 = Adam(self.actor2.parameters(), lr=self.lr)
        self.critic_optim2 = Adam(self.critic2.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.act_dim,),fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def _init_hyperparameters(self):
        #Default values for hyperparameters
        self.timesteps_per_batch = 4800         #timesteps per batch
        self.max_timesteps_per_episode = 1600   #timesteps per episode
        self.gamma = 0.95                       #discount factor
        self.n_updates_per_iteration = 5        #number of epochs per iteration
        self.clip = 0.2                         #clip threshold
        self.lr = 0.005                         #learnig rate of optimizers

    def rollout(self):
        #This collects data
        batch_obs1 = [] #batch observations
        batch_acts1 = [] #batch actions
        batch_log_probs1 = [] #log probs of each action
        batch_rews1 = [] #batch rewards
        batch_rtgs1 = []  #batch rewards-to-go
        batch_lens1 = []  #episodic lengths in batch

        batch_obs2 = [] #batch observations
        batch_acts2 = [] #batch actions
        batch_log_probs2 = [] #log probs of each action
        batch_rews2 = [] #batch rewards
        batch_rtgs2 = []  #batch rewards-to-go
        batch_lens2 = []  #episodic lengths in batch

        t = 0
        while t < self.timesteps_per_batch:

            ep_rews1 = []
            ep_rews2 = []

            obs1, obs2 = self.env.reset()
            done1 = False
            done2 = False


            for episode_timestep in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs1.append(obs1)
                batch_obs2.append(obs2)

                action1, log_prob1 = self.get_action(obs1,1)
                action2, log_prob2 = self.get_action(obs2,2)

                output1, output2 = self.env.step(action1, action2)
                n_state1, reward1, done1, info1 = output1
                n_state2, reward2, done2, info2 = output2

                ep_rews1.append(reward1)
                batch_acts1.append(action1)
                batch_log_probs1.append(log_prob1)

                ep_rews2.append(reward2)
                batch_acts2.append(action2)
                batch_log_probs2.append(log_prob2)

                if done1 or done2:
                    break

            batch_lens1.append(episode_timestep + 1)
            batch_rews1.append(ep_rews1)

            batch_lens2.append(episode_timestep + 1)
            batch_rews2.append(ep_rews2)

        batch_obs1 = torch.tensor(batch_obs1, dtype=torch.float)
        batch_acts1 = torch.tensor(batch_acts1, dtype=torch.float)
        batch_log_probs1 = torch.tensor(batch_log_probs1, dtype=torch.float)

        batch_obs2 = torch.tensor(batch_obs2, dtype=torch.float)
        batch_acts2 = torch.tensor(batch_acts2, dtype=torch.float)
        batch_log_probs2 = torch.tensor(batch_log_probs2, dtype=torch.float)

        batch_rtgs1 = self.compute_rtgs(batch_rews1)
        batch_rtgs2 = self.compute_rtgs(batch_rews2)

        return [batch_obs1, batch_acts1, batch_log_probs1, batch_rtgs1 ,batch_lens1] , [batch_obs2, batch_acts2, batch_log_probs2, batch_rtgs2 ,batch_lens2]

    def get_action(self,obs, actorIndex):
        mean = None
        if actorIndex == 1:
            mean = self.actor1(obs)
        if actorIndex == 2:
            mean = self.actor2(obs)

        dist = MultivariateNormal(mean,self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach() #might break for me here

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_retgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts, critic_index):
        V = None
        mean = None
        dist = None
        log_probs = None
        if critic_index == 1:
            V = self.critic1(batch_obs).squeeze()
            mean = self.actor1(batch_obs)

        if critic_index == 2:
            V = self.critic2(batch_obs).squeeze()
            mean = self.actor2(batch_obs)

        dist = MultivariateNormal(mean,self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

    def learn(self, total_timesteps):

        current_step = 0
        while current_step < total_timesteps: #Algorithm Step 2
            rollout1, rollout2 = self.rollout()
            batch_obs1, batch_acts1, batch_log_probs1, batch_rtgs1, batch_lens1 = rollout1
            batch_obs2, batch_acts2, batch_log_probs2, batch_rtgs2, batch_lens2 = rollout2

            current_step += np.sum(batch_lens1)

            V1, _ = self.evaluate(batch_obs1, batch_acts1, 1)
            A_k1 = batch_rtgs1 - V1.detach()
            A_k1 = (A_k1 - A_k1.mean()) / (A_k1.std() + 1e-10)

            V2, _ = self.evaluate(batch_obs2, batch_acts2, 2)
            A_k2 = batch_rtgs2 - V2.detach()
            A_k2 = (A_k2 - A_k2.mean()) / (A_k2.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                V1, curr_log_probs1 = self.evaluate(batch_obs1,batch_acts1, 1)
                V2, curr_log_probs2 = self.evaluate(batch_obs2,batch_acts2, 2)
                ratios1 = torch.exp(curr_log_probs1 - batch_log_probs1)
                ratios2 = torch.exp(curr_log_probs2 - batch_log_probs2)

                surr1_1 = ratios1 * A_k1
                surr1_2 = ratios2 * A_k2

                surr2_1 = torch.clamp(ratios1, 1 - self.clip, 1 + self.clip) * A_k1
                surr2_2 = torch.clamp(ratios2, 1 - self.clip, 1 + self.clip) * A_k2

                actor_loss1 = (-torch.min(surr1_1,surr2_1)).mean()
                actor_loss2 = (-torch.min(surr1_2,surr2_2)).mean()

                critic_loss1 = nn.MSELoss()(V1, batch_rtgs1)
                critic_loss2 = nn.MSELoss()(V2, batch_rtgs2)

                self.actor_optim1.zero_grad()
                self.actor_optim2.zero_grad()
                actor_loss1.backward(retain_graph=True)
                actor_loss2.backward(retain_graph=True)
                self.actor_optim1.step()
                self.actor_optim2.step()

                self.critic_optim1.zero_grad()
                self.critic_optim2.zero_grad()
                critic_loss1.backward()
                critic_loss2.backward()
                self.critic_optim1.step()
                self.critic_optim2.step()
