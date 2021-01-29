import gym
import simulation
import time
import torch
from neuralnet import ActorNet
from ppo import PPO


def main():
    env = gym.make('RCSumo-v0')
    states = env.observation_space
    actions = env.action_space
    #print(env.action_space.sample()) #this shows a sample of the actions
    #print(env.observation_space.sample()) #this shows a sample of the observations
    #print(env.observation_space.shape)
    #print(env.action_space.shape)
    #print(env.observation_space.shape[0])
    #print(env.action_space.shape[0])
    competitiveAgents(env)
    #randomActions(env,states,actions)


#Competitive Baselines features
#MultiProcess multi actions
#MultiProcess multi observations
#1 to N different models in training
#Model training against pretrained models
#Model training with 1 to 1 advancement rate, no advancement, limited advancement rate.
#Episodic performance report
#Models can have different action space and observation space

def competitiveAgents(env):
    models = PPO(env)
    models.learn(10000)

def randomActions(env,states,actions):
    episodes = 10
    for episode in range(1, episodes + 1):
        state1, state2 = env.reset()
        score1 = 0
        score2 = 0

        while True:
            action1 = env.action_space.sample()
            action2 = env.action_space.sample()
            [n_state1, reward1, done1, info1], [n_state2, reward2, done2, info2] = env.step(action1,action2)
            score1 += reward1
            score2 += reward2
            if done1 or done2:
                break
        print('Episode: {}, Score1: {}, Score2: {}'.format(episode,score1,score2))

if __name__ == '__main__':
    main()
