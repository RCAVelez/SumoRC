import gym
import numpy as np
import math
import pybullet as p
from simulation.resources.rcclass import RCClass
from simulation.resources.sumoarena import SumoArena
import time

# Current works for one car
# My simulation will have two cars
# Need to update everything to work for two cars
#http://robogames.net/rules/all-sumo.php
class simulationEnv(gym.Env):

    def __init__(self):
        self.client = p.connect(p.GUI)
        #self.client = p.connect(p.DIRECT)

        #Actions available
        #Each 3 is a wheel with options:
        #0 to reverse 1 revolution
        #1 to do nothing
        #2 to forward 1 revolution
        self.action_space = gym.spaces.MultiDiscrete([3,3,3,3])

        self.observation_space = gym.spaces.Box(low=-154.0, high=154.0, shape=(16,13), dtype=np.float32)

        self.match_start_time = None

        self.reward1 = 0
        self.done1 = False
        self.rc_car1 = None
        self.rcCarObservation1 = 0

        self.reward2 = 0
        self.done2 = False
        self.rc_car2 = None
        self.rcCarObservation2 = 0

        self.conversions = {    0 : -10, #rotate back
                                1 : 0,   #no rotation
                                2 : 10   #rotate forward
                            }
    def convert(self, action):
        newArr = []
        for value in action:
            newArr.append(self.conversions[value])
        return newArr

    def step(self, action1, action2):
        # Feed actions to the rc_cars and get observation of rc_cars' state

        action1 = np.array(action1).reshape(4,3)
        action2 = np.array(action2).reshape(4,3)

        action1 = np.argmax(action1,axis=1)
        action2 = np.argmax(action2,axis=1)

        action1 = self.convert(action1).copy()
        action2 = self.convert(action2).copy()
        
        self.rc_car1.applyAction(action1) #will need multiprocessing for both actions at same time
        self.rc_car2.applyAction(action2)

        #reduce length of episode
        p.stepSimulation()

        #get environment feedback
        self.rcCarObservation1 = self.rc_car1.getObservation() #will need multiprocessing for both observations at same time
        self.rcCarObservation2 = self.rc_car2.getObservation()


        current_time = int(time.time() % 60)

        if current_time - self.match_start_time == 7:
            self.reward1 -= 200
            self.reward2 -= 200
            self.done1 = True
            self.done2 = True

        if self.rcCarObservation1[0][3] > 0.0 and self.rcCarObservation2[0][3] < 0.0:
            self.reward1 += 100
            self.reward2 -= 100
            self.done1 = True
            self.done2 = True

        if self.rcCarObservation1[0][3] < 0.0 and self.rcCarObservation2[0][3] > 0.0:
            self.reward1 -= 100
            self.reward2 += 100
            self.done1 = True
            self.done2 = True

        if self.rcCarObservation1[0][3] < 0.0 and self.rcCarObservation2[0][3] < 0.0:
            self.reward1 -= 100
            self.reward2 -= 100
            self.done1 = True
            self.done2 = True

        if self.rcCarObservation1[0][3] > 0.0 and self.rcCarObservation2[0][3] > 0.0:
            self.reward1 += 10
            self.reward2 += 10

        #how to reward/punish for inactivity/activity?

        info1 = {}
        info2 = {}

        return [self.rcCarObservation1, self.reward1, self.done1, info1] , [self.rcCarObservation2, self.reward2, self.done2, info2]

    def seed(self, seed=None):
        # self.np_random, seed = gym.utils.seeding.np_random(seed)
        # return [seed]
        return None

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # Reload the SumoArena and rc car, in the future reset both rc car positions
        SumoArena(self.client)
        self.rc_car1 = RCClass(self.client, [0.25, 0, 0.5], [1,0,2,0]) #fix wxyz for base orientation
        self.reward1 = 0
        self.done1 = False

        self.rc_car2 = RCClass(self.client, [-0.25, 0, 0.5], [-1,0,2,0])
        self.reward2 = 0
        self.done2 = False

        self.rcCarObservation1 = self.rc_car1.getObservation()
        self.rcCarObservation2 = self.rc_car2.getObservation()

        assert self.rcCarObservation1 is not None
        assert self.rcCarObservation2 is not None

        self.match_start_time = int(time.time() % 60)

        return [self.rcCarObservation1, self.rcCarObservation2]

    def render(self):
        return None

    def close(self):
        p.disconnect(self.client)
