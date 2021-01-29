import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNet(nn.Module):
    #input shoudl be 208
    #outout should be 12
    def __init__(self, inputDimension, outputDimension):
        super(ActorNet, self).__init__()
        self.inputLayer = nn.Linear(in_features=208,out_features=100)
        self.hiddenLayer1 = nn.Linear(in_features=100,out_features=100)
        self.hiddenLayer2 = nn.Linear(in_features=100,out_features=100)
        self.hiddenLayer3 = nn.Linear(in_features=100,out_features=100)
        self.hiddenLayer4 = nn.Linear(in_features=100,out_features=100)
        self.outputLayer = nn.Linear(in_features=100,out_features=12)
        self.reluActivation = nn.ReLU()
    def forward(self,obs):

        obsMod = torch.tensor(np.array(obs), dtype=torch.float)
        obsMod = torch.flatten(obsMod)
        activation1 = self.reluActivation(self.inputLayer(obsMod))
        activation2 = self.reluActivation(self.hiddenLayer1(activation1))
        activation3 = self.reluActivation(self.hiddenLayer2(activation2))
        activation4 = self.reluActivation(self.hiddenLayer3(activation3))
        activation5 = self.reluActivation(self.hiddenLayer4(activation4))
        output = self.reluActivation(self.outputLayer(activation5))
        return output
