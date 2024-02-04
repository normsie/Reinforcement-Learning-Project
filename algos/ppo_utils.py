import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
import  torch

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, env, hidden_size=32):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        
        #adding code from this last onward
        
        #Action Branch - capture features from the input state relevant for deciding the prob distribution over actions
        self.fc1_a = torch.nn.Linear(state_space, hidden_size)
        self.fc2_a = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_a = torch.nn.Linear(hidden_size, action_space)
        
        #State Value Branch - estimating the value of the state
        self.fc1_c = torch.nn.Linear(state_space, hidden_size)
        self.fc2_c = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_c = torch.nn.Linear(hidden_size, 1) 
        
        self.init_weights() #initialize the weights of the model

    def init_weights(self):
        #This method initialize the weight of the linear layers with a normal dist (mean=0, std=0.1) and set biases=0
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 1e-1) 
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        #Defining the forward pass of the NN
        x_a = self.fc1_a(x)
        x_a = F.relu(x_a)
        x_a = self.fc2_a(x_a)
        x_a = F.relu(x_a)
        x_a = self.fc3_a(x_a)

        x_c = self.fc1_c(x)
        x_c = F.relu(x_c)
        x_c = self.fc2_c(x_c)
        x_c = F.relu(x_c)
        x_c = self.fc3_c(x_c)

        mean = torch.tanh(x_a)
        std = F.softplus(x_a)
        action_dist = Normal(mean, std)

        return action_dist, x_c