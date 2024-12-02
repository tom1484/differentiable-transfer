import torch.nn as nn
import torch.nn.functional as F

class SpmModel(nn.Module):
    def __init__(self, obs_dim, act_dim, num_frames, num_params, hidden_dim, is_binary=True) -> None:
        super().__init__()
        
        input_dim = (obs_dim + act_dim) * num_frames + num_params
        
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, num_params)
        
        self.is_binary = is_binary
        
    def forward(self, x):
        x = F.selu(self.l1(x))
        x = F.selu(self.l2(x))
        x = self.l3(x)
        
        if self.is_binary:
            x = F.sigmoid(x)
        
        return x
