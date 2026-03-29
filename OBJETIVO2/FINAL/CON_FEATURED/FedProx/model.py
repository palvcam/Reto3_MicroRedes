import torch
import torch.nn as nn


class PVModel(nn.Module):
    
    def __init__(self, layers_sizes=[128, 64, 32], input_size=3, dropout=0):
        super().__init__()
        
        layers = []
        
        for size in layers_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = size
        
        layers.append(nn.Linear(input_size, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)