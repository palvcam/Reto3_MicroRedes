import torch
import torch.nn as nn


class PVModel(nn.Module):
    
    def __init__(self, layers_sizes=[256,128,64], input_size=3, dropout=0.1):
        super().__init__()
        
        layers = []
        
        for size in layers_sizes:
            layers.append(nn.Linear(input_size, size)) # Capa lineal
            layers.append(nn.BatchNorm1d(size)) # Batch Normalization
            layers.append(nn.ReLU()) # Activación
            layers.append(nn.Dropout(dropout)) # Dropout
            input_size = size # Actualiza para siguiente capa
        
        layers.append(nn.Linear(input_size, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)