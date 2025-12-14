import torch
import torch.nn as nn

class MLP(torch.nn.Module):
    r"""
        An implementation of Multilayer Perceptron (MLP).
    """
    def __init__(self, input_dim=1025, hidden_sizes=(256,), activation='elu', num_classes=64):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        if sum(self.hidden_sizes) > 0: # multi-layer model
            layers = []
            for i in range(len(hidden_sizes)):
                layers.append(torch.nn.Linear(input_dim, hidden_sizes[i])) 
                if activation=='relu':
                  layers.append(torch.nn.ReLU())
                elif activation=='elu':
                  layers.append(torch.nn.ELU())
                else:
                  pass 
                input_dim = hidden_sizes[i]
            self.layers = torch.nn.Sequential(*layers)
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """forward pass"""
        if sum(self.hidden_sizes) > 0:
            x = self.layers(x)
        return self.fc(x), x
