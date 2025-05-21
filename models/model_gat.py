import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from loss import Loss

class Encoder(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels,
                 base_model=GATConv,
                 dropout: float = 0.5,
                 heads: int = 4,
                 attn_dropout: float = 0.4,
                 negative_slope: float = 0.2,
                 concat: bool = True):
        super().__init__()
        self.base_model = base_model
        self.dropout = dropout
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.negative_slope = negative_slope
        self.concat = concat
        self.k = len(hidden_channels)

        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            base_model(in_channels, 
                      hidden_channels[0] // heads,  # Split dimension per head
                      heads=heads,
                      dropout=attn_dropout,
                      negative_slope=negative_slope,
                      concat=concat)
        )

        # Subsequent layers
        for i in range(1, self.k):
            self.convs.append(
                base_model(hidden_channels[i-1], 
                          hidden_channels[i] // heads,
                          heads=heads,
                          dropout=attn_dropout,
                          negative_slope=negative_slope,
                          concat=concat if i < self.k-1 else False)  # Last layer no concat
            )

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index=None, adjs=None, dropout=True):
        if not adjs:  # Full batch processing
            for i, conv in enumerate(self.convs):
                if dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = conv(x, edge_index)
                if i < self.k - 1:  # No ReLU after last layer
                    x = F.leaky_relu(x, self.negative_slope)
        else:  # Subgraph sampling
            for i, (edge_index, _, size) in enumerate(adjs):
                if dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i < self.k - 1:
                    x = F.leaky_relu(x, self.negative_slope)
        return x

class Model(torch.nn.Module):
    def __init__(self, 
                 encoder: Encoder, 
                 in_channels: int, 
                 project_hidden,
                 activation=nn.PReLU,
                 tau: float = 0.2):  # Lower default temperature
        super().__init__()
        self.encoder = encoder
        self.tau = tau
        self.in_channels = in_channels
        self.project_hidden = project_hidden
        self.activation = activation
        self.Loss = Loss(temperature=self.tau)

        self.project = None
        if self.project_hidden is not None:
            self.project = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.project.append(nn.Linear(self.in_channels, self.project_hidden[0]))
            self.activations.append(self.activation())
            for i in range(1, len(self.project_hidden)):
                self.project.append(
                    nn.Linear(self.project_hidden[i-1], self.project_hidden[i]))
                self.activations.append(self.activation())

    def forward(self, x: torch.Tensor, edge_index=None, adjs=None) -> torch.Tensor:
        x = self.encoder(x, edge_index, adjs)
        if self.project is not None:
            for proj, activ in zip(self.project, self.activations):
                x = activ(proj(x))
        return x

    def loss(self, x: torch.Tensor, mask):
        return self.Loss(x, mask)