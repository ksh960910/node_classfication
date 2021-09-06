import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes):
        super(GCN, self).__init__()
        self.num_features =num_features
        self.num_classes = num_classes
        
        # Initialize layers (2 message passing 단계)
        self.conv1 = GCNConv(self.num_features, hidden_channels, normalize=True, cached=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, self.num_classes)
    
    def forward(self, x, edge_index):
        # First message passing layer
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)  # training시에만 dropout 할 수 있도록
        
        # Second message passing layer
        x = self.conv2(x, edge_index)
        #x = x.relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        
        # Output layer
        x = F.softmax(self.out(x), dim=1)
        return x
    