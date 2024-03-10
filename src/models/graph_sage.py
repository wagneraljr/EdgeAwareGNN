import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# Classe que define um modelo GraphSAGE
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GraphSAGE, self).__init__()
        # Define duas camadas convolucionais SAGE
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    # Define o passo de propagação para frente (forward) do modelo
    def forward(self, x, edge_index):
        # Aplica a primeira camada convolucional, ReLU e dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # Aplica a segunda camada convolucional
        x = self.conv2(x, edge_index)
        return x
