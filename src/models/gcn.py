import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  GraphConv

# Classe que define um modelo GCN (Graph Convolutional Network)
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        # Define duas camadas convolucionais de grafos
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.dropout = dropout

    # Define o passo de propagação para frente do modelo
    def forward(self, x, edge_index):
        # Aplica a primeira camada convolucional, ReLU e dropout
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        # Aplica a segunda camada convolucional
        x = self.gc2(x, edge_index)
        return x 
