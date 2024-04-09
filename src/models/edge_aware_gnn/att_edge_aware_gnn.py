import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphConv
from src.models.edge_aware_gnn.edge_attention import EdgeAttention

# Classe que define um modelo GCN ciente dos atributos das arestas com mecanismo de atenção
class AttEdgeAwareGCN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, output_dim, dropout=0.5):
        super(AttEdgeAwareGCN, self).__init__()
        # Define camadas convolucionais de grafos para os atributos dos nós
        self.gc1 = GraphConv(node_input_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        # Define uma camada SAGE para os atributos das arestas
        self.edge_gcn = SAGEConv(edge_input_dim, hidden_dim)
        # Inicializa o mecanismo de atenção para as arestas
        self.edge_attention = EdgeAttention(edge_input_dim, hidden_dim)
        # Camada linear para combinar os atributos dos nós e arestas
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, node_features, edge_indices, edge_features):
        # Aplica a primeira camada convolucional e ativação ReLU aos nós
        x = F.relu(self.gc1(node_features, edge_indices))
        x = F.dropout(x, self.dropout, training=self.training)
        # Aplica a segunda camada convolucional aos nós
        x = F.relu(self.gc2(x, edge_indices))
        
        # Aplica a camada convolucional aos atributos das arestas
        e = F.relu(self.edge_gcn(edge_features, edge_indices))
        
        # Calcula os coeficientes de atenção para as arestas
        attention_coeffs = self.edge_attention(edge_features)
        
        # Inicializa tensores para agregação de informações dos vizinhos e arestas
        row, col = edge_indices
        aggregated_neighbors = torch.zeros_like(x)
        aggregated_edges = torch.zeros_like(x)

        # Realiza a agregação por soma ponderada pelas atenções
        for src, dest, edge, coeff in zip(row, col, e, attention_coeffs):
            aggregated_neighbors[dest] += coeff * x[src]
            aggregated_edges[dest] += coeff * edge
        
        # Combina os atributos dos nós com os atributos agregados das arestas
        x = torch.cat([x + aggregated_neighbors, aggregated_edges], dim=1)
        x = self.fc(x)
        
        return x
