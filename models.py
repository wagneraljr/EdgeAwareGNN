import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphConv

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

# Classe que implementa um mecanismo de atenção para arestas em redes neurais baseadas em grafos
class EdgeAttention(nn.Module):
    def __init__(self, edge_feature_dim, hidden_dim):
        super(EdgeAttention, self).__init__()
        # Uma camada linear que transforma os atributos das arestas
        self.edge_weight = nn.Linear(edge_feature_dim, hidden_dim)
        # Função de ativação LeakyReLU
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, edge_features):
        # Aplica a transformação linear aos atributos das arestas
        edge_transformed = self.edge_weight(edge_features)
        # Calcula a pontuação de atenção para cada aresta, usando produto escalar
        attention_scores = (edge_transformed * edge_transformed).sum(dim=1)
        # Aplica a função de ativação LeakyReLU às pontuações
        attention_scores = self.leaky_relu(attention_scores)
        # Normaliza as pontuações usando softmax para obter coeficientes de atenção
        attention_coeffs = F.softmax(attention_scores, dim=0)
        return attention_coeffs

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
