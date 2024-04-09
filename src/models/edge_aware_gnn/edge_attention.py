import torch.nn as nn
import torch.nn.functional as F

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
