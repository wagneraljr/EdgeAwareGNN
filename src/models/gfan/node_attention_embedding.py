import torch

from src.models.gfan.mlp import MLP
from torch import nn


class NodeAttentionEmbedding(nn.Module):
    def __init__(self, d_in_nodes: int, d_out_mlp_nodes: int, d_in_edges: int, d_out_mlp_edges: int, d_out_model: int,
                  hidden_layers_mlp_nodes : list= [], hidden_layers_mlp_edges : list= [],hidden_layers_mlp_model: list= [],
                  hidden_layers_mlp_attention_nodes : list= [], hidden_layers_mlp_attention_edges : list= [],
                  n_heads: int = 1, non_linearity = torch.nn.functional.softmax) -> None:
        super(NodeAttentionEmbedding,self).__init__()
        self.d_in_nodes = d_in_nodes
        self.d_out_mlp_nodes = d_out_mlp_nodes
        self.d_out_mlp_edges = d_out_mlp_edges
        self.n_heads = n_heads
        self.d_out_model = d_out_model

        self.non_linearity = non_linearity
        n_heads
        self.Ws1 = nn.ModuleList([MLP(d_in_nodes,hidden_layers_mlp_nodes,d_out_mlp_nodes) for _ in range(n_heads)])       
        self.Ws2 = nn.ModuleList([MLP(d_in_edges,hidden_layers_mlp_edges,d_out_mlp_edges) for _ in range(n_heads)])
        self.Ws3 = nn.ModuleList([MLP(d_in_nodes+d_out_mlp_nodes+d_out_mlp_edges,hidden_layers_mlp_model,d_out_model)for _ in range(n_heads)])

        self.MLPs_attention_V = nn.ModuleList([MLP(2*d_out_mlp_nodes,hidden_layers_mlp_attention_nodes,1) for _ in range(n_heads)])
        self.MLPs_attention_E = nn.ModuleList([MLP(2*d_out_mlp_nodes+d_out_mlp_edges, hidden_layers_mlp_attention_edges,1) for _ in range(n_heads)])
        # for i in range(n_heads):
        #     nn.init.xavier_uniform_(self.Ws1[i].weight)
        #     nn.init.xavier_uniform_(self.Ws2[i].weight)
        #     nn.init.xavier_uniform_(self.MLPs_attention_V[i].weight)
        #     nn.init.xavier_uniform_(self.MLPs_attention_E[i].weight)

    def forward(self, node_features, edge_features, edge_indexes,leaky_relu):
        alphas_V = []
        num_nodes = node_features.shape[0]
        for i in range(self.n_heads):
            alpha_matrix = torch.zeros(num_nodes,num_nodes)
            for (u,v) in (zip(edge_indexes[0],edge_indexes[1])):
                transformed_hu = self.Ws1[i](node_features[u])
                transformed_hv = self.Ws1[i](node_features[v])
                alpha_matrix[u][v] = leaky_relu(self.MLPs_attention_V[i](torch.concat([transformed_hu,transformed_hv])))
             
            for i in range(len(alpha_matrix)):
                non_zero_attentions = alpha_matrix[i][alpha_matrix[i] != 0.0]
                normalized_attentions = torch.softmax(non_zero_attentions,dim=0)  
                alpha_matrix[i][alpha_matrix[i] != 0] = normalized_attentions 

            alphas_V.append(alpha_matrix)

        alphas_E = []
        for i in range (self.n_heads):
            alpha_matrix = torch.zeros((num_nodes,num_nodes))
            for e,(u,v) in enumerate(zip(edge_indexes[0],edge_indexes[1])):
                transformed_hu = self.Ws1[i](node_features[u])
                transformed_hv = self.Ws1[i](node_features[v])
                transformed_e = self.Ws2[i](edge_features[e])
                alpha_matrix[u][v] = leaky_relu(self.MLPs_attention_E[i](torch.concat([transformed_hu,
                                                                                    transformed_hv,
                                                                                    transformed_e])))
            
            for i in range(len(alpha_matrix)):
                non_zero_attentions = alpha_matrix[i][alpha_matrix[i] != 0.0]
                normalized_attentions = torch.softmax(non_zero_attentions,dim=0)  
                alpha_matrix[i][alpha_matrix[i] != 0] = normalized_attentions

            alphas_E.append(alpha_matrix)
        
        embeddings = []

        for i in range(self.n_heads):
            alpha_matrix_V = alphas_V[i]
            alpha_matrix_E = alphas_E[i] 
            embedding = torch.zeros(num_nodes,self.d_out_model)
            for e,(u,v) in enumerate(zip(edge_indexes[0],edge_indexes[1])):
                embedding[u] += self.Ws3[i](torch.concat([self.Ws1[i](alpha_matrix_V[u][v]*node_features[v]),
                                              self.Ws2[i](alpha_matrix_E[u][v]*edge_features[e]),
                                              node_features[u]]))
            embeddings.append(embedding)
        return self.non_linearity(torch.mean(torch.stack(embeddings),dim=0),dim=1)