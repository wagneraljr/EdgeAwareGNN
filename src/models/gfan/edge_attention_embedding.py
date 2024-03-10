import torch
from torch import nn
from src.models.gfan.mlp import MLP


class EdgeAttentionEmbedding(nn.Module):
    def __init__(self, d_in_nodes: int, d_out_mlp_nodes: int, d_in_edges: int, d_out_mlp_edges: int, d_out_model: int,
                  hidden_layers_mlp_nodes : list= [], hidden_layers_mlp_edges : list= [],
                  hidden_layers_mlp_model : list= [],
                  hidden_layers_mlp_attention_edges : list= [],
                  n_heads: int = 1, non_linearity = torch.nn.functional.softmax) -> None:
        super(EdgeAttentionEmbedding,self).__init__()
        self.d_in_nodes = d_in_nodes
        self.d_out_mlp_nodes = d_out_mlp_nodes
        self.d_out_mlp_edges = d_out_mlp_edges
        self.n_heads = n_heads
        self.d_out_model = d_out_model

        self.non_linearity = non_linearity

        self.Ws1 = nn.ModuleList([MLP(d_in_nodes,hidden_layers_mlp_nodes,d_out_mlp_nodes) for _ in range(n_heads)])
        self.Ws2 = nn.ModuleList([MLP(d_in_edges,hidden_layers_mlp_edges,d_out_mlp_edges) for _ in range(n_heads)])
        self.Ws3 = nn.ModuleList([MLP(d_out_mlp_nodes+d_in_edges,hidden_layers_mlp_model,d_out_model) for _ in range(n_heads)])

        self.MLPs_attention_E = nn.ModuleList([MLP(d_out_mlp_nodes+d_out_mlp_edges,hidden_layers_mlp_attention_edges,1) for _ in range(n_heads)])
        # for i in range(n_heads):
        #     nn.init.xavier_uniform_(self.Ws1[i].weight)
        #     nn.init.xavier_uniform_(self.Ws2[i].weight)
        #     nn.init.xavier_uniform_(self.MLPs_attention_E[i].weight)
    def forward(self, node_features, edge_features, edge_indexes,leaky_relu):

        num_edges = edge_features.shape[0]
        num_nodes = node_features.shape[0]
        #fixme: trocar estrutura para armazenar as atenções
        alpha_matrices = []
        for i in range(self.n_heads):
            alpha_matrix = torch.zeros((num_edges,num_nodes))
            for e,(u,v) in enumerate(zip(edge_indexes[0],edge_indexes[1])):
                transformed_hu = self.Ws1[i](node_features[u])
                transformed_hv = self.Ws1[i](node_features[u])
                transformed_huv = self.Ws2[i](edge_features[e])
                alpha_matrix[e][u] = leaky_relu(self.MLPs_attention_E[i](torch.concat([transformed_hu,transformed_huv])))
                alpha_matrix[e][v] = leaky_relu(self.MLPs_attention_E[i](torch.concat([transformed_hv,transformed_huv])))
            
            for i in range(len(alpha_matrix)):
                non_zero_attentions = alpha_matrix[i][alpha_matrix[i] != 0.0]
                normalized_attentions = torch.softmax(non_zero_attentions,dim=0)  
                alpha_matrix[i][alpha_matrix[i] != 0] = normalized_attentions

            alpha_matrices.append(alpha_matrix)
        
        embeddings =[]

        for i in range(self.n_heads):
            alpha_matrix_E = alpha_matrices[i]
            embedding = torch.zeros(num_edges,self.d_out_model)
            for e,(u,v) in enumerate(zip(edge_indexes[0],edge_indexes[1])):
                embedding[e] += self.Ws3[i](torch.concat([self.Ws1[i](alpha_matrix_E[e][u]*node_features[u])+
                                                          self.Ws1[i](alpha_matrix_E[e][v]*node_features[v]),
                                                          edge_features[e]]),)

            embeddings.append(embedding)

        return self.non_linearity(torch.mean(torch.stack(embeddings),dim=0),dim=1)
