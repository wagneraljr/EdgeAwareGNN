import torch
from torch import nn
from src.models.gfan.edge_attention_embedding import EdgeAttentionEmbedding

from src.models.gfan.node_attention_embedding import NodeAttentionEmbedding

class GFAN(nn.Module):
    def __init__(self, d_in_nodes: int, d_out_mlp_nodes: int, d_in_edges: int, d_out_mlp_edges: int,
                  d_out_node_attention_embed: int, d_out_edge_attention_embed :int,
                 hidden_layers_mlp_nodes : list= [], hidden_layers_mlp_edges : list= [],
                  hidden_layers_mlp_model_node_attentions : list = [],hidden_layers_mlp_model_edge_attentions : list = [],hidden_layers_mlp_attention_nodes : list= [], hidden_layers_mlp_attention_edges : list= [],
                  n_heads: int = 1,non_linearity_node_embedd = nn.functional.softmax,non_linearity_edge_embedd = nn.functional.softmax,) -> None:
        super(GFAN,self).__init__()
        self.node_embedding = NodeAttentionEmbedding(d_in_nodes,d_out_mlp_nodes,d_in_edges,d_out_mlp_edges,d_out_node_attention_embed,
                                                     hidden_layers_mlp_nodes,hidden_layers_mlp_edges,hidden_layers_mlp_model_node_attentions,hidden_layers_mlp_attention_nodes,hidden_layers_mlp_attention_edges,

                                                     n_heads,non_linearity_node_embedd) 
        
        self.edge_embedding = EdgeAttentionEmbedding(d_in_nodes,d_out_mlp_nodes,d_in_edges,d_out_mlp_edges,d_out_edge_attention_embed,
                                                             hidden_layers_mlp_nodes,hidden_layers_mlp_edges,hidden_layers_mlp_model_edge_attentions,
                                                             hidden_layers_mlp_attention_edges,n_heads,non_linearity_edge_embedd)

    
    def forward(self, node_features, edge_features, edge_indexes,leaky_relu):
        node_embedding = self.node_embedding(node_features,edge_features,edge_indexes,leaky_relu)
        edge_embedding = self.edge_embedding(node_features,edge_features,edge_indexes,leaky_relu)
        return node_embedding, edge_embedding    

# leaky_relu = nn.LeakyReLU(0.2)
# gfan = GFAN(3,10,4,10,20,20,n_heads=1)
# a = gfan(torch.tensor([[1,1,1],
#                                  [1,1,1],
#                                  [1,1,1],
#                                 ],dtype=float),torch.tensor([[2,2,2,2],
#                                                                [3,3,3,3],
#                                                                [4,4,4,4],
#                                                                [5,5,5,5]],dtype=float),torch.tensor([[0,1,2,1],
#                                                                                                  [1,0,1,2]],dtype=int),leaky_relu)
# print(a)
       
