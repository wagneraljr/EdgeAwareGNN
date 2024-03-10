from torch import nn
from src.models.gfan.gfan import GFAN

class AttEdgeAwareGNNV2(nn.Module):
    def __init__(self,d_in_nodes,d_in_edges):
        super(AttEdgeAwareGNNV2,self,).__init__()
        self.gfan = GFAN(d_in_nodes, 32, d_in_edges,32,32,32,n_heads=4,
                         non_linearity_edge_embedd=nn.functional.softmax,
                         non_linearity_node_embedd=nn.functional.softmax)
        self.gfan2 = GFAN(32, 32, 32,32,32,32,n_heads=4)
        self.linear= nn.Linear(32,1)
    
    def forward(self, node_features, edge_features, edge_indexes):
        leaky_relu = nn.LeakyReLU(0.2)
        # x = drop1(x)
        x,y = self.gfan(node_features,edge_features,edge_indexes,leaky_relu)
        x = nn.functional.dropout(x,0.5,training=self.training)
        y = nn.functional.dropout(y,0.5,training=self.training)
        x,y = self.gfan2(x,y,edge_indexes,leaky_relu)
        x = self.linear(x)
        return x