import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphConv, EdgeConv, aggr


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_dim, heads, dropout=0.5):
        super(MultiHeadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        
        self.W_q = torch.nn.Linear(hidden_dim, heads * hidden_dim)
        self.W_k = torch.nn.Linear(hidden_dim, heads * hidden_dim)
        self.W_v = torch.nn.Linear(hidden_dim, heads * hidden_dim)
        
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        # Linearly transform & reshape query, key, and value
        Q = self.W_q(query).view(B, self.heads, self.hidden_dim)
        K = self.W_k(key).view(B, self.heads, self.hidden_dim)
        V = self.W_v(value).view(B, self.heads, self.hidden_dim)
        print("self.heads:", self.heads)
        print("self.hidden_dim:", self.hidden_dim)
        print("self.W_v(value).shape:", self.W_v(value).shape)
        # Calculate attention scores
        attention_scores = torch.einsum('bhd,bhd->bh', (Q, K))  # Adjusted equation
        
        # print("attention_scores shape:", attention_scores.shape)
        # print("mask shape:", mask.shape)


        # Apply mask if provided
        # if mask is not None:
        #     attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Normalize the attention scores
        attention_scores = self.softmax(attention_scores / self.hidden_dim**0.5)

        # Calculate the weighted sum of the values
        weighted_values = torch.einsum('bh,bhd->bhd', (attention_scores, V))  # Adjusted equation

        # Concatenate the heads
        attention_output = weighted_values.view(B, self.heads * self.hidden_dim)

        # Apply dropout
        attention_output = torch.nn.functional.dropout(attention_output, self.dropout, training=self.training)

        return attention_output


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
 
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        #nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# class EdgeAwareGCN(nn.Module):
#     def __init__(self, node_input_dim, edge_input_dim, hidden_dim, output_dim,
#                   dropout=0.5):
#         super(EdgeAwareGCN, self).__init__()
        
#         # GCN Layers for node features
#         # self.gc1 = GraphConvolution(node_input_dim, hidden_dim)
#         # self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
#         self.gc1 = GraphConv(node_input_dim, hidden_dim)
#         self.gc2 = GraphConv(hidden_dim, hidden_dim)
        
#         # Linear Layer for edge features
#         self.edge_layer = nn.Linear(edge_input_dim, hidden_dim)
        
#         # Batch Normalization
#         # self.batch_norm1 = nn.BatchNorm1d(hidden_dim * 2)
        
#         # Final Linear Layer
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
#         # Dropout
#         self.dropout = dropout
        
#     # def forward(self, node_features, adj, edge_indices, edge_features):

#     #     # Node features through GCN layers
#     #     x = self.gc1(node_features, adj)
#     #     x = F.dropout(x, self.dropout, training=self.training)
#     #     x = F.relu(x)
    
#     #     x = self.gc2(x, adj)
#     #     x = F.dropout(x, self.dropout, training=self.training)
    
#     #     # Edge features through a Linear layer
#     #     e = F.relu(self.edge_layer(edge_features))
    
#     #     # Sum Aggregation
#     #     row, col = edge_indices
#     #     aggregated_neighbors = torch.zeros_like(x)
#     #     aggregated_edges = torch.zeros_like(x)
    
#     #     for src, dest, edge in zip(row, col, e):
#     #         aggregated_neighbors[dest] += x[src]
#     #         aggregated_edges[dest] += edge
    
#     #     # Avg Aggregation 
#     #     # for src, dest, edge in zip(row, col, e):
#     #     #     aggregated_neighbors[dest] += x[src] / len(edge_indices)
#     #     #     aggregated_edges[dest] += edge / len(edge_indices)  
    
#     #     # Concatenation of node and edge features
#     #     x = torch.cat([x + aggregated_neighbors, aggregated_edges], dim=1)
    
#     #     # Batch normalization and final layer
#     #     x = self.batch_norm1(x)
#     #     x = self.fc(x)
    
#     #     # Apply sigmoid activation function if we are doing classification
#     #     if self.training:
#     #         x = F.sigmoid(x)
    
#     #     return x

#     #def forward(self, node_features, adj, edge_indices, edge_features):
#     def forward(self, node_features, edge_indices, edge_features):

#         # Node features through GCN layers
#         x = F.relu(self.gc1(node_features, edge_indices))
#         x = F.dropout(x, self.dropout)#, training=self.training)
#         x = F.relu(self.gc2(x, edge_indices))

#         # Edge features through a Linear layer
#         e = F.relu(self.edge_layer(edge_features))
#         #e = F.leaky_relu(self.edge_layer(edge_features))
#         row, col = edge_indices
#         aggregated_neighbors = torch.zeros_like(x)
#         aggregated_edges = torch.zeros_like(x)
        
#         # Sum Aggregation
#         for src, dest, edge in zip(row, col, e):
#             aggregated_neighbors[dest] += x[src]
#             aggregated_edges[dest] += edge

#         # Avg Aggregation 
#         # for src, dest, edge in zip(row, col, e):
#         #     aggregated_neighbors[dest] += x[src] / len(edge_indices)
#         #     aggregated_edges[dest] += edge / len(edge_indices)  
            
#         # Concatenation of node and edge features
#         x = torch.cat([x + aggregated_neighbors, aggregated_edges], dim=1)
        
#         # Batch normalization and final layer
#         # x = self.batch_norm1(x)
#         x = self.fc(x)
        
#         return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()

        # Use PyTorch Geometric's GraphConv layers
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):

        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return x 

class EdgeAwareGCN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, output_dim, dropout=0.5):
        super(EdgeAwareGCN, self).__init__()

        self.gc1 = GraphConv(node_input_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)

        # SAGEConv layer for edge features (meta-graph nodes)
        self.edge_gcn = GraphConv(edge_input_dim, hidden_dim)

        # Final Linear Layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Dropout
        self.dropout = dropout

    def forward(self, node_features, edge_index, edge_features, edge_meta_index):
        
        # print(edge_meta_index.shape)
        # print(edge_index.shape)
        
        # Node features through SAGEConv layers
        x = F.relu(self.gc1(node_features, edge_index))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.gc2(x, edge_index))

        # Edge features (meta-graph nodes) through a SAGEConv layer
        e = self.edge_gcn(edge_features, edge_meta_index)

        row, col = edge_index
        aggregated_neighbors = torch.zeros_like(x)
        aggregated_edges = torch.zeros_like(x)
        
        # Sum Aggregation
        for idx, (src, dest) in enumerate(zip(row, col)):
            aggregated_neighbors[dest] += x[src]
            aggregated_edges[dest] += e[idx]

        # Concatenation of node and edge features
        x = torch.cat([x + aggregated_neighbors, aggregated_edges], dim=1)

        x = self.fc(x)

        return x

class EdgeAttention(nn.Module):
    def __init__(self, edge_feature_dim, hidden_dim):
        super(EdgeAttention, self).__init__()
        self.edge_weight = nn.Linear(edge_feature_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, edge_features):
        # Transform edge features
        edge_transformed = self.edge_weight(edge_features)
        
        # Compute attention scores as dot product (check different methods)
        attention_scores = (edge_transformed * edge_transformed).sum(dim=1)
        
        # Apply leaky relu
        attention_scores = self.leaky_relu(attention_scores)
        
        # Convert scores to attention coefficients using softmax
        attention_coeffs = F.softmax(attention_scores, dim=0)
        
        return attention_coeffs

class EdgeAttention2(nn.Module):
    def __init__(self, edge_feature_dim, hidden_dim):
        super(EdgeAttention2, self).__init__()
        self.edge_weight = nn.Linear(edge_feature_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, edge_features):
        # Transform edge features
        edge_transformed = self.edge_weight(edge_features)
        
        # Compute attention scores with learnable weights
        attention_scores = torch.sum(edge_transformed, dim=1)
        
        # Apply leaky relu
        attention_scores = self.leaky_relu(attention_scores)
        
        # Convert scores to attention coefficients using softmax
        attention_coeffs = F.softmax(attention_scores, dim=0)
        
        return attention_coeffs

class AttEdgeAwareGCN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, output_dim, dropout=0.5):
        super(AttEdgeAwareGCN, self).__init__()
        
        self.gc1 = GraphConv(node_input_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.edge_gcn = SAGEConv(edge_input_dim, hidden_dim)
        
        # Define the attention mechanism
        self.edge_attention = EdgeAttention(edge_input_dim, hidden_dim)
        
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = dropout
        
    def forward(self, node_features, edge_indices, edge_features):
        x = F.relu(self.gc1(node_features, edge_indices))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, edge_indices))
    
        e = F.relu(self.edge_gcn(edge_features, edge_indices))
        
        # Compute attention coefficients for edges
        attention_coeffs = self.edge_attention(edge_features)
        
        row, col = edge_indices
        aggregated_neighbors = torch.zeros_like(x)
        aggregated_edges = torch.zeros_like(x)
           
        for src, dest, edge, coeff in zip(row, col, e, attention_coeffs):
            aggregated_neighbors[dest] += coeff * x[src]
            aggregated_edges[dest] += coeff * edge
            
        x = torch.cat([x + aggregated_neighbors, aggregated_edges], dim=1)

        x = self.fc(x)
        
        return x
    
        # neighbor_counts = torch.zeros_like(x).sum(dim=1)  # [num_nodes]
   
        # for src, dest, edge, coeff in zip(row, col, e, attention_coeffs):
        #     aggregated_neighbors[dest] += coeff * x[src]
        #     aggregated_edges[dest] += coeff * edge
        #     neighbor_counts[dest] += 1
   
        # # Ensure that the count is never zero to avoid division by zero.
        # neighbor_counts = torch.clamp(neighbor_counts, min=1)
        # aggregated_neighbors = aggregated_neighbors.div(neighbor_counts.unsqueeze(-1))
        # aggregated_edges = aggregated_edges.div(neighbor_counts.unsqueeze(-1))


# class EGATLayer(nn.Module):
#     def __init__(self, node_dim, edge_dim, out_dim, heads=8):
#         super(EGATLayer, self).__init__()

#         # Local feature extractor for nodes
#         self.node_feat = nn.Linear(node_dim, out_dim * heads)
#         #print("edge_dim:", edge_dim)
# #        self.edge_feat = nn.Linear(edge_dim, out_dim * heads)
#         self.edge_feat = nn.Linear(out_dim + edge_dim, out_dim * heads)

#         # Edge feature aggregator and updater
#         print(self.edge_feat.weight.shape)

#         # Attention mechanism components
#         self.attention_W = nn.Parameter(torch.Tensor(1, heads, out_dim))
#         self.leakyrelu = nn.LeakyReLU()

#         self.heads = heads
#         self.out_dim = out_dim

#     def forward(self, x, edge_index, edge_features):
#         # Extracting node features
#         node_features = self.node_feat(x).view(-1, self.heads, self.out_dim)
      
#         edge_features = self.edge_feat(edge_features).view(-1, self.heads, self.out_dim)
        
#         # Creating edge updates based on connected node features
#         start, end = edge_index
#         print("node_features[start] shape:", node_features[start].shape)
#         print("node_features[end] shape:", node_features[end].shape)
#         print("edge_features shape:", edge_features.shape)

#         edge_updates = torch.cat([node_features[start], node_features[end], edge_features], dim=-1)
        
#         # Updating edge features
#         edge_features = self.edge_feat(edge_updates).view(-1, self.heads, self.out_dim)

#         # Attention mechanism
#         attention_score = self.leakyrelu((node_features[start] * self.attention_W).sum(dim=-1) + (edge_features * self.attention_W).sum(dim=-1))
#         attention_score = F.softmax(attention_score, dim=0)

#         # Aggregating neighboring features with attention scores
#         out = torch.zeros_like(node_features)
#         for i, (s, e) in enumerate(zip(start, end)):
#             out[e] += attention_score[i] * edge_features[i]
        
#         return out, edge_features

# class EGAT(nn.Module):
#     def __init__(self, node_dim, edge_dim, hidden_dim, out_dim, heads=8):
#         super(EGAT, self).__init__()

#         # Define EGAT layers
#         self.layer1 = EGATLayer(node_dim, edge_dim, hidden_dim, heads)
#         self.layer2 = EGATLayer(hidden_dim, hidden_dim, out_dim, heads)

#     def forward(self, x, edge_index, edge_features):
#         # Apply first EGAT layer
#         x1, edge_features = self.layer1(x, edge_index, edge_features)
#         x1 = F.elu(x1)
        
#         # Apply second EGAT layer
#         x2, edge_features = self.layer2(x1, edge_index, edge_features)

#         # Multi-Scale merge strategy: concatenating features from both layers
#         return torch.cat([x1, x2], dim=-1)



class EGAT(nn.Module):
    def __init__(self, in_features, out_features, edge_features, hidden_dim=128, heads=8, dropout=0.5):
        super(EGAT, self).__init__()

        # Local feature extractor
        self.local_feature_extractor = nn.Linear(in_features, hidden_dim)
        
        num_edge_features = 8  # Or whatever is the correct number
        self.edge_feature_aggregator = nn.Linear(num_edge_features, hidden_dim)

        # Edge feature aggregator
        #self.edge_feature_aggregator = nn.Linear(edge_features, hidden_dim)

        # Attention mechanism
        self.attention_mechanism = MultiHeadAttention(hidden_dim, heads, dropout=dropout)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, out_features)

    def forward(self, x, edge_index, edge_features):

        # Extract local feature representations
        local_features = self.local_feature_extractor(x)

        # Aggregate edge features
        edge_features = self.edge_feature_aggregator(edge_features)

        # Calculate attention scores
        #attention_scores = self.attention_mechanism(local_features, local_features, edge_features, edge_index)[0]
        attention_scores = self.attention_mechanism(local_features, local_features, local_features, edge_index)[0]


        # Refine aggregated feature representations
        aggregated_features = torch.matmul(attention_scores, local_features)

        # Output layer
        output = self.output_layer(aggregated_features)

        return output