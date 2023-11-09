import networkx as nx
import numpy as np
import torch

def load_data(filepath, traffic_matrix_filepath):
    # Load the Abilene network graph
    G = nx.read_gml(filepath)

    # Create a mapping from node labels to integer indices
    label_to_index = {label: idx for idx, label in enumerate(G.nodes())}
    
    # Update the graph to have integer labels
    G = nx.relabel_nodes(G, label_to_index)

    # Extract node features (here, simply using a one-hot encoding)
    node_features = np.eye(G.number_of_nodes())
    node_features = torch.tensor(node_features, dtype=torch.float32)

    # Extract edge indices
    edge_indices = np.array(G.edges())
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                    
    # Calculate edge features
    edge_features = []
    for edge in G.edges(data=True):
        feature = edge[2].get('feature', [0]*8)  # Use a list of 8 zeros as default value if 'feature' attribute is not found
        edge_features.append(feature)
        
    # Enhanced edge features
    # Calculate edge betweenness centrality
    edge_betweenness = nx.edge_betweenness_centrality(G)
    
    # Calculate edge features
    # edge_features = []
    # for edge in G.edges(data=True):
    #     src, dst = edge[0], edge[1]
        
    #     # Original feature
    #     feature = edge[2].get('feature', [0]*8)  # Retain the original edge feature
        
    #     # Edge betweenness
    #     feature.append(edge_betweenness[(src, dst)])
        
    #     # Edge degree (sum of the degrees of the two nodes connected by the edge)
    #     feature.append(G.degree[src] + G.degree[dst])
        
    #     # Edge clustering coefficient
    #     clustering_src = nx.clustering(G, src)
    #     clustering_dst = nx.clustering(G, dst)
    #     avg_clustering = (clustering_src + clustering_dst) / 2
    #     feature.append(avg_clustering)
        
    #     edge_features.append(feature)

    # # Convert edge features to tensor    
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    
    # Parsing the file and extracting the traffic matrix values
    traffic_matrix = []
    with open(traffic_matrix_filepath, "r") as file:
#        reader = sorted(csv.reader(file))
        reader = csv.reader(file)
        for row in reader:
            # Filtering out the metadata lines
            if not row[0].startswith("#"):
                traffic_matrix.append([float(value) for value in row])

    traffic_matrix = [row[1:] for row in traffic_matrix[1:]]
    # Calculating node loads
    node_loads_values = []
    for i in range(len(traffic_matrix)):
        node_load = sum(traffic_matrix[i]) + sum(row[i] for row in traffic_matrix)
        node_loads_values.append(node_load)
     
    # Compute the total load
    total_load = sum(node_loads_values)

    # Normalize the given node loads
    normalized_node_loads = [load / total_load for load in node_loads_values] 
    node_loads_values = normalized_node_loads
    
    node_loads = torch.tensor(node_loads_values, dtype=torch.float32).view(-1, 1)
       
    return node_features, edge_indices, edge_features, node_loads

import csv

def generate_edge_meta_index(edge_index):
    edge_connections = []
    
    # Create a dictionary where keys are nodes and values are lists of incident edges
    node_to_edges = {}
    for idx, (src, dest) in enumerate(edge_index.t()):
        node_to_edges.setdefault(src.item(), []).append(idx)
        node_to_edges.setdefault(dest.item(), []).append(idx)
    
    # For each node, generate all pairs of incident edges
    for node, edges in node_to_edges.items():
        num_edges = len(edges)
        for i in range(num_edges):
            for j in range(i + 1, num_edges):  # This ensures no double-counting or self-connections
                edge_connections.append((edges[i], edges[j]))
                
    edge_meta_index = torch.tensor(edge_connections).t()
    
    return edge_meta_index

def generate_edge_meta_index_with_self_loops(edge_indices):
    """
    Generate edge_meta_index for the meta-graph based on the provided edge_indices.
    
    Parameters:
    - edge_indices (torch.Tensor): Tensor of shape [2, num_edges] indicating source and destination nodes of each edge.
    
    Returns:
    - edge_meta_index (torch.Tensor): Tensor of shape [2, num_meta_edges] for the meta-graph.
    """

    # Convert edge_indices to a list of edges for easier processing
    edge_list = edge_indices.t().tolist()
    
    # Create a dictionary to hold edges connected to each node
    node_to_edges = {}
    for idx, (src, dst) in enumerate(edge_list):
        if src not in node_to_edges:
            node_to_edges[src] = []
        if dst not in node_to_edges:
            node_to_edges[dst] = []
        node_to_edges[src].append(idx)
        node_to_edges[dst].append(idx)
    
    # Create meta-edges based on common nodes
    meta_edges = set()  # Use a set to avoid duplicate edges
    for edges in node_to_edges.values():
        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                e1, e2 = edges[i], edges[j]
                meta_edges.add((e1, e2))
                meta_edges.add((e2, e1))  # Add both directions to make it undirected

    # Convert meta_edges to the desired tensor format
    edge_meta_index = torch.tensor(list(meta_edges)).t()

    return edge_meta_index
