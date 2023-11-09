import torch
import torch.nn as nn
from models import GCN, GraphSAGE, EdgeAwareGCN, AttEdgeAwareGCN
from data_utils import load_data
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import r2_score

seed = 132
torch.manual_seed(seed)
np.random.seed(seed)

# Load the Abilene network graph
node_features, edge_indices, edge_features, \
    actual_node_loads = load_data("Abilene.gml", "Data/tm.2004-09-10.16-00-00.dat")

# Defining loss function
loss_fn = nn.MSELoss()
num_epochs = 200

# Create models
#edge_aware_model = EdgeAwareGCN(node_features.size(1), edge_features.size(1), 64, 1, 0.44198923668560325)
#edge_aware_model = EdgeAwareGCN(node_features.size(1), edge_features.size(1), 64, 1, 0.5)
#edge_aware_model = EGAT(node_features.size(1), edge_features.size(1), 64, 1, 8)
edge_aware_model = AttEdgeAwareGCN(node_features.size(1), edge_features.size(1), 64, 1, 0.5)
optimizer_edgeaware = torch.optim.Adam(list(edge_aware_model.parameters()), lr=0.01)
#optimizer_edgeaware = torch.optim.Adam(list(edge_aware_model.parameters()), lr=0.002027845462984863)
#scheduler_edgeaware = StepLR(optimizer_edgeaware, step_size=50, gamma=0.2939709566560924)
scheduler_edgeaware = StepLR(optimizer_edgeaware, step_size=50, gamma=0.9)
edge_aware_losses = []

traffic_matrix_files = sorted([file for file in os.listdir() if file.endswith('.dat')])


for epoch in range(num_epochs):        
    for traffic_matrix_filepath in traffic_matrix_files:
        # Load the Abilene network graph and respective data from the current traffic matrix file
        node_features, edge_indices, edge_features, node_loads = load_data("Abilene.gml", traffic_matrix_filepath)
        
        optimizer_edgeaware.zero_grad()

        edge_aware_predictions = edge_aware_model(node_features, edge_indices, edge_features)
        #edge_aware_predictions = edge_aware_model(node_features, edge_indices, edge_features, edge_meta_index)
        # Forward pass and loss calculation for EdgeAwareGNN
        edge_aware_loss = loss_fn(edge_aware_predictions, node_loads)
        edge_aware_losses.append(edge_aware_loss.item())
        
        # Backward pass and update for EdgeAwareGNN
        edge_aware_loss.backward()    
        optimizer_edgeaware.step()    
        scheduler_edgeaware.step()

edge_aware_predictions = edge_aware_model(node_features, edge_indices, edge_features)
#edge_aware_predictions = edge_aware_model(node_features, edge_indices, edge_features, edge_meta_index)
edge_aware_predictions = edge_aware_predictions.detach().numpy()

actual_node_loads = actual_node_loads.detach().numpy()

edge_aware_r2 = r2_score(actual_node_loads, edge_aware_predictions)
print('Edge Aware GNN R2: ', edge_aware_r2)