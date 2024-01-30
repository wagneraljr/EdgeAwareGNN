import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models import GCN, GraphSAGE, EdgeAwareGCN, AttEdgeAwareGCN
from data_utils import load_data, generate_edge_meta_index, generate_edge_meta_index_with_self_loops
import numpy as np
import scipy.sparse as sp
import os
from torch.optim.lr_scheduler import StepLR
#from censnet import GCN_cens, create_T_matrix, create_edge_adjacency
from sklearn.metrics import mean_absolute_error, mean_squared_error

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Load the Abilene network graph
node_features, edge_indices, edge_features, \
    actual_node_loads = load_data("Abilene.gml", "Data/tm.2004-09-10.16-00-00.dat")

# Defining loss function
loss_fn = nn.MSELoss()
num_epochs = 200

#edge_meta_index = generate_edge_meta_index(edge_indices)
#edge_meta_index = generate_edge_meta_index_with_self_loops(edge_indices)

# torch.manual_seed(seed)
# np.random.seed(seed)

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

# window_size = 5
for epoch in range(num_epochs):
#     # # print(f'EAGNN Epoch: {epoch+1}')
#     for idx in range(0, len(traffic_matrix_files) - window_size + 1):
#         # Initialize accumulators with zeros
#         node_loads_accumulator = torch.zeros(11)#_like(node_features[0])  # Assuming node_loads has same number of rows as node_features and is a 1D tensor
#         #print("Shape of node_loads_accumulator:", node_loads_accumulator.shape)
#         #print(f"  Processing window starting at index {idx}")  # Print current window's starting index
#         for file_to_use in sorted(traffic_matrix_files)[idx:idx+window_size]:
#             #print(f"    Processing file: {file_to_use}")  # Print current file being processed
#               node_loads_curr = get_node_loads(file_to_use)
#               node_loads_curr = node_loads_curr.squeeze()
#               #print("Shape of node_loads_curr from the first file:", node_loads_curr.shape)
#               # print('CURR LOADS')
#               # print(node_loads_curr)
#               # Add data to accumulators
#               node_loads_accumulator += node_loads_curr
#               # print('ACC LOADS')
#               # print(node_loads_accumulator)
             
#         # Compute the average
#         node_loads_avg = node_loads_accumulator / window_size
#         # print('AVG:')
#         # print(node_loads_avg)
#         # Training step using averaged data
#         optimizer_edgeaware.zero_grad()
#         predictions = edge_aware_model(node_features, adj, edge_indices, edge_features)
#         loss = loss_fn(predictions, node_loads_avg)
#         #print(f"    Loss: {loss.item():.4f}")  # Print current loss
#         loss.backward()
#         optimizer_edgeaware.step()
#         scheduler_edgeaware.step()
        
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

torch.manual_seed(seed)
np.random.seed(seed)

gcn_model = GCN(node_features.size(1), 64, 1)
optimizer_gcn = torch.optim.Adam(list(gcn_model.parameters()), lr=0.01)
scheduler_gcn = StepLR(optimizer_gcn, step_size=50, gamma=0.9)

gcn_losses = []

for epoch in range(num_epochs):
    # print(f'GCN Epoch: {epoch+1}')
    for traffic_matrix_filepath in traffic_matrix_files:
        node_features, edge_indices, edge_features, node_loads = load_data("Abilene.gml", traffic_matrix_filepath)
        
        optimizer_gcn.zero_grad()
        # Forward pass and loss calculation for GCN
        #gcn_predictions = gcn_model(node_features, adj)
        gcn_predictions = gcn_model(node_features, edge_indices)
        gcn_loss = loss_fn(gcn_predictions, node_loads)
        gcn_losses.append(gcn_loss.item())
    
        # Backward pass and update for GCN
        gcn_loss.backward() 
        optimizer_gcn.step()
        scheduler_gcn.step()

#gcn_predictions = gcn_model(node_features, adj)
gcn_predictions = gcn_model(node_features, edge_indices)
gcn_predictions = gcn_predictions.detach().numpy()

torch.manual_seed(seed)
np.random.seed(seed)

graphsage_model = GraphSAGE(node_features.size(1), 64, 1)
optimizer_graphsage = torch.optim.Adam(list(graphsage_model.parameters()), lr=0.01)
scheduler_graphsage = StepLR(optimizer_graphsage, step_size=50, gamma=0.9)

graphsage_losses = []

for epoch in range(num_epochs):
    # print(f'GraphSAGE Epoch: {epoch+1}')
    for traffic_matrix_filepath in traffic_matrix_files:
        # Load the Abilene network graph and respective data from the current traffic matrix file
        node_features, edge_indices, edge_features, node_loads = load_data("Abilene.gml", traffic_matrix_filepath)
       
        optimizer_graphsage.zero_grad()
 
        # Forward pass and loss calculation for GraphSAGE
        graphsage_predictions = graphsage_model(node_features, edge_indices)
        graphsage_loss = loss_fn(graphsage_predictions, node_loads)
        graphsage_losses.append(graphsage_loss.item())
        
        # Backward pass and update for GraphSAGE
        graphsage_loss.backward()
        
        optimizer_graphsage.step()
        scheduler_graphsage.step()

graphsage_predictions = graphsage_model(node_features, edge_indices)
graphsage_predictions = graphsage_predictions.detach().numpy()

actual_node_loads = actual_node_loads.detach().numpy()

def plot_loss_curves(losses_dict):
    plt.figure(figsize=(10, 6))
    for model_name, loss_values in losses_dict.items():
        plt.plot(loss_values, label=model_name)
    plt.title('Loss Curve Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.xlim(0,100)
    plt.legend()
    plt.grid(True)
    plt.show()

losses_dict = {
    'EdgeAwareGNN': edge_aware_losses,
    'GCN': gcn_losses,
    'GraphSAGE': graphsage_losses
}
plot_loss_curves(losses_dict)


predictions_dict = {
    'EdgeAwareGNN': edge_aware_predictions,
    'GCN': gcn_predictions,
    'GraphSAGE': graphsage_predictions
}

def plot_predicted_vs_actual(predictions_dict, actual_values):
    plt.figure(figsize=(15, 6))
    for idx, (model_name, predicted_values) in enumerate(predictions_dict.items(), 1):
        plt.subplot(1, len(predictions_dict), idx)
        plt.scatter(actual_values, predicted_values, alpha=0.5)
        plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'r--')
        plt.title(f'Actual vs Predicted for {model_name}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_predicted_vs_actual(predictions_dict, actual_node_loads)


# After training on all .dat files, evaluate and compare the performance of the models
# This can involve computing metrics on test data or visualizing the training losses.
from sklearn.metrics import r2_score

edge_aware_r2 = r2_score(actual_node_loads, edge_aware_predictions)
print('Edge Aware GNN R2: ', edge_aware_r2)
graphsage_r2 = r2_score(actual_node_loads, graphsage_predictions)
print('GraphSAGE R2: ', graphsage_r2)
gcn_r2 = r2_score(actual_node_loads, gcn_predictions)
print('GCN R2: ', gcn_r2)

#Plot R2 Score
plt.figure(figsize=(10, 5))
plt.bar(['Edge-Aware GNN', 'GraphSAGE', 'GCN'], [edge_aware_r2, graphsage_r2, gcn_r2])
plt.title('R2 Score Comparison')
plt.ylabel('R2 Score')
plt.show()

# Create a dictionary to hold the metrics for each model
metrics_dict = {}

# Helper function to compute metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'MAE': mae, 'RMSE': rmse}

# Compute metrics for each model and add to dictionary
metrics_dict['EdgeAwareGNN'] = compute_metrics(actual_node_loads, edge_aware_predictions)
metrics_dict['GCN'] = compute_metrics(actual_node_loads, gcn_predictions)
metrics_dict['GraphSAGE'] = compute_metrics(actual_node_loads, graphsage_predictions)

# Function to plot metrics
def plot_metrics(metrics_dict):
    labels = list(metrics_dict.keys())
    mae_scores = [metrics['MAE'] for metrics in metrics_dict.values()]
    rmse_scores = [metrics['RMSE'] for metrics in metrics_dict.values()]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    ax.bar(x, mae_scores, width, label='MAE')
    ax.bar(x + width, rmse_scores, width, label='RMSE')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Models')
    ax.set_title('Metrics comparison among different models')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    fig.tight_layout()
    plt.show()

# Plot the metrics
plot_metrics(metrics_dict)

print('')

# Compute additional metrics for Edge Aware GNN
edge_aware_mae = mean_absolute_error(actual_node_loads, edge_aware_predictions)
edge_aware_rmse = np.sqrt(mean_squared_error(actual_node_loads, edge_aware_predictions))
print(f'Edge Aware GNN MAE: {edge_aware_mae}, RMSE: {edge_aware_rmse}')

graphsage_mae = mean_absolute_error(actual_node_loads, graphsage_predictions)
graphsage_rmse = np.sqrt(mean_squared_error(actual_node_loads, graphsage_predictions))
print(f'GraphSAGE MAE: {graphsage_mae}, RMSE: {graphsage_rmse}')

gcn_mae = mean_absolute_error(actual_node_loads, gcn_predictions)
gcn_rmse = np.sqrt(mean_squared_error(actual_node_loads, gcn_predictions))
print(f'GCN MAE: {gcn_mae}, RMSE: {gcn_rmse}')

torch.save(edge_aware_model, 'models/edge_aware_model.pth')
torch.save(gcn_model, 'models/gcn_model.pth')
torch.save(graphsage_model, 'models/graphsage_model.pth')
