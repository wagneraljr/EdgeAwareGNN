import json
import torch
import torch.nn as nn
from src.enums.enum_name_models import EnumNameModel
from src.models.gcn import GCN
from src.models.graph_sage import GraphSAGE
from src.models.edge_aware_gnn.att_edge_aware_gnn import  AttEdgeAwareGCN
from src.models.att_edge_aware_gnn_v2 import AttEdgeAwareGNNV2
from src.utils.data_utils import DataUtils
import os

class TrainUtil:
    @staticmethod
    def predict_by_model(model:nn.Module,node_features:torch.tensor, edge_indices: torch.tensor, 
          edge_features: torch.tensor,):
        if model.__class__ == AttEdgeAwareGCN or model.__class__ == AttEdgeAwareGNNV2:
            return model(node_features, edge_indices, edge_features)
        elif model.__class__ == GCN or model.__class__ == GraphSAGE:
            return model(node_features, edge_indices)
        else:
            raise ValueError(f"model.__class__ {model.__class__} doesn\'t exist")
    
    @staticmethod
    def create_model_by_model_name(model_name, node_feat_size, edge_feat_size, hidden_dim, out_dim, dropout,):
        if model_name == EnumNameModel.ATT_EDGE_AWARE.value: 
            return AttEdgeAwareGCN(node_feat_size, edge_feat_size, hidden_dim, out_dim, dropout)
        elif model_name == EnumNameModel.ATT_EDGE_AWARE_V2.value:
            return AttEdgeAwareGNNV2(node_feat_size, edge_feat_size, hidden_dim, out_dim, dropout)
        elif model_name == EnumNameModel.GCN.value:
            return GCN(node_feat_size, hidden_dim, out_dim, dropout)
        elif model_name == EnumNameModel.GRAPHSAGE.value:
            return GraphSAGE(node_feat_size, hidden_dim, out_dim, dropout)
        else:
            raise ValueError(f"model name {model_name} doesn\'t exist"
    )
        
    def get_hyperparameters_by_json_file(model_name: str, hyper_param_file:str):
        with open(hyper_param_file,'r') as file:
            data = json.load(file)
            data = data[model_name]
            hidden_dim = data['hidden_dim']
            out_dim = data['out_dim']
            dropout = data['dropout']
            lr = data['lr']
            gamma = data['gamma']
            step_size = data["step_size"]
            epochs = data["epochs"]
            seed = data["seed"]
            return hidden_dim,out_dim,dropout,lr,gamma,step_size,epochs, seed

        
    @staticmethod
    def train_model(model:nn.Module,optimizer: torch.optim.Optimizer,loss_fn,
        scheduler: torch.optim.Optimizer,node_features:torch.tensor, edge_indices: torch.tensor, 
        edge_features: torch.tensor, node_loads: torch.tensor):
        optimizer.zero_grad()
        predictions = TrainUtil.predict_by_model(model,node_features,edge_indices,edge_features)
        loss = loss_fn(predictions, node_loads)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss
    
    @staticmethod
    def train_by_path_traffic_matrix(gml_file:str,path_traffic_matrix_files: str, extension_files:str,
                    model:nn.Module,optimizer: torch.optim.Optimizer, epochs: int, loss_fn,
                    scheduler: torch.optim.Optimizer,):
        traffic_matrix_files = sorted([file for file in os.listdir(path_traffic_matrix_files) if file.endswith(extension_files)])
        losses = []
        node_features, edge_indices, edge_features = DataUtils.load_data(gml_file)
        for epoch in range(epochs):
            print(f"Progress:  {epoch/epochs:.2f}",end="\r")
            for tm_file in traffic_matrix_files:
                tm = path_traffic_matrix_files + tm_file
                node_loads = DataUtils.get_node_loads(tm)
                loss = TrainUtil.train_model( model,optimizer, loss_fn,
                    scheduler, node_features, edge_indices, 
                    edge_features,node_loads, )
                losses.append(loss)
        print("\r" + " "*20 + "\r",end='')

        return losses
