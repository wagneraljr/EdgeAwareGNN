import sys
import torch
import torch.nn as nn
from src.constants.k_paths import KPaths
from src.enums.enum_name_models import EnumNameModel
from src.models.gcn import GCN
from src.models.graph_sage import GraphSAGE
from src.models.edge_aware_gnn.att_edge_aware_gnn import  AttEdgeAwareGCN
from src.models.att_edge_aware_gnn_v2 import AttEdgeAwareGNNV2
from src.utils.data_utils import DataUtils
import numpy as np
from torch.optim.lr_scheduler import StepLR
from src.utils.train_util import TrainUtil
import json
from colorama import Fore, Style

def train(model_name,period):
    hidden_dim, out_dim,dropout, lr,gamma, step_size, epochs, seed = TrainUtil.get_hyperparameters_by_json_file(
        model_name, KPaths.path_hyperparameters + f'hyperparams_{period}.json')

    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    loss_fn = nn.MSELoss()
    gml_file = KPaths.path_data+"abilene/Abilene.gml"
    tm_test = KPaths.path_data + "abilene/target/test/tm.2004-09-10.16-00-00.dat"
    
    node_features, edge_indices, edge_features = DataUtils.load_data(gml_file)
    model = TrainUtil.create_model_by_model_name(model_name,node_features.size(1),edge_features.size(1),hidden_dim,out_dim,dropout)
    optim = torch.optim.Adam(list(model.parameters()), lr=lr)
    scheduler = StepLR(optim, step_size=step_size, gamma=gamma)

    path_traffic_matrix_files_train = KPaths.path_data + f"abilene/target/train/{period}/" 
    #TODO: manter as losses durante todo o treinamento usa muita memória. Implementar outra abordagem.
    losses = TrainUtil.train_by_path_traffic_matrix(gml_file, path_traffic_matrix_files_train,'.dat',
                                                         model,optim,epochs,loss_fn,scheduler)
    
    prediction = TrainUtil.predict_by_model(model,node_features, edge_indices, edge_features)
    results = {
        'losses': [tensor.item() for tensor in losses],
        'predictions': prediction.detach().numpy().tolist()
    }
    with open(KPaths.path_results + f'/{period}/{model_name}_train_results_{period}.json','w') as file:
        json.dump(results, file)
        
    torch.save(model,KPaths.path_checkcpoints+model_name+f'_{period}.pth') 

 
if __name__ == "__main__":
    args = sys.argv
    model_name = args[1]
    period = args[2]
    train(model_name, period)
    print(Fore.GREEN + 'Completed\n' + Style.RESET_ALL)
