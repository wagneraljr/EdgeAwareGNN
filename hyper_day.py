import torch
import torch.nn as nn
from models import AttEdgeAwareGCN, GCN, GraphSAGE
from data_utils import load_data, get_node_loads
import numpy as np
import os
import optuna
from torch.optim.lr_scheduler import StepLR
import logging
from sklearn.metrics import r2_score

def objective_gcn(trial):
    try:
        seed = 4096
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Carrega o grafo da rede Abilene a partir do arquivo .gml  
        # e a matriz de tráfego com os valores de carga reais
        node_features, edge_indices, edge_features = load_data("Abilene.gml")
        actual_node_loads = get_node_loads("Data/tm.2004-09-10.16-00-00.dat")
            
        # Hyperparâmetros
        hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 0.001, 0.01)
        num_epochs = trial.suggest_categorical('num_epochs', [100, 200, 300])
        step_size = trial.suggest_categorical('step_size', [25, 50, 75])
        gamma = trial.suggest_float('gamma', 0.1, 0.99)
        
        # Inicializa o modelo
        model = GCN(node_features.size(1), hidden_dim, 1, dropout)

        # Inicializa o otimizador e o escalonador
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size, gamma)
        
        # Define a função de perda
        loss_fn = nn.MSELoss()
        
        # Lê os valores históricos das matrizes de tráfego
        traffic_matrix_files = sorted([file for file in os.listdir("./Measured/day") if file.endswith('.dat')])

        # Loop de treino
        for epoch in range(num_epochs):  
            for traffic_matrix_filepath in traffic_matrix_files:
                tm = "Measured/day/" + traffic_matrix_filepath
                node_loads = get_node_loads(tm)
                                             
                # Zera gradientes
                optimizer.zero_grad()
                
                # Etapa forward
                predictions = model(node_features, edge_indices)
                
                # Calcula perda
                loss = loss_fn(predictions, node_loads)
                
                # Passagem Backward, steps de otimizador e escalonador
                loss.backward()
                optimizer.step()
                scheduler.step()  
        
        # A cada epoch, armazena o valor intermediário da perda
        intermediate_value = loss.item()
        trial.report(intermediate_value, epoch)

        # Verifica se o a etapa do teste deve ser podada
        # (não funcionou bem, mas fica comentado para referência)
        #if trial.should_prune():
        #    raise optuna.TrialPruned()

        # Avaliação do modelo - No exemplo estou usando o R2 e tentando maximizar o valor
        # É possível usar outros critérios, como a perda e tentar minimizar
        with torch.no_grad():
            #val_predictions = model(node_features, edge_indices, edge_features)
            val_predictions = model(node_features, edge_indices)
            #val_loss = loss_fn(val_predictions, actual_node_loads)

            actual_node_loads = actual_node_loads.detach().numpy()
            val_predictions = val_predictions.detach().numpy()

            r2 = r2_score(actual_node_loads, val_predictions)

    #        return val_loss
            return r2
    
    except AttributeError as e:
        print(f"An AttributeError occurred: {e}")
       

def objective_gsage(trial):
    try:
        seed = 4096
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Carrega o grafo da rede Abilene a partir do arquivo .gml  
        # e a matriz de tráfego com os valores de carga reais
        node_features, edge_indices, edge_features = load_data("Abilene.gml")
        actual_node_loads = get_node_loads("Data/tm.2004-09-10.16-00-00.dat")
            
        # Hyperparâmetros
        hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 0.001, 0.01)
        num_epochs = trial.suggest_categorical('num_epochs', [100, 200, 300])
        step_size = trial.suggest_categorical('step_size', [25, 50, 75])
        gamma = trial.suggest_float('gamma', 0.1, 0.99)
        
        # Inicializa o modelo
        model = GraphSAGE(node_features.size(1), hidden_dim, 1, dropout)

        # Inicializa o otimizador e o escalonador
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size, gamma)
        
        # Define a função de perda
        loss_fn = nn.MSELoss()
        
        # Lê os valores históricos das matrizes de tráfego
        traffic_matrix_files = sorted([file for file in os.listdir("./Measured/day") if file.endswith('.dat')])

        # Loop de treino
        for epoch in range(num_epochs):  
            for traffic_matrix_filepath in traffic_matrix_files:
                tm = "Measured/day/" + traffic_matrix_filepath
                node_loads = get_node_loads(tm)
                             
                # Zera gradientes
                optimizer.zero_grad()
                
                # Etapa forward
                predictions = model(node_features, edge_indices)
                
                # Calcula perda
                loss = loss_fn(predictions, node_loads)
                
                # Passagem Backward, steps de otimizador e escalonador
                loss.backward()
                optimizer.step()
                scheduler.step()  
        
        # A cada epoch, armazena o valor intermediário da perda
        intermediate_value = loss.item()
        trial.report(intermediate_value, epoch)

        # Verifica se o a etapa do teste deve ser podada
        # (não funcionou bem, mas fica comentado para referência)
        #if trial.should_prune():
        #    raise optuna.TrialPruned()

        # Avaliação do modelo - No exemplo estou usando o R2 e tentando maximizar o valor
        # É possível usar outros critérios, como a perda e tentar minimizar
        with torch.no_grad():
            #val_predictions = model(node_features, edge_indices, edge_features)
            val_predictions = model(node_features, edge_indices)
            #val_loss = loss_fn(val_predictions, actual_node_loads)

            actual_node_loads = actual_node_loads.detach().numpy()
            val_predictions = val_predictions.detach().numpy()

            r2 = r2_score(actual_node_loads, val_predictions)

    #        return val_loss
            return r2
    
    except AttributeError as e:
        print(f"An AttributeError occurred: {e}")

def objective_eagnn(trial):
    try:
        seed = 4096
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Carrega o grafo da rede Abilene a partir do arquivo .gml  
        # e a matriz de tráfego com os valores de carga reais
        node_features, edge_indices, edge_features = load_data("Abilene.gml")
        actual_node_loads = get_node_loads("Data/tm.2004-09-10.16-00-00.dat")
            
        # Hyperparâmetros
        hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 0.001, 0.01)
        num_epochs = trial.suggest_categorical('num_epochs', [100, 200, 300])
        step_size = trial.suggest_categorical('step_size', [25, 50, 75])
        gamma = trial.suggest_float('gamma', 0.1, 0.99)
        
        # Inicializa o modelo
        model = AttEdgeAwareGCN(node_features.size(1), edge_features.size(1), hidden_dim, 1, dropout)       

        # Inicializa o otimizador e o escalonador
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size, gamma)
        
        # Define a função de perda
        loss_fn = nn.MSELoss()
        
        # Lê os valores históricos das matrizes de tráfego
        traffic_matrix_files = sorted([file for file in os.listdir("./Measured/day") if file.endswith('.dat')])
      
        # Loop de treino
        for epoch in range(num_epochs):  
            for traffic_matrix_filepath in traffic_matrix_files:
                tm = "Measured/day/" + traffic_matrix_filepath
                node_loads = get_node_loads(tm)
                             
                # Zera gradientes
                optimizer.zero_grad()
                
                # Passagem forward
                predictions = model(node_features, edge_indices, edge_features)
                
                # Calcula perda
                loss = loss_fn(predictions, node_loads)
                
                # Passagem Backward, steps de otimizador e escalonador
                loss.backward()
                optimizer.step()
                scheduler.step()  
        
        # A cada epoch, armazena o valor intermediário da perda
        intermediate_value = loss.item()
        trial.report(intermediate_value, epoch)

        # Verifica se o a etapa do teste deve ser podada
        # (não funcionou bem, mas fica comentado para referência)
        #if trial.should_prune():
        #    raise optuna.TrialPruned()

        # Avaliação do modelo - No exemplo estou usando o R2 e tentando maximizar o valor
        # É possível usar outros critérios, como a perda e tentar minimizar
        with torch.no_grad():
            #val_predictions = model(node_features, edge_indices, edge_features)
            val_predictions = model(node_features, edge_indices)
            #val_loss = loss_fn(val_predictions, actual_node_loads)

            actual_node_loads = actual_node_loads.detach().numpy()
            val_predictions = val_predictions.detach().numpy()

            r2 = r2_score(actual_node_loads, val_predictions)

    #        return val_loss
            return r2
    
    except AttributeError as e:
        print(f"An AttributeError occurred: {e}")
        

# Create an Optuna study and run optimization
if __name__ == "__main__":    

    logger = optuna.logging.get_logger('optuna')
    logger.setLevel(logging.INFO)
    # Alterar o nome do arquivo de acordo com o valor da seed
    file_handler = logging.FileHandler('optuna_logs.txt.gcn.4096.day')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
 
    study = optuna.create_study(direction='maximize')

    # Alterar conforme o modelo a ser usado
    study.optimize(objective_gcn, n_trials=100)

    print(f"Testes concluídos: {len(study.trials)}")
    # Imprime os parâmetros do melhor teste
    print(f"Melhor teste: {study.best_trial.params}")
