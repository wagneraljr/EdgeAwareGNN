import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from data_utils import load_traffic_pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Caminho para a pasta onde os dados foram extraídos
extract_folder = 'Measured/day'
extracted_files = os.listdir(extract_folder)

# Carregar dados de treinamento
train_files = [f for f in extracted_files if '.dat' in f]
train_data_list = [load_traffic_pd(os.path.join(extract_folder, f)) for f in train_files]

# Concatenar todos os DataFrames de treinamento em uma única série temporal
train_data = pd.concat(train_data_list, ignore_index=True).sum(axis=1)  # Somar colunas para obter um total por intervalo

# Preparar dados para o modelo SVR
X_train = np.arange(len(train_data)).reshape(-1, 1)
y_train = train_data.values

# Definir e ajustar o modelo SVR com pipeline incluindo escalonamento
svr_model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1))
svr_model.fit(X_train, y_train)

# Carregar dados de teste (arquivo de destino)
y_true_path = 'Data/tm.2004-09-10.16-00-00.dat'
y_true = load_traffic_pd(y_true_path).sum(axis=1)
X_test = np.arange(len(train_data), len(train_data) + len(y_true)).reshape(-1, 1)

# Fazer previsões
y_pred = svr_model.predict(X_test)

# Calcular métricas
svr_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
svr_mae = mean_absolute_error(y_true, y_pred)
svr_mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"SVR RMSE: {svr_rmse:.3f}")
print(f"SVR MAE: {svr_mae:.3f}")
print(f"SVR MAPE: {svr_mape:.3f}")
