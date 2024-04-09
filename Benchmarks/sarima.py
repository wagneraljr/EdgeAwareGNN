import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from data_utils import load_traffic_pd

# Caminho para a pasta onde os dados foram extraídos
extract_folder = 'Measured/day/'
extracted_files = os.listdir(extract_folder)

# Carregar dados de treinamento
train_files = [f for f in extracted_files if '.dat' in f]
train_data_list = [load_traffic_pd(os.path.join(extract_folder, f)) for f in train_files]

# Concatenar todos os DataFrames de treinamento em uma única série temporal
train_data = pd.concat(train_data_list, ignore_index=True).sum(axis=1)  # Somar colunas para obter um total por intervalo

# Definir e ajustar o modelo SARIMA
# Observação: (1, 1, 1) são os parâmetros não sazonais; (1, 1, 1, 12) são os parâmetros sazonais com 12 como período sazonal
sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 48))
sarima_result = sarima_model.fit()

# Carregar dados de teste
y_true_path = 'Data/tm.2004-09-10.16-00-00.dat'
y_true = load_traffic_pd(y_true_path).sum(axis=1)
X_test_indices = np.arange(len(train_data), len(train_data) + len(y_true))

# Fazer previsões
y_pred = sarima_result.get_forecast(steps=len(y_true))
y_pred_values = y_pred.predicted_mean

# Calcular métricas
sarima_rmse = np.sqrt(mean_squared_error(y_true, y_pred_values))
sarima_mae = mean_absolute_error(y_true, y_pred_values)
sarima_mape = mean_absolute_percentage_error(y_true, y_pred_values)

print(f"SARIMA RMSE: {sarima_rmse:.3f}")
print(f"SARIMA MAE: {sarima_mae:.3f}")
print(f"SARIMA MAPE: {sarima_mape:.3f}")
print('')
