import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from data_utils import load_traffic_pd

# Caminho para a pasta onde os dados foram extraídos
extract_folder = 'Measured/day'
extracted_files = os.listdir(extract_folder)

# Carregar dados de treinamento
train_files = [f for f in extracted_files if '.dat' in f]
train_data_list = [load_traffic_pd(os.path.join(extract_folder, f)) for f in train_files]

# Concatenar todos os DataFrames de treinamento em uma única série temporal
train_data = pd.concat(train_data_list, ignore_index=True).sum(axis=1)  # Somar colunas para obter um total por intervalo

# Definir e ajustar o modelo ARIMA
arima_model = ARIMA(train_data, order=(1, 1, 1))
arima_result = arima_model.fit()

# Carregar dados de teste (arquivo de destino)
test_data_path = 'Data/tm.2004-09-10.16-00-00.dat'
y_true = load_traffic_pd(test_data_path).sum(axis=1)

# Fazer previsões
y_pred = arima_result.forecast(steps=len(y_true))

# Calcular o RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# Calcular o MAE
mae = mean_absolute_error(y_true, y_pred)
# Calcular o MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"ARIMA RMSE: {rmse:.3f}")
print(f"ARIMA MAE: {mae:.3f}")
print(f"ARIMA MAPE: {mape:.3f}")
