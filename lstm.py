import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Função para carregar e preparar os dados
def load_prepare_data(file_path):
    data = pd.read_csv(file_path, comment='#', header=None)
    data = data.iloc[1:, 1:] 
    return data.apply(pd.to_numeric, errors='coerce').values

# Função para converter a série em dados supervisionados
def series_to_supervised(data, n_in=1, n_out=1):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # Sequência de entrada (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # Sequência de previsão (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Juntar tudo
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Remover linhas com NaN
    agg.dropna(inplace=True)
    return agg

# Modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0, c0 = self.init_hidden(x.size(0))
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return h0, c0

# Carregar dados, normalizar e preparar para treinamento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = np.vstack([load_prepare_data(f'Measured/day/{f}') for f in train_files])
y_true = load_prepare_data('Data/tm.2004-09-10.16-00-00.dat')

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.sum(axis=1).reshape(-1, 1))
test_scaled = scaler.transform(y_true.sum(axis=1).reshape(-1, 1))

train_reframed = series_to_supervised(train_scaled, 1, 1)
train_X, train_y = train_reframed.values[:, :-1], train_reframed.values[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
train_y = train_y.reshape((train_y.shape[0], 1))

train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
train_y = torch.tensor(train_y, dtype=torch.float32).to(device)

# Definir parâmetros do modelo, otimizador e função de custo
model = LSTMModel(input_dim=1, hidden_dim=50).to(device)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Treinamento do modelo
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(train_X)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()
    #print(f'Epoch {epoch+1} Loss {loss.item()}')

# Previsão e avaliação
model.eval()
test_X = torch.tensor(test_scaled.reshape((test_scaled.shape[0], 1, 1)), dtype=torch.float32).to(device)
predicted = model(test_X)
predicted = predicted.cpu().detach().numpy()
y_pred = scaler.inverse_transform(predicted)

y_true = scaler.inverse_transform(test_scaled)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"LSTM RMSE: {rmse:.3f}")
print(f"LSTM MAE: {mae:.3f}")
print(f"LSTM MAPE: {mape:.3f}")
