import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data_bvsp = pd.read_csv('./data/^BVSP.csv')
data_BRL = pd.read_csv('./data/BRL=X.csv')
data_GOL = pd.read_csv('./data/GOLL4.SA.csv')
data_ouro_tratado = pd.read_csv('./data/Ouro Tratado.csv')
data_petroleo_tratado = pd.read_csv('./data/Petroleo Tratado.csv')


#--- Dados da empresa GOL ---

#convertendo os dados para data e hora
data_GOL['Date'] = pd.to_datetime(data_GOL['Date'])

# Reorganiza os dados em ordem crescente
data_GOL = data_GOL.sort_values('Date')

#Renomeia a coluna 'Date' para 'Data'
data_GOL.rename(columns={'Date': 'Data'}, inplace=True)

#Define o intervalo de dados
primeira_data_GOL = data_GOL['Data'].loc[0]
ultima_data_GOL = data_GOL['Data'].iloc[-1]

#--------------------------------------------------------------------------------------------------------------

#--- Dados da bolsa BOVESPA ---

#convertendo os dados para data e hora
data_bvsp['Date'] = pd.to_datetime(data_bvsp['Date'])

# Reorganiza os dados em ordem crescente
data_bvsp = data_bvsp.sort_values('Date')

#Renomeia a coluna 'Date' para 'Data'
data_bvsp.rename(columns={'Date': 'Data'}, inplace=True)

#--------------------------------------------------------------------------------------------------------------

#--- Ouro ---

#Converte a coluna do Data para o formato de dia/mês/ano
data_ouro_tratado['Data'] = pd.to_datetime(data_ouro_tratado['Data'], format='%d%m%Y', errors='coerce')

# Remove linhas que contêm valores nulos
data_ouro_tratado = data_ouro_tratado.dropna(how='any')

#Organiza o Data baseado na coluna Data organizando-as de forma ascendente
data_ouro_tratado = data_ouro_tratado.sort_values('Data')

#--------------------------------------------------------------------------------------------------------------

#--- Dólar ---

#Converte os dados para data e hora
data_BRL['Date'] = pd.to_datetime(data_BRL['Date'])

#Reorganiza os dados de forma crescente
data_BRL = data_BRL.sort_values('Date')

#Renomeia a coluna 'Date' para 'Data'
data_BRL.rename(columns={'Date': 'Data'}, inplace=True)

#--------------------------------------------------------------------------------------------------------------

#--- Petróleo ---

#Converte a coluna do Data para o formato de dia/mês/ano
data_petroleo_tratado['Data'] = pd.to_datetime(data_petroleo_tratado['Data'],  format='%d%m%Y', errors='coerce')

#Reorganiza o Data de acordo com a ordem crescente
data_petroleo_tratado = data_petroleo_tratado.sort_values('Data')

# Remove linhas que contêm valores nulos
data_petroleo_tratado = data_petroleo_tratado.dropna(how='any')

#--------------------------------------------------------------------------------------------------------------


#Verificação para cada uma das linhas se há valores nulos

# print(data_GOL.isnull().any())
# print(data_bvsp.isnull().any())
# print(data_ouro_tratado.isnull().any())
# print(data_BRL.isnull().any())
# print(data_petroleo_tratado.isnull().any())

#--- X e y --- 

#Torna a coluna Data como primeira
X = data_GOL[['Data']]

#Adiciona a coluna 'Close' do DataFrame data_GOL a 'X', usando a coluna 'Data' como referência.
X = X.merge(data_GOL[['Data', 'Close']], on='Data', how='inner')

#Adiciona a coluna 'Ultimo' do DataFrame data_ouro_tratado a 'X', usando a coluna 'Data' como referência.
X = X.merge(data_ouro_tratado[['Data', 'Ultimo']], on='Data', how='inner')

#Adiciona a coluna 'Ultimo' do DataFrame data_petroleo_tratado a 'X', usando a coluna 'Data' como referência.
X = X.merge(data_petroleo_tratado[['Data', 'Ultimo']], on='Data', how='inner')

#Adiciona a coluna 'Close' do DataFrame data_bvsp a 'X', usando a coluna 'Data' como referência.
X = X.merge(data_bvsp[['Data', 'Close']], on='Data', how='inner')

#Adiciona a coluna 'Close' do DataFrame data_BRL a 'X', usando a coluna 'Data' como referência.
X = X.merge(data_BRL[['Data', 'Close']], on='Data', how='inner')

#Renomeia as colunas do dataFrame "X"
X.columns = ['Data', 'Close_GOL', 'Ultimo_ouro', 'Ultimo_petroleo', 'Ultimo_bovespa', 'Ultimo_dólar']

#Remove de "X" qualquer linha que contêm valores nulos
X = X.dropna(how='any')
Z = X
print(X)

#--------------------------------------------------------------------------------------------------------------

#Definição de y, após definição de X para garantir que o tamanho de X e y sejam os mesmos, e obedeçam às datas em todas as entradas 
y = X['Close_GOL']

#--------------------------------------------------------------------------------------------------------------

# Obtendo apenas as caracteristicas de cada coluna
X = X[['Ultimo_ouro', 'Ultimo_petroleo', 'Ultimo_bovespa', 'Ultimo_dólar']]

# Definição da massa de teste e treino
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size= 0.25)

linear_regression = LinearRegression()

#Aplicando regressão linear nos dados de treino
linear_regression.fit(X_treino, y_treino)
y_prev = linear_regression.predict(X_teste)

#--------------------------------------------------------------------------------------------------------------

#Respostas
print(y_prev)
print(np.sqrt(metrics.mean_squared_error(y_teste, y_prev)))


#Representação visual

sns.pairplot(Z, x_vars=['Ultimo_ouro', 'Ultimo_petroleo', 'Ultimo_bovespa', 'Ultimo_dólar'], y_vars='Close_GOL', height=5, kind='reg')

plt.show()