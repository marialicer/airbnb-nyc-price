# %%
# importando bibliotecas necessárias 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%

# carregando o df 

df = pd.read_csv("C:/Users/alice/OneDrive/Documentos/airbnb-nyc-price/data/AB_NYC_2019.csv")
# %%
# visualizar o df e as colunas

df.head()
# %%
df.columns
# %%
df.describe()
# %%

df.info()
# %%

# soma de nulos

df.isnull().sum()
# %%
# limpando nulos e verificando mudanças

df.drop(['id', 'host_name','last_review'], axis=1, inplace=True)

df.head()
# %%
df.fillna({'reviews_per_month':0}, inplace=True)

# %%

df.fillna({'name':'unknown'}, inplace=True)

df.isnull().sum()
# %%

# Como os preços se distribuem em NYC?
# plotando histograma

plt.figure(figsize=(8,5))

sns.histplot(
    df["price"],
    kde=True,
    color="#9E219E",
    edgecolor='black'
)

plt.savefig("../img/distribuicao_preco_com_outliers.png")

plt.show()
# %%

# Calculando outliers para melhor plotagem de preços no histograma

# 1. Calcular Q1 (25%) e Q3 (75%)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)

# 2. Calcular o IQR (Interquartile Range)
IQR = Q3 - Q1

# 3. Definir os limites inferior (abaixo do padrão) e superior (acima do padrão)
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# 4. Identificar os outliers
outliers = df[(df['price'] < limite_inferior) | (df['price'] > limite_superior)]

# 5. Verificar % de outliers no dataset

len(outliers) / len(df) * 100

## Cerca de 6% dos registros foram identificados como outliers com base no método IQR
# %%

df_sem_outliers = df[
    (df['price'] >= limite_inferior) & 
    (df['price'] <= limite_superior)
]
# %%

## Plotar gráfico sem outliers

plt.figure(figsize=(8,5))

sns.histplot(
    df_sem_outliers["price"],
    kde=True,
    color="#9E219E",
    edgecolor='black'
)

plt.title('Distribuição de preços (sem outliers)')
plt.xlabel('Preço')
plt.ylabel('Frequência')

plt.savefig("../img/distribuicao_preco_sem_outliers.png")

plt.show()
# %%
