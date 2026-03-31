# %%
# importando bibliotecas necessárias 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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
    color="#f56505",
    edgecolor='black'
)

plt.title('Distribuição de preços (com outliers)')
plt.xlabel('Preço')
plt.ylabel('Frequência')

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
    color="#f56505",
    edgecolor='black'
)

plt.title('Distribuição de preços (sem outliers)')
plt.xlabel('Preço')
plt.ylabel('Frequência')

plt.savefig("../img/distribuicao_preco_sem_outliers.png")

plt.show()
# %%

# identificar e plotar os bairros com maior média de preços

media_precos = df.groupby('neighbourhood_group', as_index=False)['price'].mean()
media_precos.head().sort_values('price', ascending=False)
# %%

paleta_cores = sns.color_palette("YlOrBr_r")

ax = sns.barplot(
    data=media_precos,
    x='neighbourhood_group',
    y='price',
    order=['Manhattan', 'Brooklyn', 'Staten Island', 'Queens', 'Bronx'],
    palette=paleta_cores
)

plt.title('Média de preços por bairros')
plt.xlabel('Bairros')
plt.ylabel('Média de preços')


for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width() / 2,
        height + 0.1,
        f'{height:.1f}',
        ha='center'
    )

plt.savefig("../img/media_precos_bairros.png")

plt.show()
# %%

#plotar preço por latitude e longitude

plt.figure(figsize=(10, 8))

plt.scatter(
    df_sem_outliers['longitude'],
    df_sem_outliers['latitude'],
    c=df_sem_outliers['price'],
    cmap='YlOrBr',
    s=10
)

plt.colorbar(label='Preço ($)')

plt.title('Distribuição geográfica dos preços de Airbnb em NYC')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.savefig("../img/mapa_precos_geografico.png")

plt.show()
# %%


