# %%
# importando bibliotecas necessárias 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

# identificar e plotar os distritos com maior média de preços

media_precos = df_sem_outliers.groupby('neighbourhood_group', as_index=False)['price'].mean()
media_precos.sort_values('price', ascending=False)
# %%

paleta_cores = sns.color_palette("YlOrBr_r")

ax = sns.barplot(
    data=media_precos,
    x='neighbourhood_group',
    y='price',
    order=['Manhattan', 'Brooklyn', 'Staten Island', 'Queens', 'Bronx'],
    palette=paleta_cores
)

plt.title('Média de preços por distritos')
plt.xlabel('Distritos')
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

# plotar bairros mais caros

media_precos_bairros = df_sem_outliers.groupby('neighbourhood', as_index=False)['price'].mean()
media_precos_bairros.sort_values('price',ascending=False)

# %%

contagem = df_sem_outliers['neighbourhood'].value_counts()

media_precos_bairros['qtd_imoveis'] = media_precos_bairros['neighbourhood'].map(contagem)

# %%

media_precos_bairros.head()

# %%

media_precos_bairros = media_precos_bairros[media_precos_bairros['qtd_imoveis'] > 50]
# %%

top10 = media_precos_bairros.sort_values('price', ascending=False).head(10)

# %%

top10 = top10.sort_values(by='price', ascending=True)

plt.figure(figsize=(10,6))

plt.barh(
    top10['neighbourhood'],
    top10['price'],
    color = 'khaki'
)

for index, value in enumerate(top10['price']):
    plt.text(
        value,      
        index,         
        f'{value:.0f}', 
        va='center'     
    )

plt.title('Top 10 bairros mais caros (com volume relevante)')
plt.xlabel('Preço médio')
plt.ylabel('Bairro')

plt.savefig("../img/media_bairros_caros.png")

plt.show()
# %%

reviews_price = df_sem_outliers.groupby('number_of_reviews', as_index=False)['price'].mean()

# %%

# plotar relação entre reviews e preço

plt.scatter(
    reviews_price['number_of_reviews'],
    reviews_price['price'],
    alpha=0.5,
    color = 'orange'
)

plt.title('Relação entre número de reviews e preço')
plt.xlabel('Número de reviews')
plt.ylabel('Preço médio')

plt.show()
# %%

correlacao = df_sem_outliers['number_of_reviews'].corr(df_sem_outliers['price'])
print(correlacao)
# %%

# perfis de airbnbs (clusters)

df_numerico = df_sem_outliers.drop(columns=[
    'host_id',
    'latitude',
    'longitude'
])

# %%

df_numerico = df_numerico.select_dtypes(include=['int64', 'float64'])

# %%

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numerico)
# %%

kmeans = KMeans(n_clusters=3, random_state=42) # Exemplo com K=3
clusters = kmeans.fit_predict(scaled_data)
# %%

df_sem_outliers['cluster'] = clusters

# %%
centroides = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=df_numerico.columns
)

centroides
# %%

df_sem_outliers.head()
# %%

cores = ['#FFA500', '#FFD700', '#FF4500']

centroides.T.plot(kind='bar', figsize=(12,6), color=cores )
plt.title("Perfis dos clusters")

plt.legend(title='Clusters')
plt.tight_layout()

plt.savefig("../img/centroides_clusters.png")

plt.show()
# %%

# treinar modelo para responder quais variáveis impactam mais no preço

X = df_sem_outliers.drop(columns=[
    'price',
    'host_id',
    'name'
])

y = df_sem_outliers['price']
# %%

X = pd.get_dummies(X, drop_first=True)

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%

modelo = RandomForestRegressor(random_state=42)
modelo.fit(X_train, y_train)
# %%

importancias = pd.Series(
    modelo.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

importancias
# %%
importancias.head(10).plot(kind='barh')
plt.title("Top 10 variáveis que impactam o preço")
plt.show()
# %%

y_pred = modelo.predict(X_test)

# %%
# avaliando o modelo

mae = mean_absolute_error(y_test, y_pred)
print(mae)
# %%

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
# %%

r2 = r2_score(y_test, y_pred)
print(r2)
# %%

# usando os clusters como variáveis

df_sem_outliers['cluster'] = kmeans.labels_
# %%

# pegando as melhores variáveis

X = df_sem_outliers.drop(columns=[
    'price',
    'host_id',
    'name'
])

X = pd.get_dummies(X, drop_first=True)

top_features = importancias.head(10).index.tolist()

# %%

if 'cluster' in X.columns:
    top_features.append('cluster')

# %%

top_features = [col for col in top_features if col in X.columns]

# %%

X = X[top_features]

y =  df_sem_outliers['price']
# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# %%

modelo2 = RandomForestRegressor(random_state=42)
modelo2.fit(X_train, y_train)
# %%

y_pred = modelo2.predict(X_test)
# %%

# avaliando o modelo após seleção de features

mae = mean_absolute_error(y_test, y_pred)
print(mae)
# %%

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
# %%

r2 = r2_score(y_test, y_pred)
print(r2)
# %%


