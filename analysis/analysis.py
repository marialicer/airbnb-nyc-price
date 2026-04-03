# %%
# importando bibliotecas necessárias 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
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

features = ['price', 'minimum_nights', 'number_of_reviews',
            'reviews_per_month', 'calculated_host_listings_count',
            'availability_365', 'latitude', 'longitude']

df_cluster = df_sem_outliers[features]

# %%

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cluster)

# %%

# descobrindo o melhor número de clusters

# Método do Cotovelo

inercia = []

K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inercia.append(kmeans.inertia_)

plt.plot(K, inercia, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo')
plt.show()

# %%

# Método Silhueta

#for k in range(2, 10):
    #kmeans = KMeans(n_clusters=k, random_state=42)
    #labels = kmeans.fit_predict(scaled_data)
    #score = silhouette_score(scaled_data, labels)
    #print(f"k={k} → silhouette score: {score:.4f}")

# %%

kmeans = KMeans(n_clusters=3, random_state=42) 
clusters = kmeans.fit_predict(scaled_data)
# %%

df_sem_outliers['cluster'] = clusters

# %%
centroides = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=df_cluster.columns
)

centroides
# %%

# renomear clusters com base nos centroides

df_sem_outliers['cluster'] = df_sem_outliers['cluster'].map({
    0: 'Populares/Acessíveis',
    1: 'Parados/Inativos',
    2: 'Premium/Profissional'
})

df_sem_outliers.head()

# %%

centroides.index = ['Populares/Acessíveis', 'Parados/Inativos', 'Premium/Profissional']

cores = ['#FFA500', '#FFD700', '#FF4500']

centroides.T.plot(kind='bar', figsize=(12,6), color=cores )
plt.title("Perfis dos clusters")

plt.legend(title='Clusters')
plt.tight_layout()

plt.savefig("../img/centroides_clusters.png")

plt.show()

# %%

contagem_imoveis = df_sem_outliers['cluster'].value_counts()
print(contagem_imoveis)

# %%

sns.scatterplot(
    data=df_sem_outliers,
    x='longitude',
    y='latitude',
    hue='cluster',
    style='cluster',
    palette='viridis',
    s=40
)

plt.title('Clusters de Airbnbs em NYC')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()
# %%

# correlação entre variáveis

numeric_df = df.select_dtypes(include=["int64","float64"])
# %%

plt.figure(figsize=(12,8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="YlOrBr"
)

plt.title("Mapa de calor de correlação entre variáveis")

plt.tight_layout()

plt.savefig("../img/mapa_de_calor_correlacao.png")

plt.show()
# %%

# treinando modelo DBSCAN para comparar resultados

k = 9
nn = NearestNeighbors(n_neighbors=k)
nn.fit(scaled_data)
distancias, _ = nn.kneighbors(scaled_data)

distancias_k = distancias[:, -1]

distancias_ordenadas = np.sort(distancias_k)

plt.figure(figsize=(8,5))
plt.plot(distancias_ordenadas)
plt.title('Gráfico k-NN para escolha do eps')
plt.xlabel('Pontos ordenados')
plt.ylabel(f'Distância ao {k}º vizinho')
plt.grid(True)
plt.show()

# %%

# zoom na região do cotovelo para escolher melhor
plt.figure(figsize=(8,5))
plt.plot(distancias_ordenadas)
plt.title('Gráfico k-NN - zoom no cotovelo')
plt.xlabel('Pontos ordenados')
plt.ylabel(f'Distância ao 9º vizinho')
plt.ylim(0, 5)      
plt.xlim(38000, 46000)  
plt.grid(True)
plt.show()

# %%

#for eps in [1.5, 2.0, 2.5, 3.0]:
    #labels = DBSCAN(eps=eps, min_samples=9).fit_predict(scaled_data)
    #n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    #n_outliers = (labels == -1).sum()
    #print(f"eps={eps} → clusters: {n_clusters} | outliers: {n_outliers}")

# %%

# dbscan = DBSCAN(eps=2.0, min_samples=9)
# labels = dbscan.fit_predict(scaled_data)

# df_sem_outliers['cluster_dbscan'] = labels

# print(df_sem_outliers['cluster_dbscan'].value_counts())
# %%
# sns.scatterplot(
#     data=df_sem_outliers,
#     x='longitude',
#     y='latitude',
#     hue='cluster_dbscan',
#     style='cluster_dbscan',
#     palette='viridis',
#     s=40
# )
# %%

# features_geo = ['latitude', 'longitude']
# scaled_geo = StandardScaler().fit_transform(df_sem_outliers[features_geo])

# labels = DBSCAN(eps=0.1, min_samples=9).fit_predict(scaled_geo)
# print(pd.Series(labels).value_counts())
# %%

# DBSCAN - Tentativa de clustering 
# 
# Parâmetros testados:
#   min_samples = 9 (n_features + 1)
#   eps testados: 1.0, 1.5, 2.0, 2.5, 3.0 (via gráfico k-NN)
#
# Resultado: em todos os cenários, ~94-98% dos dados foram alocados
# num único cluster, indicando que os imóveis estão distribuídos de 
# forma homogênea pela cidade, sem regiões de densidade claramente distintas.
#
# Conclusão: o DBSCAN não é adequado para esse dataset.
# O K-Means com k=3 foi mantido como modelo final, validado pelo
# método do cotovelo e pela silhueta (score=0.2662).