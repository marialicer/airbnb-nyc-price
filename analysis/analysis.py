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
