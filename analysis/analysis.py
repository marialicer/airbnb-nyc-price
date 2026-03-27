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
