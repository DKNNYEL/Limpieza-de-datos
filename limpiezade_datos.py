# -*- coding: utf-8 -*-

## Taller Python para la limpieza de Datos

### Grow Up Data ds.
Hablemos un poco de como se comporta un ciclo de vida para un trabajo en Ciencia de Datos


from IPython.core.display import Image
Image('Flujo.png',width=500, height=400)

"""Antes de iniciar con cualquier analisis de datos ten en claro que pregunta del negocio estas trantando de responder o define la ruta de lo que estas intentando hacer!!

### Importamos los paquetes y archivos a trabajar en el Análisis
"""

## Importando Paquetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path=r'C:\Users\esteb\OneDrive\Documents\Python Scripts\Video_Games.csv'
df = pd.read_csv(path,sep=';',header=0,index_col=False,encoding='latin-1',engine='python')

"""### 2. Exploracion de Datos

Conozcamos la base de datos con la cual trabajaremos, entedamos la estructura y las dimensiones
"""

##Muestra las primeras 5 filas
df.head()

##Muestra las ultimas 5 filas
df.tail()

##Muestra el nombre de las columnas y el tipo de datos que contienen
df.info()

### Cantidad de Observaciones y Variables de nuestro DataSet
df.shape

##Estadisticas para campos numericos
df.describe()

### Obteniendo los nombres de las columnas
df.columns

## Nos da los valores de mi dataset en una matriz numpy
df.values

"""### Data Cleasing

Manipulemos y transformemos nuestros Dataset
"""

###Duplicando mi Dataframe
df2 = df.copy(deep=True)
df2.head()

### Contamos la cantidad de duplicados
print(df2.duplicated().sum())

### Revisamos los valores Duplicados
df_dp = df2.duplicated(keep = False)
print(df2[df_dp])

#### Eliminando los duplicados
df2.drop_duplicates(keep = 'first', inplace = True)
print(df2.shape)

### Revisamos cuales son los valores nulos tengo en mi DataFrame
df2.isna().any()

### Validamos cuantos campos dentros de cada variables tiene vacios
df2.isna().sum()

#### Creamos el bool que nos serivirá para identificar los valores vacios
df_na = df2['Publisher'].isna()
print(df_na)
print(df2[df_na].shape)

### Eliminamos los valores Vacios
df2.dropna(subset=['Publisher'],inplace = True)
print(df2.shape)
print(df2.isna().sum())

### Rellenando los valores vacios con 0
df_replace = df2[['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']].fillna(0)
print(df_replace)
print(df_replace.isna().sum())

###Duplicando mi Dataframe
df3 = df2.copy(deep=True)
df3.head()
df3.shape

df3[['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']] = df_replace
print(df3.isna().sum())

### Creamos un DataFrame con las columnas a reemplazar por la media
df_fill = df2[['Critic_Score','Critic_Count']]
print(df_fill.head())

### Aplicamos una funcion sencilla anonima para que recorra cada columna de este nuevo DataFrame y 
#la reemplace por la media de la misma
df_fill2 = df_fill.apply(lambda x: x.fillna(x.mean()),axis=0)
print(df_fill2.head())

### Reemplazamos las columnas con NaN por las nuevas columnas corregidas
df3[['Critic_Score','Critic_Count']] = df_fill2
print(df3.isna().sum())

# =============================================================================
# Correlacion
# =============================================================================

corr = df3.set_index('Year_of_Release').corr()
print(corr)

# Excluimos las correlaciones duplicadas
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Definimos el Tamaño de la Figura
f, ax = plt.subplots(figsize=(11, 9))

# Generamos la paleta de color
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Creamos un mapa de calor con el Mask correcto
sns.heatmap(corr, mask=mask, cmap=cmap,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

### Agrupemos nuestros Datos
df_group = df3.groupby('Publisher')['Global_Sales'].agg(sum).reset_index()
print(df_group)

### Ordenando nuestro Dataframe
df_group.sort_values('Global_Sales',ascending=False,inplace=True)
print(df_group)

### Seleccionamos el Top 5
df_group = df_group.iloc[0:5]
print(df_group)

#tamaño de cada barra
width = 0.35

fig, ax = plt.subplots()

rects1 = ax.bar(df_group['Publisher'],df_group['Global_Sales'],label='Sales')


#Añadimos las etiquetas de identificacion de valores en el grafico
ax.legend()
ax.set_xticklabels(df_group['Publisher'], rotation =45)
ax.set_ylabel(' Sales by Publisher')
fig.savefig('Mi_grafico.png',dpi=300)
plt.show()

import bar_chart_race as bcr
## Importamos el Dataset
path=r'C:\Users\esteb\OneDrive\Documents\Python Scripts\Visualizacion'
units_sales = pd.read_csv(path+'\Dinamico.csv',sep=';',header=0,index_col=False,
                          encoding='latin-1',engine='python')

units_sales.set_index('Order Date',inplace=True)

units_sales_acum = units_sales.cumsum(axis=0)

print(units_sales_acum)

bcr.bar_chart_race(df=units_sales_acum,filename=None,
                   figsize=(3.5,3),title = 'Ventas Acumuladas Enero 2014')

