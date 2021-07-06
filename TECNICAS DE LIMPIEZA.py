#!/usr/bin/env python
# coding: utf-8

# ## Tecnicas para la Limpieza de datos
# 

# #Cargamos los datos

# In[2]:


import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/diplomado-bigdata-machinelearning-udea/Curso1/master/s04/adult.csv')


# In[3]:


df.head()


# In[ ]:


# Encontrar el número de valores faltantes por columna


# In[4]:


df.isnull().sum()


# In[5]:


# Porcentaje de información faltante
df.isnull().sum()/len(df)*100


# In[6]:


# El comando dropna() permite eliminar las filas y/o columnas en las que hayan datos faltantes.

df_filtered=df.dropna()

print('Número de filas iniciales', len(df))
print('Número de filas después de filtrar', len(df_filtered))
print('Porcentaje de filas eliminadas',(1-len(df_filtered)/len(df))*100)


# In[11]:


###Note que los índices no cambian. Lo que realiza es la eliminación de la fila (por ejemplo la 14 no está), 
#pero mantiene la indexación. Por tanto, estos no coinciden con el número total de filas.
#Para reasignar los índices se puede hacer uso del comando reset_index()###


# In[12]:


df_filtered.reset_index(drop=True,inplace=True)


# In[13]:


df_filtered.tail()
df.dropna(subset=['native-country'])


# In[14]:


#Llenado con valores vecinos
df.iloc[25:30,:]
df.iloc[25:30,:].fillna(method='bfill')


# In[15]:


#Llenado de datos usando sklearn
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='most_frequent')
X = df.iloc[25:30,:]
trans = pd.DataFrame(imp.fit_transform(X))
trans


# In[ ]:




