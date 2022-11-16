import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


#Pasos para analizar una matriz de datos. 
# Cero: Extracción de los datos, limpieza, reducción y eliminación.
# Uno: Estandarizar, normalizar y7o reducir datos. 
# Dos: Codificar las varibles categoricas.
# Tres: Escojer un módelo de machine learning. 


#Leer la matriz de datos con los métodos de pandas. 
#matriz = pd.read_csv("matriz.csv")
matriz = pd.read_csv("iris.csv")
#Colocar indice al dataframe matriz.
indice = np.arange(0,matriz.shape[0])
matriz = matriz.reindex( indice )
#Revisar la matriz.
print("\nTamaño de la matriz: \n ", matriz.shape )
print("\nPrimeras filas de la matriz: \n ",  matriz.head() )
#Cuantificar la cantidad de datos perdidos para determinar si los eliminamos, rellenamos, etc.
print("\nCantidad de datos faltantes: \n ", matriz.isnull().sum()  )
print("\nCantidad de datos faltantes en porcentaje \n", matriz.isnull().sum()/matriz.shape[0]*100)
#Revisar los tipos de variables y determinar si son correctas o las modifcamos.
print("\nTipos de variables: \n ", matriz.info() )
#Revisar algunas estadisticas basicas.
print("\nEstadisticas basicas: \n ", matriz.describe() )

#Revisar la forma de las variables.
matriz.hist( figsize=(12,6) ) 
plt.show()


#Algorimto de aglomeramiento.
from sklearn.cluster import AgglomerativeClustering
aglomerativo = AgglomerativeClustering(n_clusters=3)
#aglomerativo.fit_predict(matriz)
#ValueError: could not convert string to float: 'GP'
#Aqui truena porque hay variables categoricas, así que hay que codificarlas.

#Separemos las columnas numericas de las categoricas. Como lo índica la matriz.
columnas_categoricas = matriz.select_dtypes("object")
columnas_numericas = matriz.select_dtypes(np.number)
print("\nVariables numericas: \n ", columnas_numericas.columns )
print("\nVariables categoricas: \n ", columnas_categoricas.columns )

# Hay un módulo que nos ayuda a realizar esta acción.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

#Hay que mirar como quedan las variable categoricas codificadas.
for columna in columnas_categoricas:
    print('\nNombre de la columna:', "\n", columna)    
    matriz[columna] = matriz[[columna]].apply(encoder.fit_transform)
    for i in range(len(encoder.classes_)):
        print(encoder.classes_[i],':', i)


#Aquí verificamos como todas las varibales ahora son numericas, tanto numeros enteros como continuos.
#Revisar los tipos de variables y determinar si son correctas o las modifcamos.
print("\nTipos de variables: \n ", matriz.info() )
#Revisar algunas estadisticas basicas.
print("\nEstadisticas basicas: \n ", matriz.describe() )

"""
#Ahora si intentemos aplicar el algoritmo de aglomeramiento.
aglomerativo_etiquetas = aglomerativo.fit_predict(matriz)
print("Etiquetas :\n{}".format(aglomerativo_etiquetas))
print("Dimensiones :\n{}".format(aglomerativo_etiquetas.shape))
#Unir el vector de clasificacion con la matriz de datos.
#Primero hay que covertir el vector a dataframe.
aglomerativo_etiquetas_ = pd.DataFrame( aglomerativo_etiquetas, columns= ["Categorias"])
matriz_aglomerativo = pd.merge(matriz, aglomerativo_etiquetas_, left_index= True, right_index= True, how = "inner")
print("\nMatriz con las categorias asignadas por aglomeramiento: \n ", matriz_aglomerativo.head(2) )
print("\nTipos de variables de la nueva matriz con las categorias de aglomeramiento: \n ", matriz_aglomerativo.info())

plt.scatter(matriz_aglomerativo.iloc[:, 0], matriz_aglomerativo.iloc[:, 1], c=aglomerativo_etiquetas, cmap="viridis", s=60)
plt.xlabel("Primer componente")
plt.ylabel("Segundo componente")
plt.show()

#Exportar los datos.
matriz_aglomerativo.to_csv("matriz_aglomerativo.csv")
"""

"""
#Reducir la matriz para poder graficar los datos.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(matriz.values)
pcaReducido = pca.transform(matriz.values)
print("Dimensiones originales: {}".format(str(matriz.shape)))
print("Dimensiones reducidas: {}".format(str(pcaReducido.shape)))
matriz_pca = pd.DataFrame (pcaReducido , columns=["pc1","pc2"])
print("\nMatriz reducida por medio de pca: \n ", matriz_pca.head(2) )
print("\nTipos de variables de la nueva matriz con pca: \n ", matriz_pca.info())

#Ahora si intentemos aplicar el algoritmo de aglomeramiento.
aglomerativo_pca_etiquetas = aglomerativo.fit_predict(matriz_pca)
print("Etiquetas :\n{}".format(aglomerativo_pca_etiquetas))
print("Dimensiones :\n{}".format(aglomerativo_pca_etiquetas.shape))
#Unir el vector de clasificacion con la matriz de datos.
#primeo hay que covertir el vector a dataframe.
aglomerativo_pca_etiquetas_ = pd.DataFrame( aglomerativo_pca_etiquetas, columns= ["Categorias"])
matriz_pca_aglomerativo = pd.merge(matriz_pca, aglomerativo_pca_etiquetas_, left_index= True, right_index= True, how = "inner")
print("\nMatriz con las categorias asignadas por aglomeramiento: \n ", matriz_pca_aglomerativo.head(2) )
print("\nTipos de variables de la nueva matriz con las categorias de aglomeramiento: \n ", matriz_pca_aglomerativo.info())

plt.scatter(matriz_pca_aglomerativo.iloc[:, 0], matriz_pca_aglomerativo.iloc[:, 1], c=aglomerativo_pca_etiquetas, cmap="viridis", s=60)
plt.xlabel("Primer componente")
plt.ylabel("Segundo componente")
plt.show()

#Exportar los datos.
matriz_pca_aglomerativo.to_csv("matriz_pca_aglomerativo.csv")
"""

"""
#Vamos a standarizar los datos con variables continuas antes de aplicar el pca.
from sklearn import preprocessing
scalar = preprocessing.StandardScaler()
for columna in columnas_numericas:
    matriz[columna] = scalar.fit_transform(matriz[[columna]])
print("\nMatriz con datos dentrados: \n ", matriz.head(2) )
print("\nTipos de variables de la nueva matriz con datos centrados: \n ", matriz.info())

#Reducir la matriz para poder graficar los datos.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(matriz.values)
pcaEscalarReducido = pca.transform(matriz.values)
print("Dimensiones originales: {}".format(str(matriz.shape)))
print("Dimensiones reducidas: {}".format(str(pcaEscalarReducido.shape)))
matriz_pca_escalar = pd.DataFrame (pcaEscalarReducido , columns=["pc1","pc2"])
print("\nMatriz reducida por medio de pca: \n ", matriz_pca_escalar.head(2) )
print("\nTipos de variables de la nueva matriz con pca: \n ", matriz_pca_escalar.info())


#Ahora si intentemos aplicar el algoritmo de aglomeramiento.
aglomerativo__pca_escalar_etiquetas = aglomerativo.fit_predict(matriz_pca_escalar)
print("Etiquetas :\n{}".format(aglomerativo__pca_escalar_etiquetas))
print("Dimensiones :\n{}".format(aglomerativo__pca_escalar_etiquetas.shape))
#Unir el vector de clasificacion con la matriz de datos.
#primeo hay que covertir el vector a dataframe.
aglomerativo__pca_escalar_etiquetas_ = pd.DataFrame( aglomerativo__pca_escalar_etiquetas, columns= ["Categorias"])
matriz_pca_escalar_aglomerativo = pd.merge(matriz_pca_escalar, aglomerativo__pca_escalar_etiquetas_, left_index= True, right_index= True, how = "inner")
print("\nMatriz con las categorias asignadas por kmeans: \n ", matriz_pca_escalar_aglomerativo.head(2) )
print("\nTipos de variables de la nueva matriz con las categorias de kmeans: \n ", matriz_pca_escalar_aglomerativo.info())


plt.scatter(matriz_pca_escalar_aglomerativo.iloc[:, 0], matriz_pca_escalar_aglomerativo.iloc[:, 1], c=aglomerativo__pca_escalar_etiquetas, cmap="viridis", s=60)
plt.xlabel("Primer componente")
plt.ylabel("Segundo componente")
plt.show()


#Exportar los datos.
matriz_pca_escalar_aglomerativo.to_csv("matriz_pca_escalar_aglomerativo.csv")
"""

"""
#Por ultimo vamos a eliminar los oautliers antes de realizar todo el proceso.
#Revisar los outliers.
matriz.plot( kind = "box", figsize=(12,6) ) 
plt.show()

#Vamos a remover los "outliers" en las columnas numericas que hacen el sesgo en los datos.
def remover_outliers(matriz, columnas):    
    for columna in columnas:
        q1 = matriz[columna].quantile(q = 0.25)
        q3 = matriz[columna].quantile(q = 0.75)
        interCuartil = q3 - q1
        rangoAlto = q3 + (1.5*interCuartil)
        rangoBajo = q1 - (1.5*interCuartil)
        matriz.loc[matriz[columna] > rangoAlto, columna] = np.nan
        matriz.loc[matriz[columna] < rangoBajo, columna] = np.nan
    return matriz.dropna()

matriz = remover_outliers(matriz, columnas_numericas)
#Revisar los outliers.
matriz.plot( kind = "box", figsize=(12,6) ) 
plt.show()

matriz = remover_outliers(matriz, columnas_numericas)
#Revisar los outliers.
matriz.plot( kind = "box", figsize=(12,6) ) 
plt.show()


#Vamos a standarizar los datos con variables continuas antes de aplicar el pca.
from sklearn import preprocessing
scalar = preprocessing.StandardScaler()
for columna in columnas_numericas:
    matriz[columna] = scalar.fit_transform(matriz[[columna]])
print("\nMatriz con datos dentrados: \n ", matriz.head(2) )
print("\nTipos de variables de la nueva matriz con datos centrados: \n ", matriz.info())

#Reducir la matriz para poder graficar los datos.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(matriz.values)
pcaEscalarReducido = pca.transform(matriz.values)
print("Dimensiones originales: {}".format(str(matriz.shape)))
print("Dimensiones reducidas: {}".format(str(pcaEscalarReducido.shape)))
matriz_pca_escalar = pd.DataFrame (pcaEscalarReducido , columns=["pc1","pc2"])
print("\nMatriz reducida por medio de pca: \n ", matriz_pca_escalar.head(2) )
print("\nTipos de variables de la nueva matriz con pca: \n ", matriz_pca_escalar.info())


#Ahora si intentemos aplicar el algoritmo de aglomeramiento.
escalar_pca_aglomerativo_etiquetas = aglomerativo.fit_predict(matriz_pca_escalar)
print("Etiquetas :\n{}".format(escalar_pca_aglomerativo_etiquetas))
print("Dimensiones :\n{}".format(escalar_pca_aglomerativo_etiquetas.shape))
#Unir el vector de clasificacion con la matriz de datos.
#primeo hay que covertir el vector a dataframe.
escalar_pca_aglomerativo_etiquetas_ = pd.DataFrame( escalar_pca_aglomerativo_etiquetas, columns= ["Categorias"])
matriz_escalar_pca_aglomerativo_outliers = pd.merge(matriz_pca_escalar, escalar_pca_aglomerativo_etiquetas_, left_index= True, right_index= True, how = "inner")
print("\nMatriz con las categorias asignadas por aglomerativo: \n ", matriz_escalar_pca_aglomerativo_outliers.head(2) )
print("\nTipos de variables de la nueva matriz con las categorias de aglomerativo: \n ", matriz_escalar_pca_aglomerativo_outliers.info())

plt.scatter(matriz_escalar_pca_aglomerativo_outliers.iloc[:, 0], matriz_escalar_pca_aglomerativo_outliers.iloc[:, 1], c=escalar_pca_aglomerativo_etiquetas, cmap="viridis", s=60)
plt.xlabel("Primer componente")
plt.ylabel("Segundo componente")
plt.show()

#Exportar los datos.
matriz_escalar_pca_aglomerativo_outliers.to_csv("matriz_escalar_pca_aglomerativo_outliers.csv")
"""