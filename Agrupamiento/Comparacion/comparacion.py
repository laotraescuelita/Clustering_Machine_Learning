import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd


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


#Visualizar todos los algoritmos al mismo tiempo asi como su efectividad.
from sklearn.metrics.cluster import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
fig, axes = plt.subplots(1, 3, figsize=(8, 4),subplot_kw={'xticks': (), 'yticks': ()})
algoritmos = [KMeans(n_clusters=3), AgglomerativeClustering(n_clusters=3),DBSCAN()]
for ax, algoritmo in zip(axes, algoritmos):
	grupos = algoritmo.fit_predict(matriz_pca_escalar)
	ax.scatter(matriz_pca_escalar.iloc[:, 0], matriz_pca_escalar.iloc[:, 1], c=grupos, cmap="viridis", s=60)	
	ax.set_title("{} : {:.2f}".format(algoritmo.__class__.__name__,silhouette_score(matriz_pca_escalar, grupos)))
plt.show()
