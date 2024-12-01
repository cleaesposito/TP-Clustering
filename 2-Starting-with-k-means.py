"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="square1.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Run clustering method for a given number of clusters
# print("------------------------------------------------------")
# print("Appel KMeans pour une valeur de k fixée")
# tps1 = time.time()
# k=4
# model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
# model.fit(datanp)
# tps2 = time.time()
# labels = model.labels_
# # informations sur le clustering obtenu
# iteration = model.n_iter_
# inertie = model.inertia_
# centroids = model.cluster_centers_

# ---------- Détermination d'une bonne solution de clustering via l'inertie ---------- #
# tab_inerties = []
# for k in range(1,15):
#     model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
#     model.fit(datanp)
#     inertie = model.inertia_
#     tab_inerties.append(inertie)

# ---------- Détermination d'une bonne solution de clustering via les 3 méthodes automatiques ---------- #
tab_sil = []
tab_ch = []
tab_db = []
for k in range (2,15):
    model = cluster.KMeans(n_clusters=k,init='k-means++', n_init=1)
    model.fit(datanp)
    labels = model.labels_

    sil = silhouette_score(datanp, labels)
    db = davies_bouldin_score(datanp, labels)
    ch = calinski_harabasz_score(datanp, labels)
    
    tab_sil.append(sil)
    tab_db.append(db)
    tab_ch.append(ch)

# ---------- Affichage du graphique de l'évolution de l'inertie en fonction du nombre de clusters k---------- #
# plt.plot(range(1,15), tab_inerties, marker="x")
# plt.xlabel("Nombre de clusters (k)")
# plt.ylabel("Inertie du clustering")
# plt.title("Inertie en fonction de k")
# plt.show()

# ---------- Affichage du graphique de l'évolution du coefficient de silhouette en fonction du nombre de clusters k---------- #
plt.plot(range(2,15), tab_sil, marker="x")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Coefficient de silhouette")
plt.title("Coefficient de silhouette en fonction de k")
plt.show()

# ---------- Affichage du graphique de l'évolution de l'indice de Davies-Bouldin en fonction du nombre de clusters k---------- #
plt.plot(range(2,15), tab_db, marker="x")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Indice de Davies-Bouldin")
plt.title("Indice de Davies-Bouldin en fonction de k")
plt.show()

# ---------- Affichage du graphique de l'évolution de l'indice de Calinski-Harabasz en fonction du nombre de clusters k---------- #
plt.plot(range(2,15), tab_ch, marker="x")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Indice de Calinski-Harabasz")
plt.title("Indice de Calinski-Harabasz en fonction de k")
plt.show()


#plt.figure(figsize=(6, 6))
# plt.scatter(f0, f1, c=labels, s=8)
# plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
# plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
# #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
# plt.show()

# print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

# from sklearn.metrics.pairwise import euclidean_distances
# dists = euclidean_distances(centroids)
# print(dists)
