import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering


path = './artificial/'
name="xclara.arff"

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



### FIXER la distance
# 
# tps1 = time.time()
# seuil_dist=10
# model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
# model = model.fit(datanp)
# tps2 = time.time()
# labels = model.labels_
# # Nb iteration of this method
# #iteration = model.n_iter_
# k = model.n_clusters_
# leaves=model.n_leaves_
# plt.scatter(f0, f1, c=labels, s=8)
# plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
# plt.show()
# print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

# ---------- Détermination d'une bonne solution de clustering avec arrêt selon n_clusters ---------- #
# tab_sil = []
# tab_ch = []
# tab_db = []
# for k in range (2,15):
#     model = cluster.AgglomerativeClustering(linkage='average', distance_threshold=None, n_clusters = k)
#     model.fit(datanp)
#     labels = model.labels_

#     sil = metrics.silhouette_score(datanp, labels)
#     db = metrics.davies_bouldin_score(datanp, labels)
#     ch = metrics.calinski_harabasz_score(datanp, labels)
    
#     tab_sil.append(sil)
#     tab_db.append(db)
#     tab_ch.append(ch)

# ---------- Affichage du graphique de l'évolution du coefficient de silhouette en fonction du nombre de clusters k---------- #
# plt.plot(range(2,15), tab_sil, marker="x")
# plt.xlabel("Nombre de clusters (k)")
# plt.ylabel("Coefficient de silhouette")
# plt.title("Coefficient de silhouette en fonction de k")
# plt.show()

# # ---------- Affichage du graphique de l'évolution de l'indice de Davies-Bouldin en fonction du nombre de clusters k---------- #
# plt.plot(range(2,15), tab_db, marker="x")
# plt.xlabel("Nombre de clusters (k)")
# plt.ylabel("Indice de Davies-Bouldin")
# plt.title("Indice de Davies-Bouldin en fonction de k")
# plt.show()

# # ---------- Affichage du graphique de l'évolution de l'indice de Calinski-Harabasz en fonction du nombre de clusters k---------- #
# plt.plot(range(2,15), tab_ch, marker="x")
# plt.xlabel("Nombre de clusters (k)")
# plt.ylabel("Indice de Calinski-Harabasz")
# plt.title("Indice de Calinski-Harabasz en fonction de k")
# plt.show()




# ---------- Détermination d'une bonne solution de clustering avec arrêt selon distance ---------- #
tab_sil = []
tab_ch = []
tab_db = []
tab_k = []
for dist in range(20,100, 5):
    model = cluster.AgglomerativeClustering(linkage='average', distance_threshold=dist, n_clusters = None)
    model.fit(datanp)
    labels = model.labels_
    k = model.n_clusters_

    if k>1:

        sil = metrics.silhouette_score(datanp, labels)
        db = metrics.davies_bouldin_score(datanp, labels)
        ch = metrics.calinski_harabasz_score(datanp, labels)
        
        tab_sil.append(sil)
        tab_db.append(db)
        tab_ch.append(ch)
        tab_k.append(k)


# ---------- Affichage du graphique de l'évolution du coefficient de silhouette en fonction du nombre de clusters k---------- #
plt.plot(tab_k, tab_sil, marker="x")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Coefficient de silhouette")
plt.title("Coefficient de silhouette en fonction de k")
plt.show()

# ---------- Affichage du graphique de l'évolution de l'indice de Davies-Bouldin en fonction du nombre de clusters k---------- #
plt.plot(tab_k, tab_db, marker="x")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Indice de Davies-Bouldin")
plt.title("Indice de Davies-Bouldin en fonction de k")
plt.show()

# ---------- Affichage du graphique de l'évolution de l'indice de Calinski-Harabasz en fonction du nombre de clusters k---------- #
plt.plot(tab_k, tab_ch, marker="x")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Indice de Calinski-Harabasz")
plt.title("Indice de Calinski-Harabasz en fonction de k")
plt.show()


###
# FIXER le nombre de clusters
###
# k=3
# tps1 = time.time()
# model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
# model = model.fit(datanp)
# tps2 = time.time()
# labels = model.labels_
# # Nb iteration of this method
# #iteration = model.n_iter_
# kres = model.n_clusters_
# leaves=model.n_leaves_
# #print(labels)
# #print(kres)

# plt.scatter(f0, f1, c=labels, s=8)
# plt.title("Clustering agglomératif (average, n_cluster= "+str(k)+") "+str(name))
# plt.show()
# print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")



#######################################################################