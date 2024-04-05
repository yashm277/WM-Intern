import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy


def hierarchialClustering(my_distance_matrix, threshold):
    # Example distance matrix (replace this with your actual distance matrix)
    distance_matrix = np.array(my_distance_matrix)

    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(distance_matrix, method='complete')

    # Plot dendrogram
    plt.figure(figsize=(8, 6))
    hierarchy.dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()

    # Choose a distance threshold to cut the dendrogram and obtain clusters
    distance_threshold = threshold  # Adjust this threshold based on the dendrogram
    clusters = hierarchy.fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
    clustersFinal = []
    maxi = max(clusters)
    for i in range(1,maxi+1):
        clusterSet = []
        for j in range(len(clusters)):
            if clusters[j]==i:
                clusterSet.append(j)
        clustersFinal.append(clusterSet)
    return clustersFinal
