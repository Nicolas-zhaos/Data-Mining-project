
print(__doc__)
from time import time
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold, datasets

digits = datasets.load_digits(n_class=10)
data = digits.data
labels_true = digits.target


# 2D embedding of the digits dataset
print("Computing embedding")
data_mf = manifold.SpectralEmbedding(n_components=2).fit_transform(data)
print("Done.")



for linkage in ('ward', 'average', 'complete', 'single'):
    ac = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    t0 = time()
    ac.fit(data_mf)
    print("%s :\t%.2fs" % (linkage, time() - t0))

print("-" * 20)

labels_pred = ac.labels_
print('Estimated number of clusters: %d' %ac.n_clusters)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_pred))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_pred))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'))




