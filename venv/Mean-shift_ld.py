print(__doc__)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels_true = digits.target

sample_size = 300
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
labels_pred = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels_pred)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_pred))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_pred))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'))