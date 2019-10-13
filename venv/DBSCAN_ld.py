print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale


# # #############################################################################
# # Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X= make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
# #                             random_state=0)
#
# X = StandardScaler().fit_transform(X)
#
# # #############################################################################
# # Compute DBSCAN
np.random.seed(42)
digits = load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels_true = digits.target
sample_size = 300


db = DBSCAN(eps=10, min_samples=1).fit(data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels_pred = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
n_noise_ = list(labels_pred).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_pred))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_pred))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'))