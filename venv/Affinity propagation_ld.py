print(__doc__)

from sklearn.cluster import AffinityPropagation
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn import metrics

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels_true = digits.target

sample_size = 300

af = AffinityPropagation(preference=-50).fit(data)
cluster_centers_indices = af.cluster_centers_indices_
labels_pred = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_pred))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_pred))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'))


