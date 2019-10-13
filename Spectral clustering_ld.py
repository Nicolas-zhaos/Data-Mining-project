print(__doc__)


from sklearn.cluster import SpectralClustering
import numpy as np
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

clustering = SpectralClustering(affinity="nearest_neighbors").fit(data)
labels_pred = clustering.labels_

print('Estimated number of clusters: %d' %clustering.n_clusters)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_pred))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_pred))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'))