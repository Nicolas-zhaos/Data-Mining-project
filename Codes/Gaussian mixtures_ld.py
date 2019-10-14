import numpy as np
from sklearn import mixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn import metrics

np.random.seed(42)
digits = load_digits()
data = scale(digits.data)

n_digits = len(np.unique(digits.target))
labels_true = digits.target
sample_size = 300


gmm = mixture.GaussianMixture(n_components=50, covariance_type='full')
labels = gmm.fit(data).predict(data)
labels_pred = labels


print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_pred))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_pred))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'))

