from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import DBSCAN
from sklearn import metrics
from time import time
import numpy as np


categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels_ture = dataset.target
true_k = np.unique(labels_ture).shape[0]

print("Extracting features from the training dataset "
      "using a sparse vectorizer")

#矩阵和权值
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
matrix = vectorizer.fit_transform(dataset.data)


print("n_samples: %d, n_features: %d" % matrix.shape)
print()

#降维
print("Performing dimensionality reduction using LSA")
t0 = time()
svd = TruncatedSVD(2)  #维度
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

matrix_l = lsa.fit_transform(matrix)

# #############################################################################
# Do the actual clustering

db = DBSCAN(eps=10, min_samples=1).fit(matrix_l)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels_pred = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
n_noise_ = list(labels_pred).count(-1)
print("done in %0.3fs" % (time() - t0))
print()



print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_ture, labels_pred))
print("Completeness: %0.3f" % metrics.completeness_score(labels_ture, labels_pred))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_ture, labels_pred, average_method='arithmetic'))



