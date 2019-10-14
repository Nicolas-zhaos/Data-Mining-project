

# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import Normalizer
# from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import  MiniBatchKMeans
import sys
from time import time
import numpy as np



def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# #############################################################################
# Load some categories from the training set
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

labels_true = dataset.target
true_k = np.unique(labels_true).shape[0]

print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(dataset.data)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()


# #############################################################################
# Do the actual clustering
km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=0)
t0 = time()
km.fit(X)
labels_Pred = km.labels_

print("done in %0.3fs" % (time() - t0))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_Pred))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_Pred))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels_Pred, average_method='arithmetic'))
print()

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()


