import logging

import numpy as np

try:
    from sklearn.cluster import DBSCAN
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import lil_matrix
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def dbscan_cluster(embeddings, threshold, min_cluster_size):
    """Run DBSCAN on embeddings and return cluster labels."""
    if not _SKLEARN_AVAILABLE:
        raise ImportError("Please install sklearn: pip install scikit-learn")
    clustering = DBSCAN(
        eps=1 - threshold,
        min_samples=min_cluster_size,
        metric='cosine',
    ).fit(embeddings)
    return clustering.labels_


def connected_components_cluster(embeddings, threshold, batch_size=1000):
    """Run connected-components grouping on embeddings, return (n_components, labels)."""
    if not _SCIPY_AVAILABLE:
        raise ImportError("Please install scipy: pip install scipy")
    n = len(embeddings)
    adjacency = lil_matrix((n, n), dtype=np.int8)

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        batch = embeddings[i:end_i]
        similarities = np.dot(batch, embeddings.T)
        for j in range(batch.shape[0]):
            for k in range(n):
                if i + j != k and similarities[j, k] >= threshold:
                    adjacency[i + j, k] = 1

    n_components, labels = connected_components(adjacency.tocsr(), directed=False)
    return n_components, labels
