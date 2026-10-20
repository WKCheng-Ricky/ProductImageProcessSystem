import numpy as np

from src.pipeline.clustering import connected_components_cluster, dbscan_cluster


def _two_clusters_and_outlier():
    a = np.array([[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.98, 0.02, 0.0]])
    b = np.array([[0.0, 1.0, 0.0], [0.01, 0.99, 0.0], [0.02, 0.98, 0.0]])
    outlier = np.array([[0.0, 0.0, 1.0]])
    stacked = np.vstack([a, b, outlier])
    return stacked / np.linalg.norm(stacked, axis=1, keepdims=True)


def test_dbscan_separates_clusters_and_marks_outlier_as_noise():
    labels = dbscan_cluster(_two_clusters_and_outlier(), threshold=0.9, min_cluster_size=2)

    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:6])) == 1
    assert labels[0] != labels[3]
    assert labels[6] == -1


def test_connected_components_groups_similar_vectors():
    n_components, labels = connected_components_cluster(
        _two_clusters_and_outlier(), threshold=0.9
    )

    assert n_components == 3
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3] != labels[6]
