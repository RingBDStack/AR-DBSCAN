import torch
from utils.build_graph import build_knn_graph, k_selector_silearn
from torch_scatter import scatter_sum

from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning, EncodingTree
from utils.aggregation_noise import aggregation_noise
import copy


def se_pre_partition(data, verbose=True):
    r"""
    use structural entropy to complete pre_partition

    Args:
        data: shape [n, emb_dim]
    Returns:
        cluster_label: shape [n]
        uncertainty: shape [n]
    """
    data = torch.tensor(copy.deepcopy(data))
    n_sample = data.shape[0]
    k = k_selector_silearn(data, verbose)
    # k = 9  # for D31
    # k = len(data) // 20
    # k = 1568  # for unbalance2
    # k = 10
    # k = 78  # for asymmetric
    edges, ew = build_knn_graph(data, k)
    dist = scatter_sum(ew, edges[:, 1], dim_size=n_sample) + scatter_sum(
        ew, edges[:, 0], dim_size=n_sample)
    dist = dist / (2 * ew.sum())
    g = GraphSparse(edges, ew, dist)
    optim = OperatorPropagation(Partitioning(g, None))
    optim.perform(m_scale=data.shape[0])
    cluster_label_with_noise = optim.enc.node_id
    cluster_label = aggregation_noise(data, cluster_label_with_noise)
    optim.enc.node_id = cluster_label
    cluster_size = torch.bincount(cluster_label)
    uncertainty = optim.enc.structural_entropy(
        reduction='module') / (cluster_size * k)
    return cluster_label, uncertainty, cluster_label_with_noise
