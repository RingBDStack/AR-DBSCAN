import torch
from sklearn.metrics import euclidean_distances
import copy


def aggregation_noise(data, cluster_label):
    r"""
    judge a cluster as noise cluster if its size is less than 3 and aggregate the noise points into the nearest cluster
    Args:
        data (torch.tensor): shape [n, emb_dim]
        cluster_label (torch.tensor): shape [n]
    """
    cluster_label = cluster_label.clone()
    cluster_size = torch.bincount(cluster_label)
    noise_cluster_id = (cluster_size <= 2).nonzero().squeeze()
    normal_points_mask = (cluster_label.view(-1, 1) != noise_cluster_id.view(
        1, -1)).all(dim=1).nonzero().squeeze()
    normal_data = data[normal_points_mask]
    normal_label = cluster_label[normal_points_mask]
    if noise_cluster_id.shape == torch.Size([]):
        # if there is only one noise cluster
        noise_cluster_id = noise_cluster_id.unsqueeze(0)
    for e in noise_cluster_id:
        noise_points = data[cluster_label == e]  # shape [noise_size, emb_dim]
        # find the nearest cluster and aggregate the noise points into it
        distance = euclidean_distances(noise_points,
                                       normal_data)  # shape [noise_size, n]
        distance = torch.tensor(distance)
        min_distance, nearest_point_idx = distance.min(
            dim=1)  # shape [noise_size]
        nearest_point_label = normal_label[nearest_point_idx]
        cluster_label[cluster_label == e] = nearest_point_label
    # re-id the cluster label
    used_label = torch.unique(cluster_label)
    idx = 0
    for e in used_label:
        cluster_label[cluster_label == e] = idx
        idx += 1
    return cluster_label
