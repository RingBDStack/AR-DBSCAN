from torch_cluster import knn_graph, radius_graph
import torch
import torch.nn.functional as F
from torch_scatter import scatter_sum
from utils.code_tree import PartitionTree
from tqdm import tqdm

from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning, EncodingTree

import logging


class EdgeRedu:
    r"""
    Edge Reduction function class.
    """

    @staticmethod
    def _reduction_edge(edges, edge_transform, *weights):
        r"""
        Sum up the weights for the same edges.

        Return tuple (edges, weights)

        Args:
            edges(torch.LongTensor): Source edges with shape :obj:`(E,2)`.
            edge_transform(torch.Tensor): Edge id returned from `get_edge_transform`.
            *weights(list): (weight1, weight2, ...), weighti is torch.Tensor with shape :obj:`E`.
        
        :rtype: (:class:`Tensor`, :class:`Tensor`, ...)
        """
        cnt_e = edge_transform.max() + 1
        e1 = torch.zeros(size=(cnt_e, edges.shape[1]),
                         dtype=edges.dtype,
                         device=edges.device)
        edges = e1.scatter(0,
                           edge_transform.reshape(-1, 1).repeat((1, 2)), edges)
        ret = [edges] + [scatter_sum(w, edge_transform) for w in weights]
        return tuple(ret)

    @staticmethod
    def _get_edge_transform(edges, identical_flag=False):
        r"""
        Assign an ID to each edge, where duplicate edges have the same ID.

        Return the assigned ID Tensor, (identical boolean Tensor)
        Args:
            edges(torch.LongTensor): Source edges with shape :obj:`(E,2)`.
            identical_flag(Bool): Identical_flag control whether returns a :class:`torch.bool` type Tensor indicating repeated edges.

        :rtype: :class:`Tensor` (, :class:`Tensor`)
        """
        max_id = int(edges[:, 1].max() + 1)
        bd = 1
        shift = 0
        while bd <= max_id:
            bd = bd << 1
            shift += 1
        # todo: hash if shift is too big
        edge_hash = (edges[:, 0] << shift) + edges[:, 1]

        if identical_flag:
            _, transform, counts = torch.unique(edge_hash,
                                                return_inverse=True,
                                                return_counts=True)
            flag = counts[transform] != 1
            return transform, flag
        else:
            _, transform = torch.unique(edge_hash, return_inverse=True)
            return transform


def sumup_duplicates(edges, *weights):
    r"""
    Sum up the weight of duplicate edge.

    Args:
        edges(torch.LongTensor): Edges with shape :obj:`(E,2)`.
        *weights(list): (weight1, weight2, ...), weighti is torch.Tensor with shape :obj:`E`.
    
    :rtype: (:class:`Tensor`, :class:`Tensor`, ...)
    """
    trans = EdgeRedu._get_edge_transform(edges)
    return EdgeRedu._reduction_edge(edges, trans, *weights)


def build_knn_graph(data, k):
    r"""
    construct k-NN graph

    return: 
        edges: shape [E, 2]
        ew: shape [E]
    """
    # euclidean_distances
    edges = knn_graph(data, k)  # [2, E]
    ew = F.pairwise_distance(data[edges[0]], data[edges[1]])
    edges, ew = sumup_duplicates(edges.t(), ew)  # edges: [E, 2]
    ew = torch.exp2(-ew / ew.mean())  #similarity
    return edges, ew

def k_selector_silearn(data, verbose=True):
    r'''
    use silearn to write k_selector
    Select k
    data: shape [n, emb_dim]
    '''
    se_list = [0, 0, 0]
    n_sample = data.shape[0]
    community_list = []
    cnt = 0
    k = 0
    stable_point = {'k': [], 'se': []}
    if verbose:
        k_iter = tqdm(range(3, n_sample))
        k_iter.set_description('k selector')
    else:
        k_iter = range(3, n_sample)
    patient = 0
    for k in k_iter:
        # k-NN
        edges, ew = build_knn_graph(data, k)

        # get stable distribution
        dist = scatter_sum(ew, edges[:, 1], dim_size=n_sample) + scatter_sum(
            ew, edges[:, 0], dim_size=n_sample)
        dist = dist / (2 * ew.sum())

        g = GraphSparse(edges, ew, dist)
        optim = OperatorPropagation(Partitioning(g, None))
        optim.perform(m_scale=data.shape[0])
        community_num = optim.enc.node_id.max() + 1
        community_list.append(community_num)

        se = optim.enc.structural_entropy(reduction='sum',
                                          norm=True) / (k * data.shape[0])
        se_list.append(se)

        if se_list[k - 1] < se_list[k - 2] and se_list[k - 1] < se_list[k]:
            stable_point['k'].append(k - 1)
            stable_point['se'].append(se_list[k - 1])
        if community_num == 1:
            patient += 1
            if patient >= 2:
                break
        if verbose:
            k_iter.set_postfix({
                'k':
                k,
                'community_num':
                int(community_num),
                'se':
                float(
                    optim.enc.structural_entropy(reduction='sum') /
                    (k * data.shape[0]))
            })
    if len(stable_point['k']) > 0:
        # get k with min se in stable point

        stable_point['se'] = torch.tensor(stable_point['se'])
        k = stable_point['k'][stable_point['se'].argmin()]
        if verbose:
            print(f'Select stable point: {k}', flush=True)
    else:
        community_list = torch.tensor(community_list)
        freq_community = torch.bincount(community_list).argmax()
        k = (community_list == freq_community).nonzero().squeeze() + 3
        if k.shape == torch.Size([]):
            k = k
        else:
            k = k[0]
        if verbose:
            print(f'Select the most freq community point: {k}', flush=True)
    return k