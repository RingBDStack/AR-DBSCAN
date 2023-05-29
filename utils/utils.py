import math
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.cluster import KMeans
from coclust.evaluation.external import accuracy
"""
    Utility functions to handle data and evaluate model.
"""


def load_data_shape(data, train_size):
    """
    Load the datasets of shape type
    :param data: path of dataset
    :param train_size: proportion of samples used to generate rewards
    :return: sample serial numbers for rewards, out-of-order data features and labels
    """
    extract_data = []
    with open(data, 'r') as f:
        for line in f:
            data = line.split()
            data[0] = float(data[0])
            data[1] = float(data[1])
            data[2] = int(data[2])
            extract_data.append(data)

    # shuffle the samples
    extract_data_idx = random.sample(extract_data, len(extract_data))

    # feature normalization
    extract_data_features = np.array(MinMaxScaler().fit_transform(
        [i[0:2] for i in extract_data_idx]))
    extract_data_labels = np.array([i[2] for i in extract_data_idx])

    # sample index for reward
    extract_data_train_masks = random.sample(
        range(len(extract_data_idx)), int(len(extract_data_idx) * train_size))

    return extract_data_train_masks, extract_data_features, extract_data_labels


def load_data_stream(data, train_size, block_num, block_size):
    """
    Load the datasets of stream type
    :param data: path of dataset
    :param train_size: proportion of samples used to generate rewards
    :param block_num: the number of data blocks
    :param block_size: the size of data block
    :return: sample serial numbers for rewards, out-of-order data features and labels
    """

    # extract features and labels of the dataset
    extract_datas = []
    with open(data, 'r') as f:
        i = 0
        for line in f:
            data = line.strip().split(',')
            # print(data)
            data = list(map(float, data))
            data[-1] = int(data[-1])
            extract_datas.append(data)
            i += 1
            if i == block_num * block_size: break
    extract_datas = [
        extract_datas[block_size * i:block_size * (i + 1)]
        for i in range(block_num)
    ]

    extract_data_train_maskss, extract_data_featuress, extract_data_labelss = [], [], []
    for extract_data in extract_datas:
        # shuffle the samples
        extract_data_idx = random.sample(extract_data, len(extract_data))

        # feature normalization
        extract_data_features = np.array(MinMaxScaler().fit_transform(
            [i[0:len(extract_data[0]) - 1] for i in extract_data_idx]))
        extract_data_labels = np.array([i[-1] for i in extract_data_idx])

        # sample index for reward
        extract_data_train_masks = random.sample(
            range(len(extract_data_idx)),
            int(len(extract_data_idx) * train_size))
        extract_data_train_maskss.append(extract_data_train_masks)
        extract_data_featuress.append(extract_data_features)
        extract_data_labelss.append(extract_data_labels)

    return extract_data_train_maskss, extract_data_featuress, extract_data_labelss


def generate_parameter_space(features, num_layer, eps_size, min_size,
                             data_path):
    """
    Parameter space automatic generator
    :param features: data features
    :param num_layer: reinforcement learning recursive layers
    :param eps_size: size of the eps parameter space in each recursive layer
    :param min_size: size of the MinPts parameter space in each recursive layer
    :param data_path: path of dataset, used to determine the maximum search range of MinPts
    :return: parameter space size per layer, step size per layer, starting point of the first layer,
    limit bound for the two parameter spaces
    """

    # the maximum search boundary of MinPts
    if "Shape" in data_path:
        mm = 4
    elif "Stream" in data_path:
        mm = 4

    # feature dimension and dataset size
    num, dim = features.shape

    # parameter space size
    param_size = [eps_size, min_size]

    # parameter search step
    # eps step is previous layer eps_step / eps_size
    # MinPts step is previous layer MinPts_step / min_size
    param_step = [[(int(math.sqrt(dim)) / eps_size**i),
                   math.ceil(int(num / mm) / min_size**i)]
                  for i in range(1, num_layer + 1)]

    # parameter starting point
    if "Shape" in data_path:
        param_center = [int(math.sqrt(dim)) / 2, math.ceil(num / mm / 2)]
    else:
        param_center = [int(math.sqrt(dim)) / 2, math.ceil(num / mm / 2)]

    # layer 0 parameter space boundary
    # data is normalized to [0,1], so the search range of eps is (0,sqrt(dim)]
    # the search range of MinPts is [1, N/m], m is the maximum search boundary of MinPts
    param_bound = [[0.0000001, int(math.sqrt(dim))], [1, math.ceil(num / mm)]]

    return param_size, param_step, param_center, param_bound


def kmeans_metrics(features, labels):
    """
    the evaluate function of clustering algorithm
    :param features: data features
    :param labels: accurate labels
    :return: normalized mutual info
    """

    # n_clusters
    extract_cluster_num = len(set(labels.tolist()))

    # Dataset size
    extract_data_num = features.shape[0]

    # Clustering results of Kmeans
    k_labels = KMeans(n_clusters=extract_cluster_num,
                      n_init='auto').fit(features).labels_

    k_nmi = round(metrics.normalized_mutual_info_score(labels, k_labels), 4)
    k_ami = round(metrics.adjusted_mutual_info_score(labels, k_labels), 4)
    k_ari = round(metrics.adjusted_rand_score(labels, k_labels), 4)
    print("\nDataset size:  " + str(extract_data_num), flush=True)
    print("Cluster number:  " + str(extract_cluster_num), flush=True)
    # print('\n+-------------------------------------------------------+\n'
    #       '                         K-Means                             '
    #       '\n+-------------------------------------------------------+\n')
    print("******* K-Means NMI: " + str(k_nmi), flush=True)
    print("******* K-Means AMI: " + str(k_ami), flush=True)
    print("******* K-Means ARI: " + str(k_ari), flush=True)

    return k_nmi


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix,
                          axis=0)) / np.sum(contingency_matrix)


def dbscan_metrics(true_labels, cur_labels):
    """
    the evaluate function of clustering algorithm
    :param true_labels: accurate labels
    :param cur_labels: clustering results
    :return: normalized mutual info, adjusted mutual info, adjusted rand info
    """

    d_nmi = round(
        metrics.normalized_mutual_info_score(true_labels, cur_labels), 4)
    d_ami = round(metrics.adjusted_mutual_info_score(true_labels, cur_labels),
                  4)
    d_ari = round(metrics.adjusted_rand_score(true_labels, cur_labels), 4)

    return d_nmi, d_ami, d_ari


def inner_metrics(features, cur_labels):
    """
    the evaluate function of clustering algorithm
    :param cur_labels: clustering results
    :return: normalized mutual info, adjusted mutual info, adjusted rand info
    """

    ch = metrics.calinski_harabasz_score(features, cur_labels)

    return ch