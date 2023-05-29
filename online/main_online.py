import collections
import sys
import os
import warnings
import argparse
from time import localtime, strftime
import torch
from utils.utils import *
from sklearn.cluster import DBSCAN
from model.model import DrlDbscan
from transformers.trainer_utils import set_seed
from torch_scatter import scatter_sum
from tqdm import tqdm
from tqdm.auto import trange
from utils.pre_partition import se_pre_partition
import logging

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',
                    default='Stream-Powersupply',
                    type=str,
                    help="Path of features and labels")
parser.add_argument('--log_path',
                    default='results/test',
                    type=str,
                    help="Path of results")

# Model dependent args
parser.add_argument('--use_cuda',
                    default=True,
                    action='store_true',
                    help="Use cuda")
parser.add_argument('--train_size',
                    default=0.20,
                    type=float,
                    help="Sample size used to get rewards")
parser.add_argument(
    '--episode_num', default=15, type=int,
    help="The number of episode")
parser.add_argument('--block_num',
                    default=8,
                    type=int,
                    help="The number of data blcoks")  # Offline: 1, Online: 8
parser.add_argument('--block_size',
                    default=3741,
                    type=int,
                    help="The size of data block")  # Offline: -, Online: 3741
parser.add_argument(
    '--layer_num', default=6, type=int,
    help="The number of recursive layer")  # Offline: 3, Online: 6
parser.add_argument('--eps_size',
                    default=5,
                    type=int,
                    help="Eps parameter space size")
parser.add_argument('--min_size',
                    default=4,
                    type=int,
                    help="MinPts parameter space size")
parser.add_argument('--reward_factor',
                    default=0.2,
                    type=float,
                    help="The impact factor of reward")

# TD3 args
parser.add_argument('--device',
                    default="cpu",
                    type=str,
                    help='"cuda" if torch.cuda.is_available() else "cpu".')
parser.add_argument('--batch_size',
                    default=16,
                    type=int,
                    help='"Reinforcement learning for sampling batch size')
parser.add_argument('--step_num',
                    default=30,
                    type=int,
                    help="Maximum number of steps per RL game")
parser.add_argument('--seed', default=41, type=int, help="Random seed")


def DRL_DBSCAN(args, features, labels, reward_mask, agent_id, block_id):
    """
    features: [feature[0], .., feature[block_num-1]], feature[i] is a numpy array [dataset_size, feature_size]
    """
    param_size, param_step, param_center, param_bound = generate_parameter_space(
        features[0], args.layer_num, args.eps_size, args.min_size,
        args.dataset)
    # build a multi-layer agent collection, each layer has an independent agent
    agents = []
    for l in range(0, args.layer_num):
        drl = DrlDbscan(param_size, param_step[l], param_center, param_bound,
                        args.device, args.batch_size, args.step_num,
                        features[0].shape[1])
        agents.append(drl)

    best_cluster_log = []
    # Train agents with serialized data blocks
    for b in range(1):
        # compare with the result of Kmeans
        k_nmi = kmeans_metrics(features[b], labels[b])

        final_reward_test = [0, param_center, 0]
        label_dic_test = set()

        # test each layer agent
        with trange(args.layer_num, leave=False) as test_iter:
            for l in test_iter:
                test_iter.set_description(
                    f'Agent {agent_id} Block {block_id} Test layer {l}')
                agent = agents[l]
                agent.reset(final_reward_test)

                # testing
                cur_labels, cur_cluster_num, param_log = agent.detect(
                    features[b], collections.OrderedDict())
                final_reward_test = [0, param_log[-1], 0]
                d_nmi, d_ami, d_ari = dbscan_metrics(labels[b], cur_labels)

                # update log
                for p in param_log:
                    label_dic_test.add(str(p[0]) + str("+") + str(p[1]))

        max_max_reward = [0, param_center, 0]
        max_reward = [0, param_center, 0]
        label_dic = collections.OrderedDict()
        first_meet_num = 0

        # train each layer agent
        with trange(args.layer_num) as train_iter:
            for l in train_iter:
                train_iter.set_description(
                    f'Agent {agent_id} Block {block_id} Train layer {l}')
                if l == 0:
                    max_nmi = -1
                agent = agents[l]
                agent.reset(max_max_reward)
                max_max_reward_logs = [max_max_reward[0]]
                early_stop = False
                his_hash_size = len(label_dic)
                cur_hash_size = len(label_dic)
                episode_iter = tqdm(range(1, args.episode_num))
                episode_iter.desc = f'Agent {agent_id} Block {block_id} Training Layer {l}'
                episode_iter.ncols = 150
                for i in episode_iter:
                    # begin training process
                    episode_iter.set_postfix(
                        {'label hash size': len(label_dic)})
                    param_logs = np.array([[], []])
                    nmi_logs = np.array([])

                    # update starting point
                    agent.reset0()

                    # train the l-th layer
                    cur_labels, cur_cluster_num, param_log, nmi_log, max_reward, max_nmi, best_cluster_log = agent.train(
                        i,
                        reward_mask[b],
                        features[b],
                        labels[b],
                        label_dic,
                        args.reward_factor,
                        max_nmi,
                        best_cluster_log,
                        log_flag=False)

                    # input()

                    # update log
                    param_logs = np.hstack(
                        (param_logs, np.array(list(zip(*param_log)))))
                    nmi_logs = np.hstack((nmi_logs, np.array(nmi_log)))
                    d_nmi, d_ami, d_ari = dbscan_metrics(labels[b], cur_labels)
                    if max_max_reward[0] < max_reward[0]:
                        max_max_reward = list(max_reward)
                        cur_hash_size = len(label_dic)
                    max_max_reward_logs.append(max_max_reward[0])

                    # test each layer agent once again

                    # update starting point
                    agent.reset0()
                    cur_labels, cur_cluster_num, param_log = agent.detect(
                        features[b], label_dic)
                    d_nmi, d_ami, d_ari = dbscan_metrics(labels[b], cur_labels)
                    episode_iter.set_postfix({
                        'DBSCAN NMI': d_nmi,
                        'DBSCAN AMI': d_ami,
                        'DBSCAN ARI': d_ari
                    })
                    # early stop
                    if len(max_max_reward_logs) > 3 and \
                            max_max_reward_logs[-1] == max_max_reward_logs[-2] == max_max_reward_logs[-3] and \
                            max_max_reward_logs[-1] != max_max_reward_logs[0]:
                        break
                first_meet_num += cur_hash_size - his_hash_size
                if cur_hash_size == his_hash_size:
                    print("......Early stop at layer {0}......".format(l),
                          flush=True)
                    break
        cur_labels = label_dic[str(max_max_reward[1][0]) + str("+") +
                               str(max_max_reward[1][1])]
        cur_cluster_num = len(set(list(cur_labels)))
        d_nmi, d_ami, d_ari = dbscan_metrics(labels[b], cur_labels)
        return cur_labels, cur_cluster_num


# Todo: add multi Block training
if __name__ == '__main__':
    # load hyper-parameters
    args = parser.parse_args()
    seed_pool = list(range(50))
    nmi_list, ami_list, ari_list = [[] for _ in range(args.block_num)
                                    ], [[] for _ in range(args.block_num)
                                        ], [[] for _ in range(args.block_num)]
    purity_list = [[] for _ in range(args.block_num)]
    for seed in seed_pool:
        set_seed(seed)
        # CUDA
        use_cuda = args.use_cuda and torch.cuda.is_available()
        print("Using CUDA:  " + str(use_cuda), flush=True)
        print("Running on:  " + str(args.dataset), flush=True)

    
        reward_mask, features, labels = load_data_stream(
            f'/datasets/SAR_DBSCAN/{args.dataset}.txt', args.train_size,
            args.block_num, args.block_size)
        # for b in range(args.block_num):
        for b in range(args.block_num):
            # pre agent find
            se_label, uncertainty, se_label_with_noise = se_pre_partition(
                features[b])
            print('uncertainty: ', uncertainty, flush=True)
            se_nmi, se_ami, se_ari = dbscan_metrics(labels[b],
                                                    se_label_with_noise)
            print('se_nmi: ', se_nmi, flush=True)
            print('se_ami: ', se_ami, flush=True)
            print('se_ari: ', se_ari, flush=True)
            pre_partition = DBSCAN(eps=0.3, min_samples=1).fit_predict(
                uncertainty.reshape(-1, 1))
            # pre_partition = torch.arange(uncertainty.shape[0])
            print('pre partition: ', pre_partition, flush=True)

            for i, e in enumerate(pre_partition):
                se_label[se_label == i] = e
            print('agent number: ', pre_partition.max() + 1, flush=True)
            data = []
            labels_ = []
            reward_mask_ = []
            original_id = torch.arange(features[b].shape[0])
            id_idx_ = []
            if pre_partition.max() == -1:
                data.append(features[b])
                labels_.append(labels[b])
                reward_mask_.append(reward_mask[b])
                id_idx_.append(original_id)
            else:
                for id in range(pre_partition.max() + 1):
                    data.append(features[b][se_label == id])
                    labels_.append(labels[b][se_label == id])
                    id_idx = (se_label == id).nonzero().squeeze(
                    )  # index of points in partition
                    id_idx_.append(id_idx)
                    combined, count = torch.cat(
                        (id_idx, torch.tensor(reward_mask[b])),
                        dim=0).unique(return_counts=True)
                    partition_original_id = original_id[
                        se_label == id]  # original id of partition
                    partition_mask = combined[
                        count > 1]  # mask id need to suit partition
                    partition_mask = (partition_mask.view(
                        1, -1) == partition_original_id.view(
                            -1, 1)).any(dim=1).nonzero().squeeze(
                            )  # correct mask id in partition
                    # print(partition_mask.shape)
                    if partition_mask.shape == torch.Size([]):
                        # if no mask in partition, randomly select two points
                        partition_mask = torch.tensor(
                            random.sample(range(data[-1].shape[0]), 2))
                    elif partition_mask.shape[0] == 1:
                        # duplicate mask to avoid sklearn nmi error
                        partition_mask = torch.cat(
                            (partition_mask, partition_mask))
                    reward_mask_.append(partition_mask)
            final_label = []
            label_max = 0
            for agent_id, feature_i, label_i, reward_mask_i in zip(
                    range(len(data)), data, labels_, reward_mask_):
                feature_i = [feature_i]
                label_i = [label_i]
                reward_mask_i = [reward_mask_i]
                cur_label, cur_cluster_num = DRL_DBSCAN(
                    args, feature_i, label_i, reward_mask_i, agent_id, b)
                cur_label[cur_label != -1] += label_max
                label_max = cur_label.max() + 1
                final_label.append(torch.tensor(cur_label))

            final_label = torch.cat(final_label, dim=0)
            id_idx_ = torch.cat(id_idx_, dim=0)
            final_label = torch.zeros_like(torch.tensor(labels[b])).scatter_(
                0, id_idx_, final_label)
            final_label = final_label.numpy()
            d_nmi, d_ami, d_ari = dbscan_metrics(labels[b], final_label)
            purity = purity_score(labels[b], final_label)
            nmi_list[b].append(d_nmi)
            ami_list[b].append(d_ami)
            ari_list[b].append(d_ari)
            purity_list[b].append(purity)
            print(
                f'V {b+1}: NMI {d_nmi:.4f}, AMI {d_ami:.4f}, ARI {d_ari:.4f}, Purity {purity:.4f}',
                flush=True)
    print(nmi_list)
    print(ami_list)
    print(ari_list)
    print(purity_list)
    print('NMI: ', [np.mean(nmi_list[i]) for i in range(args.block_num)])
    print('AMI: ', [np.mean(ami_list[i]) for i in range(args.block_num)])
    print('ARI: ', [np.mean(ari_list[i]) for i in range(args.block_num)])
    print('Purity: ', [np.mean(purity_list[i]) for i in range(args.block_num)])