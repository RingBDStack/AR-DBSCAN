import numpy as np
from sklearn.cluster import DBSCAN
from model.TD3 import Skylark_TD3, ReplayBuffer
from model.environment import get_reward, get_state, convergence_judgment
from utils.utils import *

class DrlDbscan:
    r"""
    define our deep-reinforcement learning dbscan class
    """

    def __init__(self, param_size, param_step, param_center, param_bound,
                 device, batch_size, step_num, dim):
        r"""
        initialize the reinforcement learning agent
        :param param_size: parameter space size
        :param param_step: step size
        :param param_center: starting point of the first layer
        :param param_bound: limit bound for the two parameter spaces
        :param device: cuda
        :param batch_size: batch size
        :param step_num: Maximum number of steps per RL game
        :param dim: dimension of feature
        """

        # TD3
        self.agent = Skylark_TD3(global_state_dim=7,
                                 local_state_dim=dim + 2,
                                 action_dim=5,
                                 max_action=1.0,
                                 device_setting=device)
        self.replay_buffer = ReplayBuffer(action_dim=5)
        self.batch_size = batch_size
        self.step_num = step_num

        # parameter space: param_center, param_size, param_step, param_bound0
        self.param_center = list(param_center)
        self.param_size, self.param_step, self.param_bound0 = list(
            param_size), list(param_step), list(param_bound)
        self.param_bound = self.get_parameter_space()

        # cur_param: current parameter
        self.cur_param = list(param_center)
        # logs for score, reward, parameter, action, nmi
        self.state_log, self.reward_log, self.im_reward_log, self.param_log, self.action_log, self.nmi_log = [], [], [], \
                                                                                                         [], [], []

        self.max_reward = [0, list(param_center), 0]

    def reset0(self):
        """
        update the new parameter space and related records
        """
        self.cur_param = list(self.param_center)
        self.state_log, self.reward_log, self.im_reward_log, self.param_log, self.action_log, self.nmi_log = [], [], [], \
                                                                                                        [], [], []

        # print("The starting point of the parameter is:  " +
        #       str(self.param_center),
        #       flush=True)
        # print("The parameter space boundary is:  " + str(self.param_bound),
        #       flush=True)
        # print("The size of the parameter space is:  " + str(self.param_size),
        #       flush=True)
        # print("The step of the parameter space is:  " + str(self.param_step),
        #       flush=True)

    def reset(self, max_reward):
        """
        reset the environment
        :param max_reward: record the max reward
        """

        self.param_center = list(max_reward[1])
        self.param_bound = self.get_parameter_space()
        self.cur_param = list(max_reward[1])
        self.state_log, self.reward_log, self.im_reward_log, self.param_log, self.action_log, self.nmi_log = [], [], [], \
                                                                                                        [], [], []
        self.max_reward = list(max_reward)

        # print("The starting point of the parameter is:  " +
        #       str(self.param_center),
        #       flush=True)
        # print("The parameter space boundary is:  " + str(self.param_bound),
        #       flush=True)
        # print("The size of the parameter space is:  " + str(self.param_size),
        #       flush=True)
        # print("The step of the parameter space is:  " + str(self.param_step),
        #       flush=True)

    def get_parameter_space(self):
        """
        get parameter space of the current layer
        :return: parameter space
        """

        param_bound = [[
            max(
                self.param_center[i] -
                self.param_step[i] * int(self.param_size[i] / 2),
                self.param_bound0[i][0]),
            min(
                self.param_center[i] +
                self.param_step[i] * int(self.param_size[i] / 2),
                self.param_bound0[i][1])
        ] for i in range(2)]

        return param_bound

    def action_to_parameters(self, cur_param, action):
        """
        Translate reinforcement learning output actions into specific parameters
        :param cur_param: current parameter
        :param action: current action, shape [1, 5]
        :return: parameters and bump flags
        """

        bump_flag = [0, 0]
        new_param = [0, 1]
        new_action = [0, 0, 0, 0, 0]
        new_action[action.argmax(axis=0)] = 1

        # if the action goes beyond the parameter space, the parameters remain unchanged
        # action: {left, right, up, down, stop}
        # parameter: {eps, MinPts}
        for i in range(2):
            new_param[i] = cur_param[i] - self.param_step[i] * new_action[0 + 2 * i] + \
                       self.param_step[i] * new_action[1 + 2 * i]
            # bump flags for helping judge whether our new parameters out of space
            # 0: not bump, 1: bump upper bound, -1: bump lower bound
            if new_param[i] < self.param_bound[i][0]:
                new_param[i] = self.param_bound[i][0]
                bump_flag[i] = -1
            elif new_param[i] > self.param_bound[i][1]:
                new_param[i] = self.param_bound[i][1]
                bump_flag[i] = 1
        return new_param, bump_flag

    def stop_processing(self, buffer, final_factor, max_factor):
        """
        Sample training data and store
        :param buffer: store historical data for training
        :param final_factor: reward_factor
        :param max_factor: 1 - reward_factor
        """
        buffer.append([
            self.state_log[-2], self.action_log[-1], self.state_log[-1],
            self.reward_log[-1], self.im_reward_log[-1]
        ])
        final_reward = buffer[-1][4]  # immediate reward
        post_max_reward = buffer[-1][4]
        for bu in reversed(buffer):
            post_max_reward = max(post_max_reward, bu[4])
            bu[3] = final_factor * final_reward + max_factor * post_max_reward
        # print([bu[3] for bu in buffer])

        for bu in buffer[:-1]:
            self.replay_buffer.add(bu[0], bu[1], bu[2], bu[3], float(0))
        self.replay_buffer.add(buffer[-1][0], buffer[-1][1], buffer[-1][2],
                               buffer[-1][3], float(1))

        # train agent
        if self.replay_buffer.size >= self.batch_size:
            for _ in range(len(buffer)):
                self.agent.learn(self.replay_buffer, self.batch_size)

    def train(self,
              episode_i,
              extract_masks,
              extract_features,
              extract_labels,
              label_dic,
              reward_factor,
              max_nmi=-1,
              best_cluster_log=None,
              log_flag=False):
        """
        Train DRL-DBSCAN: RL searching for parameters
        :param episode_i: episode_num
        :param extract_masks: sample serial numbers for rewards
        :param extract_features: features
        :param extract_labels: labels
        :param label_dic: records for parameters and its clustering results (cur_labels)
        :param reward_factor: factors for final reward

        :return: cur_labels, cur_cluster_num, self.param_log, self.nmi_log, self.max_reward
        """

        extract_data_num = extract_features.shape[0]

        # DBSCAN clustering
        # if the parameters have been searched before, we can directly get the clustering results
        # label_dic: {parameters: labels}
        if str(self.cur_param[0]) + str("+") + str(
                self.cur_param[1]) in label_dic:
            cur_labels = label_dic[str(self.cur_param[0]) + str("+") +
                                   str(self.cur_param[1])]
        else:
            cur_labels = DBSCAN(
                eps=self.cur_param[0],
                min_samples=self.cur_param[1]).fit_predict(extract_features)
            label_dic[str(self.cur_param[0]) + str("+") +
                      str(self.cur_param[1])] = np.array(cur_labels)
            if log_flag:
                assert best_cluster_log is not None
                local_nmi, local_ami, local_ari = dbscan_metrics(
                    extract_labels, cur_labels)
                local_purity = purity_score(extract_labels, cur_labels)
                local_nmi = metrics.normalized_mutual_info_score(
                    extract_labels[extract_masks], cur_labels[extract_masks])
                if local_nmi > max_nmi:
                    max_nmi = local_nmi
                    best_cluster_log.append(cur_labels)
                else:
                    best_cluster_log.append(best_cluster_log[-1])

        cur_cluster_num = len(set(list(cur_labels)))

        # Get state: [global state, local state]
        # global state: [[eps, eps-eps_lowbound, eps_upbound-eps, minPts, minPts-minPts_lowbound, minPts_upbound-minPts, cluster_ratio]]
        # local state: [[||feature_center, cluster_center||, cluster size,cluster_center] for _ in cluster_num]
        state = get_state(extract_features, cur_labels, cur_cluster_num,
                          extract_data_num, self.cur_param, [0, 0],
                          self.param_bound)

        bump_flag = [0, 0]
        buffer = []

        final_factor = reward_factor
        max_factor = 1 - reward_factor
        # begin RL game
        for e in range(self.step_num):

            self.state_log.append(state)

            # early stop
            if e >= 2 and convergence_judgment(self.action_log[-1]):
                self.stop_processing(buffer, final_factor, max_factor)
                # print("! Early stop.", flush=True)
                break
            # out of bounds stop
            elif bump_flag != [0, 0]:
                self.stop_processing(buffer, final_factor, max_factor)
                # print("! Out of bounds stop.", flush=True)
                break
            # Timeout stop
            elif e == self.step_num - 1:
                self.stop_processing(buffer, final_factor, max_factor)
                # print("! Timeout stop.", flush=True)
                break
            # play the game
            else:
                if e != 0:
                    buffer.append([
                        self.state_log[-2], self.action_log[-1],
                        self.state_log[-1], self.reward_log[-1],
                        self.im_reward_log[-1]
                    ])

                # train
                # predict actions and obtain parameters based on state
                real_action = self.agent.select_action(self.state_log[-1])
                # print(real_action)
                if episode_i == 1:
                    new_action = (real_action).clip(0, self.agent.max_action)
                else:
                    new_action = (real_action + np.random.normal(
                        0,
                        self.agent.max_action * self.agent.expl_noise,
                        size=self.agent.action_dim)).clip(
                            0, self.agent.max_action)  # add noise
                new_param, bump_flag = self.action_to_parameters(
                    self.cur_param, new_action)

            if str(new_param[0]) + str("+") + str(new_param[1]) in label_dic:
                cur_labels = label_dic[str(new_param[0]) + str("+") +
                                       str(new_param[1])]
            else:
                cur_labels = DBSCAN(
                    eps=new_param[0],
                    min_samples=new_param[1]).fit_predict(extract_features)
                label_dic[str(new_param[0]) + str("+") +
                          str(new_param[1])] = np.array(cur_labels)
                if log_flag:
                    assert best_cluster_log is not None
                    local_nmi, local_ami, local_ari = dbscan_metrics(
                        extract_labels, cur_labels)
                    local_purity = purity_score(extract_labels, cur_labels)
                    local_nmi = metrics.normalized_mutual_info_score(
                        extract_labels[extract_masks],
                        cur_labels[extract_masks])
                    if local_nmi > max_nmi:
                        max_nmi = local_nmi
                        best_cluster_log.append(cur_labels)
                    else:
                        best_cluster_log.append(best_cluster_log[-1])

            cur_cluster_num = len(set(list(cur_labels)))
            state = get_state(extract_features, cur_labels, cur_cluster_num,
                              extract_data_num, new_param, bump_flag,
                              self.param_bound)

            # get reward
            reward, nmi, im_reward = get_reward(
                extract_features, extract_labels, cur_labels, cur_cluster_num,
                extract_data_num, extract_masks, bump_flag, buffer, e)

            # log
            self.cur_param = list(new_param)
            self.param_log.append(new_param)
            self.action_log.append(new_action)
            self.reward_log.append(reward)
            self.im_reward_log.append(im_reward)
            self.nmi_log.append(nmi)

        if max(self.im_reward_log) > self.max_reward[0]:
            self.max_reward = [
                max(self.im_reward_log),
                self.param_log[self.im_reward_log.index(max(
                    self.im_reward_log))],
                self.nmi_log[self.im_reward_log.index(max(self.im_reward_log))]
            ]
        # print(
        #     "! The current maximum reward {0} appears when the parameter is {1}."
        #     .format(self.max_reward[2], self.max_reward[1]),
        #     flush=True)
        return cur_labels, cur_cluster_num, self.param_log, self.nmi_log, self.max_reward, max_nmi, best_cluster_log

    def detect(self,
               extract_features,
               label_dic,
               max_nmi=-1,
               best_cluster_log=None,
               log_flag=False):
        """
        Detect DRL-DBSCAN:
        :param extract_features: features
        :param label_dic: records for parameters and its clustering results (cur_labels)

        :return: cur_labels, cur_cluster_num, param_log
        """

        extract_data_num = extract_features.shape[0]

        # DBSCAN clustering
        if str(self.cur_param[0]) + str("+") + str(
                self.cur_param[1]) in label_dic:
            cur_labels = label_dic[str(self.cur_param[0]) + str("+") +
                                   str(self.cur_param[1])]
        else:
            cur_labels = DBSCAN(
                eps=self.cur_param[0],
                min_samples=self.cur_param[1]).fit_predict(extract_features)
        cur_cluster_num = len(set(list(cur_labels)))

        # Get state
        state = get_state(extract_features, cur_labels, cur_cluster_num,
                          extract_data_num, self.cur_param, [0, 0],
                          self.param_bound)

        # begin RL game
        param_log = []
        action_log = []
        bump_flag = [0, 0]
        cur_param = self.cur_param

        for e in range(self.step_num):
            # early stop
            if e >= 2 and convergence_judgment(action_log[-1]):
                # print("! Early stop.", flush=True)
                break
            # out of bounds stop
            elif bump_flag != [0, 0]:
                # print("! Out of bounds stop.", flush=True)
                break
            # Timeout stop
            elif e == self.step_num - 1:
                # print("! Timeout stop.", flush=True)
                break
            # play the game
            else:
                # predict actions and obtain parameters based on state
                new_action = (self.agent.select_action(state)).clip(
                    0, self.agent.max_action)
                new_param, bump_flag = self.action_to_parameters(
                    cur_param, new_action)

            # get new state
            if str(new_param[0]) + str("+") + str(new_param[1]) in label_dic:
                cur_labels = label_dic[str(new_param[0]) + str("+") +
                                       str(new_param[1])]
            else:
                cur_labels = DBSCAN(
                    eps=new_param[0],
                    min_samples=new_param[1]).fit_predict(extract_features)
            cur_cluster_num = len(set(list(cur_labels)))

            state = get_state(extract_features, cur_labels, cur_cluster_num,
                              extract_data_num, new_param, bump_flag,
                              self.param_bound)
            cur_param = list(new_param)
            param_log.append(new_param)
            action_log.append(new_action)
        # print(action_log)
        # print(param_log)
        # print("! Stop at step {0} with parameter {1}.".format(
        #     e, str(cur_param)),
        #       flush=True)

        return cur_labels, cur_cluster_num, param_log
