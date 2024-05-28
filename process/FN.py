import networkx as nx
import pandas as pd
import numpy as np
from sklearn import metrics

# 使用矩阵计算Q值
# label: 为每个节点划分的community编号
def cal_Q_mat(A_mat, labels):
    S = pd.get_dummies(labels) # 类别标签
    m = sum(sum(A_mat))/2
    k = A_mat.sum(axis=1, keepdims=True)
    B = A_mat - (np.tile(k, (1, len(A_mat))) * np.tile(k.T, (len(A_mat), 1))) / (2 * m)
    Q = 1 / (2 * m) * np.trace(S.T @ B @ S)  # @等价于np.matmul点乘
    return Q


# 计算Q值
def cal_Q(partition, G):
    m = len(list(G.edges()))
    a = []
    e = []

    # 计算每个社区的a值
    for community in partition:
        t = 0
        for node in community:
            t += len(list(G.neighbors(node)))
        a.append(t / float(2 * m))

    # 计算每个社区的e值
    for community in partition:
        t = 0
        for i in range(len(community)):
            for j in range(len(community)):
                if i != j:
                    if G.has_edge(community[i], community[j]):
                        t += 1
        e.append(t / float(2 * m))

    # 计算Q
    q = 0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q

# 克隆
def clone_graph(G):
    cloned_graph = nx.Graph()
    for edge in G.edges():
        cloned_graph.add_edge(edge[0], edge[1])
    return cloned_graph


# 计算两个类别标签的NMI指数
class FN(object):
    def __init__(self, G):
        self._G_cloned = clone_graph(G)
        self._G = G
        self._partition = [[n for n in G.nodes()]]
        self._max_Q = 0.0
        self._Group = [[n for n in G.nodes()]]
        self._Node_GroupIndex = self._G.nodes()
        for i in self._Node_GroupIndex:
            self._Group.append([i])

    # FN算法
    def execute(self):
        while len(self._Group) > 1:
            det_Q = float('-inf')
            max_edge = None
            for edge in self._G.edges():
                index_i = self._Node_GroupIndex[edge[0]]
                index_j = self._Node_GroupIndex[edge[1]]


