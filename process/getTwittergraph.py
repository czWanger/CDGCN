# -*- coding: utf-8 -*-
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
from GN import *
import copy
from communities.algorithms import louvain_method
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain
cwd = os.getcwd()


class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None


def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        index = int(pair.split(':')[0])
        freq = float(pair.split(':')[1])
        if index <= 5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

# 构建会话树
def constructMat(tree):
    index2node = {}
    # 构建树节点列表 i>=1
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    # 填充每个节点内容
    for j in tree:
        indexC = j  # 孩子节点索引
        indexP = tree[j]['parent']  # 父节点索引
        nodeC = index2node[indexC]  # 孩子节点内容
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex  # 词索引列表
        nodeC.word = wordFreq  # 词频列表
        # print("NODEC:", nodeC.index, "END")
        # print("INDEX2NODE:", index2node[indexC].index)
        # 如果不是根节点
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        # 如果是根节点
        else:
            rootindex = indexC - 1  # 根节点索引
            root_index = nodeC.index
            root_word = nodeC.word
    # 根特征向量X
    rootfeat = np.zeros([1, 5000])
    if len(root_index) > 0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)  # 索引和权重如何对应
        # print("I:", np.array(root_index).shape, "W:", np.array(root_word).shape)
    row = []
    col = []
    x_word = []
    x_index = []
    print(len(index2node))
    for index_i in range(len(index2node)):
        print(index_i)
        for index_j in range(len(index2node)):
            if index2node[index_i + 1].children is not None and index2node[index_j + 1] in index2node[index_i + 1].children:
                row.append(index_i)
                col.append(index_j)
        x_word.append(index2node[index_i + 1].word)
        x_index.append(index2node[index_i + 1].index)
    edgematrix = [row, col]  # [被转节点，转发节点]

    # 所有词频 所有词索引 有向图的边矩阵E 根特征向量X 根节点索引
    return x_word, x_index, edgematrix, rootfeat, rootindex


# 每一条推文的特征向量组成的事件特征
def getfeature(x_word, x_index):
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i]) > 0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x


def divisionCom(tree):
    row = tree[0]
    col = tree[1]
    # networkX构图
    G = load_graph(row, col)
    G_ = copy.deepcopy(G)
    # 计算社区
    algo = GN(G)
    partition = algo.execute()
    new_row = []
    new_col = []
    for part in partition:
        sg = G_.subgraph(part)
        for edges in list(sg.edges):
            new_row.append(edges[0])
            new_col.append(edges[1])
    return [new_row, new_col]

# def louvainDiv(tree,x_x):
#     row = tree[0]
#     col = tree[1]
#     # networkX构图
#     G = load_graph(row, col)
#     A = np.array(nx.adjacency_matrix(G).todense())
#
#     # # 计算节点之间的余弦相似度
#     #
#     # cosine_similarities = cosine_similarity(x_x)*10
#     #
#     # # 将余弦相似度替换为邻接矩阵中的对应位置
#     # A[np.where(A == 1)] = cosine_similarities[np.where(A == 1)]
#
#     communities, _ = louvain_method(A)
#     new_row = []
#     new_col = []
#     for group in communities:
#         nodes = list(group)
#         subG = nx.subgraph(G, nodes)
#         for e in subG.edges():
#             new_row.append(e[0])
#             new_col.append(e[1])
#     return [new_row, new_col]


import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain


def louvainDiv(tree, x_x):
    row = tree[0]
    col = tree[1]

    # networkX构图
    G = nx.Graph()
    G.add_edges_from(zip(row, col))

    # 计算节点之间的余弦相似度
    cosine_similarities = cosine_similarity(x_x)

    # 创建新的带权重的图
    G_weighted = nx.Graph()
    G_weighted.add_nodes_from(G.nodes())

    # 复制原图中的边到新图中，并设置权重
    for u, v in G.edges():
        weight_i = cosine_similarities[u][v] * 10
        G_weighted.add_edge(u, v, weight=weight_i)

    # 使用Louvain算法检测社区
    partition = community_louvain.best_partition(G_weighted, weight='weight')
    new_row = []
    new_col = []

    # 遍历每个社区
    for community_id in set(partition.values()):
        # 获取社区内的子图
        subgraph_nodes = [node for node, comm_id in partition.items() if comm_id == community_id]
        subgraph = G_weighted.subgraph(subgraph_nodes)

        # 遍历子图的边，添加到新的边列表中
        for u, v, data in subgraph.edges(data=True):
            new_row.append(u)
            new_col.append(v)

    return [new_row, new_col]


def main():
    treePath = '../data/Twitter15/data.TD_RvNN.vol_5000.txt'
    """
    文件结构：
    事件ID 父节点索引 子节点索引 最大度 最大长度 特征向量
    """
    print("reading twitter tree")
    treeDic = {}
    """
    treeDic: Dic[str, dict] = {} 事件id, 事件
    treeDic[eid]: Dic[str, dict] = {} 孩子节点id, 节点信息
    treeDic[eid][indexC]: Dic[str, dict] = {} 节点属性名，值
    """
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]

        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    print('tree no:', len(treeDic))

    labelPath = '../data/Twitter15/' + 'Twitter15' + "_label_All.txt"
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']

    print("loading tree label")
    event, y = [], []
    l1 = l2 = l3 = l4 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        label = label.lower()
        event.append(eid)
        if label in labelset_nonR:
            labelDic[eid] = 0
            l1 += 1
        if label in labelset_f:
            labelDic[eid] = 1
            l2 += 1
        if label in labelset_t:
            labelDic[eid] = 2
            l3 += 1
        if label in labelset_u:
            labelDic[eid] = 3
            l4 += 1
    print(len(labelDic))  # 事件总数
    print(l1, l2, l3, l4)  # 四个标签对应的事件数量 374 370 372 374

    def loadEid(event, id, y):
        """
        :param event: {1: {'parent': 'None', 'max_degree': 3, 'maxL': 24, 'vec': '384:2 1:1 2:1 3:2 38:1 1706:1 290:1 1964:1 162:1 14:1 16:1 177:1 1907:1 23:1 24:1 831:1'}, 30: {'parent': '1', 'max_degree': 3, 'maxL': 24, 'vec': '0:1 367:1'},...}
        :param id:377519445578895360
        :param y:1
        :return:
        """
        if event is None:
            return None
        if len(event) < 2:
            #  独立推文，无转发关系
            return None
        if len(event) > 1:
            x_word, x_index, tree, rootfeat, rootindex = constructMat(event)

            x_x = getfeature(x_word, x_index)
            division = louvainDiv(tree,x_x)
            # division_2 = divisionCom(tree)
            rootfeat, tree, x_x, rootindex, y, division = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                rootindex), np.array(y), np.array(division)
            # division_2 = np.array(division_2)
            np.savez('../data/Twitter15/louvaindiv_cos/' + id + '.npz', x=x_x, root=rootfeat, edgeindex=tree, rootindex=rootindex, y=y,
                     division=division)
            return None

    print("loading dataset", )
    Parallel(n_jobs=80, backend='threading')(
        delayed(loadEid)(treeDic[eid] if eid in treeDic else None, eid, labelDic[eid]) for eid in tqdm(event))
    return


if __name__ == '__main__':
    main()
