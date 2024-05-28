import networkx as nx


# 加载网络
def load_graph(p, c):
    G = nx.Graph()
    for i in range(len(p)):
        source = p[i]
        target = c[i]
        G.add_edge(source, target)
    return G


# 克隆
def clone_graph(G):
    cloned_graph = nx.Graph()
    for edge in G.edges():
        cloned_graph.add_edge(edge[0], edge[1])
    return cloned_graph


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


class GN(object):
    """docstring for GN"""

    def __init__(self, G):
        self._G_cloned = clone_graph(G)
        self._G = G
        self._partition = [[n for n in G.nodes()]]
        self._max_Q = 0.0

    # GN算法
    def execute(self):
        while len(self._G.edges()) > 0:
            # 1.计算所有边的edge betweenness
            edge = max(nx.edge_betweenness_centrality(self._G).items(),
                       key=lambda item: item[1])[0]
            # 2.移去edge betweenness最大的边
            self._G.remove_edge(edge[0], edge[1])
            # 获得移去边后的子连通图
            components = [list(c) for c in list(nx.connected_components(self._G))]
            if len(components) != len(self._partition):
                # 3.计算Q值
                cur_Q = cal_Q(components, self._G_cloned)
                if cur_Q > self._max_Q:
                    self._max_Q = cur_Q
                    self._partition = components
        return self._partition