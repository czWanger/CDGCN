import sys,os
import torch as th
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DNAConv
import copy
from math import sqrt
import torch
import torch.nn as nn

sys.path.append(os.getcwd())


class SelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: n, dim_in
        n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # n, dim_k
        k = self.linear_k(x)  # n, dim_k
        v = self.linear_v(x)  # n, dim_v

        dist = torch.matmul(q, k.T) * self._norm_fact  # n, n
        dist = torch.softmax(dist, dim=-1)  # n, n

        att = torch.matmul(dist, v)
        return att


class TDrumorGCN(th.nn.Module):
    def __init__(self, args):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(args.input_features, args.hidden_features)
        self.fc = th.nn.Linear(args.input_features, args.output_features)
        self.device = args.device

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        root_extend = self.fc(root_extend)
        x = th.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x


class BUrumorGCN(th.nn.Module):
    def __init__(self,args):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(args.input_features, args.hidden_features)
        self.fc = th.nn.Linear(args.input_features, args.output_features)
        self.device = args.device

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        root_extend = self.fc(root_extend)
        x = th.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x


class CommunityGCN(th.nn.Module):
    def __init__(self, args):
        super(CommunityGCN, self).__init__()
        self.conv1 = GCNConv(args.input_features, args.hidden_features + args.output_features)
        self.device = args.device

    def forward(self, data):
        x, division = data.x, data.BUdiv_edge_index
        x = self.conv1(x, division)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x


class CFGCN(th.nn.Module):
    def __init__(self, args):
        super(CFGCN, self).__init__()
        self.CommunityGCN = CommunityGCN(args)
        self.TDrumorGCN = TDrumorGCN(args)
        self.fc = th.nn.Linear(2*(args.hidden_features + args.output_features), args.num_class)

    def forward(self, data):
        C_x = self.CommunityGCN(data)
        TD_x = self.TDrumorGCN(data)
        x = th.cat((C_x, TD_x), 1)
        x = scatter(x, data.batch, dim=0, reduce='mean')
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x