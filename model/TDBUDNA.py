import sys,os
import torch as th
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DNAConv
import copy
import torch

sys.path.append(os.getcwd())


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


class TDgroupGCN(th.nn.Module):
    def __init__(self, args):
        super(TDgroupGCN, self).__init__()
        self.conv1 = GCNConv(args.input_features, args.output_features+args.hidden_features)
        self.device = args.device

    def forward(self, data):
        x, division = data.x, data.TDdiv_edge_index
        x = self.conv1(x, division)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x


class BUgroupGCN(th.nn.Module):
    def __init__(self, args):
        super(BUgroupGCN, self).__init__()
        self.conv1 = GCNConv(args.input_features, args.output_features+args.hidden_features)
        self.device = args.device

    def forward(self, data):
        x, division = data.x, data.BUdiv_edge_index
        x = self.conv1(x, division)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x


class DNAfusionGNN(th.nn.Module):
    def __init__(self, args):
        super(DNAfusionGNN, self).__init__()
        self.conv = DNAConv(args.hidden_features+args.output_features, args.heads, args.groups, cached=args.cached)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x


class CFGCN(th.nn.Module):
    def __init__(self, args):
        super(CFGCN, self).__init__()
        self.tdGroupGCN = TDgroupGCN(args)
        self.tdRumorGCN = TDrumorGCN(args)
        self.buGroupGCN = BUgroupGCN(args)
        self.buRumorGCN = BUrumorGCN(args)
        self.dnaGNN = DNAfusionGNN(args)
        self.fc = th.nn.Linear(2*(args.hidden_features + args.output_features), args.num_class)

    def forward(self, data):
        tdGroup_x = self.tdGroupGCN(data)
        buGroup_x = self.buGroupGCN(data)
        td_x = self.tdRumorGCN(data)
        bu_x = self.buGroupGCN(data)
        td_x = self.dnaGNN(torch.stack((buGroup_x, td_x), 1), data.edge_index)
        bu_x = self.dnaGNN(torch.stack((tdGroup_x, bu_x), 1), data.BU_edge_index)
        x = th.cat((bu_x, td_x), 1)
        x = scatter(x, data.batch, dim=0, reduce='mean')
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x