import sys,os
import torch as th
from torch_scatter import scatter, scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import DNAConv
import torch


class DGN(th.nn.Module):
    def __init__(self, args):
        super(DGN, self).__init__()
        self.conv = DNAConv(args.input_features, args.heads, args.groups, cached=args.cached, dropout=args.dropout)
        self.fc_1 = th.nn.Linear(args.input_features, args.hidden_features)
        self.fc_2 = th.nn.Linear(args.hidden_features, args.num_class)
        self.device = args.device

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x.shape:(n, 2, 5000)

        x1 = x[:,0,:]
        rootindex = data.rootindex
        root = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root[index] = x1[rootindex[num_batch]]

        x = self.conv(torch.stack((root, x[:,0,:]), 1), edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = scatter_mean(x, data.batch, dim=0)
        x = self.fc_1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc_2(x)
        x = F.log_softmax(x, dim=1)
        return x


