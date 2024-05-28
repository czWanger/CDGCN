import sys,os
sys.path.append(os.getcwd())
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import copy


class BUrumorGCN(th.nn.Module):
    def __init__(self,args):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(args.input_features, args.hidden_features)
        self.conv2 = GCNConv(args.input_features + args.hidden_features, args.output_features)
        self.device = args.device

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x= scatter_mean(x, data.batch, dim=0)
        return x

class BUGCN(th.nn.Module):
    def __init__(self, args):
        super(BUGCN, self).__init__()
        self.BUrumorGCN = BUrumorGCN(args)
        self.fc = th.nn.Linear(args.hidden_features + args.output_features, args.num_class)

    def forward(self, data):
        x = self.BUrumorGCN(data)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


