import torch
import torch.nn as nn
import torch.nn.functional as F

class spacegraph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=1, static_feat=None):
        super(spacegraph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.device = torch.device("cuda:0")

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        adj = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        return adj


class timegraph_constructor(nn.Module):
    def __init__(self,num_patterns):
        super(timegraph_constructor, self).__init__()
        self.num_patterns = num_patterns
        self.day_type = {}
        self.week_type = {}
        self.beta = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, time_in_day_feat,day_in_week_feat,graph_type):
        adj = {}
        batch_size,seq_length, num_nodes, node_dim = time_in_day_feat.shape
        for i in range(self.num_patterns):
            self.day_type[i] = []
            self.week_type[i] = []
            adj[i] = []
        # print('day_type:',day_type)
        for i in range(self.num_patterns):
            for j in graph_type[i]:
                self.day_type[i].append(time_in_day_feat[:, :, j, :])
                self.week_type[i].append(day_in_week_feat[:, :, j, :])
        for i in range(self.num_patterns):
            if len(self.day_type[i])==0 or len(self.week_type[i])==0:
                adj[i] = torch.zeros(batch_size, 1, 1).to('cuda:0')
            else:
                self.day_type[i] = torch.cat(self.day_type[i],dim=2)
                self.day_type[i] = self.day_type[i].reshape(batch_size,seq_length,-1,node_dim)
                self.week_type[i] = torch.cat(self.week_type[i],dim=2)
                self.week_type[i] = self.week_type[i].reshape(batch_size,seq_length,-1,node_dim)
                adj[i] =  torch.matmul(self.day_type[i],self.week_type[i].transpose(2, 3))
                adj[i] = self.beta*F.relu(torch.tanh(adj[i].mean(dim=1))).to("cuda:0")
        return adj


class DSTGNN_Module(nn.Module):
    def __init__(self, nnodes, **model_args):
        super().__init__()
        self.device = torch.device("cuda:0")  # 设置设备
        self.nnodes = nnodes
        self.dim = model_args['node_hidden']
        self.k = model_args['TOP-K']
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.dropout = model_args['dropout']
        self.num_nodes = torch.arange(self.nnodes).to(self.device)
        self.num_patterns = model_args['num_patterns']
        self.sgc = spacegraph_constructor(self.nnodes, model_args['TOP-K_s'], self.dim, alpha=self.alpha, static_feat=None)
        self.tgc = timegraph_constructor(self.num_patterns)
        self.sg = {}
        self.tg = {}
    
    def forward(self,time_in_day_feat,day_in_week_feat,graph_type):
        batch_size,_,_,_ = time_in_day_feat.shape
        stg = {}
        for i in range(self.num_patterns):
            self.sg[i] = []
            stg[i] = []
        self.tg = self.tgc(time_in_day_feat,day_in_week_feat,graph_type)
        for i in range(self.num_patterns):
            if len(graph_type[i])>0:
                self.sg[i] = self.alpha*F.relu(torch.tanh(self.sgc(torch.arange(len(graph_type[i])).to('cuda:0'))))
            else:
                self.sg[i] = torch.zeros(1,1)
            self.sg[i] = self.sg[i].unsqueeze(0).expand(batch_size, -1, -1).to('cuda:0')
            stg[i] =  self.gamma*F.relu(torch.tanh(self.sg[i]*self.tg[i].transpose(1,2)).to(self.device))
            if stg[i].size(1)>=self.k:
                mask = torch.zeros(stg[i].size(0),stg[i].size(1), stg[i].size(2)).to(self.device)
                mask.fill_(float('0'))
                s1,t1 = (stg[i] + torch.rand_like(stg[i])*0.01).topk(self.k,1)
                mask_clone = mask.clone()  # 使用clone()方法克隆mask张量
                mask_clone.scatter_(1, t1, s1.fill_(1))
                stg[i] = stg[i]*mask_clone
        return stg