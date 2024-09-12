import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = torch.einsum('ncwl,nvw->ncvl',(x,G))
        return x 
    
class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)
    
class residualconv(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha,num_patterns):
        super(residualconv, self).__init__()
        self.nconv = HGNN_conv(c_in*num_patterns,c_out*num_patterns)
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.a_list = []
    def forward(self,x,adj):
        matrix = torch.eye(adj.size(1)).to(x.device)
        matrix = matrix.unsqueeze(0)
        adj_new = adj + matrix
        d = adj_new.sum(1)
        d = d.view(adj_new.size(0), 1, adj_new.size(2))
        a = adj_new / d
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho

class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell   = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim]  = X.shape
        X   = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        hx  = torch.zeros_like(X[:, 0, :])
        output  = []
        for _ in range(X.shape[1]):
            hx  = self.gru_cell(X[:, _, :], hx)
            output.append(hx)
        output  = torch.stack(output, dim=0)
        output  = self.dropout(output)
        return output
    
    
class RNN_Module(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat   = hidden_dim
        self.hidden_dim = hidden_dim
        self.rnn_layer          = RNNLayer(hidden_dim, model_args['dropout'])

    def forward(self, hidden_signal):
        [batch_size, seq_len, num_nodes, num_feat]  = hidden_signal.shape
        hidden_states_rnn   = self.rnn_layer(hidden_signal)
        hidden_states   = hidden_states_rnn.reshape(seq_len, batch_size, num_nodes, num_feat)
        hidden_states   = hidden_states.transpose(0, 1)
        return hidden_states

class LearnableMatrixModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LearnableMatrixModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_size, output_size))  # 定义可学习参数

    def forward(self, x):
        return torch.mul(x, self.weight)


class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        # self.k = model_args['k']
        self.output_seq_len = 12
        self.forecast_fc    = nn.Linear(hidden_dim, forecast_hidden_dim)
        self.model_args     = model_args

    def forward(self, gated_history_data, hidden_states_dif):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))
        for _ in range(int(self.output_seq_len / self.model_args['gap'])-1):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2  = history[:, -sub:, :, :]
                _1  = torch.cat([_2] + _1, dim=1)
            else:
                _1  = torch.cat(_1, dim=1)
        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)
        return predict
    

class SIE_Module(nn.Module):
    def __init__(self,num_patterns,num_layers,gconv1,gconv2,start_conv):
        super(SIE_Module, self).__init__()
        self.num_patterns = num_patterns
        self.num_layers = num_layers
        self.gconv1 = gconv1
        self.gconv2 = gconv2
        self.start_conv = start_conv
    def forward(self,history_data, node_type,graph_type,stg):
        batch_size,seq_length,num_nodes,num_feature = history_data.shape
        decouple_flow = {}
        H = {}
        non_empty_tensors = []
        out = []
        for i in range(self.num_patterns):
            H[i] = []
            decouple_flow[i] = []
        for i in range(self.num_patterns):
            if len(node_type[i])>0:
                decouple_flow[i] = F.relu(self.start_conv(node_type[i])).to('cuda:0')
        for i in range(self.num_layers):
            for p in range(self.num_patterns):
                if len(decouple_flow[p])>0:
                    H[p] = self.gconv1[i](decouple_flow[p],stg[p])+self.gconv2[i](decouple_flow[p],stg[p].transpose(1,2))
                    non_empty_tensors.append(H[p])
            out.append(torch.cat(non_empty_tensors,dim=2))
            non_empty_tensors = []
        com_flow = torch.cat(out, axis=-1)
        x_clone = torch.empty(batch_size, 32, num_nodes, num_feature*self.num_patterns*self.num_layers).to('cuda:0')
        count = 0
        # print('x_clone.shape:',x_clone.shape)
        # print('com_flow.shape:',com_flow.shape)
        for key in sorted(graph_type.keys()):
            for node in graph_type[key]:
                x_clone[:, :, node, :] = com_flow[:, :, count, :]
                count += 1
        return x_clone
