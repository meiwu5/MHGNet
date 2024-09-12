import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DSTGG_Module import DSTGNN_Module
from models.SIE_Module import SIE_Module,residualconv,RNN_Module,Forecast,LearnableMatrixModel
from models.STD_Module import STD_Module
import logging
import os


class MHDNet(nn.Module):
    def __init__(self,  static_feat=None, conv_channels=32, residual_channels=32, tanhalpha=3,skip_channels=64, end_channels=128, propalpha=0.05, **model_args):
        super().__init__()
        self.embedding = nn.Linear(model_args['num_feat'], model_args['num_hidden'])
        self.in_feat = model_args['num_feat']
        self.hidden_dim = model_args['num_hidden']
        self.node_dim      = model_args['node_hidden']
        self.num_nodes     = model_args['num_nodes']
        self.device         = torch.device("cuda:0")
        self._output_hidden = 512
        self.output_hidden = 512
        self.num_layers = model_args['num_layers']
        self.num_patterns = model_args['num_patterns']
        self.dropout = model_args['dropout']
        self.seq_length = model_args['seq_length']
        self.gcn_depth = model_args['gcn_depth']
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.STD_Module = nn.ModuleList()
        self._model_args    = model_args
        self.tanhalpha = tanhalpha
        self.para = []
        self.mlp_list = []
        self.DSTGNN_Module = DSTGNN_Module(self.num_nodes, **model_args)
        self.start_conv = nn.Conv2d(in_channels=12,
                                    out_channels=conv_channels,
                                    kernel_size=(1, 1))
        for i in range(self.num_patterns-1):
            self.STD_Module.append(STD_Module(node_emb_dim=model_args['node_hidden'], time_emb_dim=model_args['time_emb_dim'], hidden_dim=64))
        for i in range(self.num_layers):
            self.gconv1.append(residualconv(conv_channels, residual_channels, self.gcn_depth, self.dropout, propalpha,self.num_patterns))
            self.gconv2.append(residualconv(conv_channels, residual_channels, self.gcn_depth, self.dropout, propalpha,self.num_patterns))
        self.SIE_Module = SIE_Module(self.num_patterns,self.num_layers,self.gconv1,self.gconv2,self.start_conv)
        self.end_conv_1 = nn.Conv2d(in_channels=32,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=12,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.RNN_Module = RNN_Module(self.num_patterns*self.hidden_dim*self.num_layers, forecast_hidden_dim=256, **model_args)
        self._num_nodes = model_args['num_nodes']
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb  = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))
        self.node_emb_u = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        self.end_conv_3 = nn.Conv2d(in_channels=12,
                                    out_channels=4,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.gated_history_data = []
        self.out_fc_in = 2*self.num_patterns*self.hidden_dim+self.num_layers*self.hidden_dim+2*model_args['time_emb_dim']
        self.out_fc_1 = nn.Linear(self.out_fc_in,out_features=self.output_hidden)
        self.out_fc_2   = nn.Linear(self.output_hidden, 3)
        self.reset_parameter()
        self.forecast = Forecast(self.seq_length, forecast_hidden_dim=256, **model_args)
        self.idx = torch.arange(self.num_nodes).to(self.device)
        self.mlp = nn.ModuleList()
        for i in range(self.num_patterns+1):
            self.mlp.append(nn.Linear(self.hidden_dim,1))
        self.matrics1 = LearnableMatrixModel(self.num_nodes,self.num_patterns*self.hidden_dim*self.num_layers).to('cuda:0')
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)
    def _prepare_inputs(self, history_data):
        num_feat    = self._model_args['num_feat']
        # node embeddings
        node_emb_u  = self.node_emb_u  # [N, d]
        node_emb_d  = self.node_emb_d  # [N, d]
        # time slot embedding
        time_in_day_feat = self.T_i_D_emb[(history_data[:, :, :, num_feat] * 288).type(torch.LongTensor)]    # [B, L, N, d]
        day_in_week_feat = self.D_i_W_emb[(history_data[:, :, :, num_feat+1]).type(torch.LongTensor)]          # [B, L, N, d]
        # traffic signals
        history_data = history_data[:, :, :, :num_feat]
        
        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat
    
    def Node_Clusterer(self,history_data,x):
        batch_size,seq_length,num_nodes,num_feature = history_data.shape
        concat_x = []
        centers = []
        h0 = []
        x_mlp = {}
        x_mean = []
        node_type = {}
        graph_type = {}
        history_data = self.mlp[0](history_data)
        history_data_mean = torch.mean(history_data, dim=1)
        for i in range(self.num_patterns):
            h0.append(x[i])
            x_mlp[i] = self.mlp[i+1](x[i])
            x_mean.append(torch.mean(x_mlp[i], dim=1))
            node_type[i] = []
            graph_type[i] = []
        distance = torch.zeros(batch_size,num_nodes,self.num_patterns)
        h0 = torch.cat(h0,dim=-1)
        concat_x = torch.cat(x_mean,dim=-1)
        x_ratio = concat_x/history_data_mean
        for i in range(x_ratio.shape[-1]):
            center_value,center_idx = torch.max(x_ratio[:, :, i], axis=1, keepdims=True)
            centers.append(center_value)
        centers_value = torch.cat(centers,dim=-1)
        centers_value = centers_value.unsqueeze(1).expand(-1,num_nodes,-1)
        for i in range(x_ratio.shape[-1]):
            distance[:,:,i] = torch.abs(x_ratio[:,:,i] - centers_value[:,:,i])
        for node in range(distance.shape[1]):
            mass = torch.argmin(distance[:,node,:],axis = 1)
            unique_values, counts = np.unique(mass, return_counts=True)
            mass_index = np.argmax(counts)
            mass_value = unique_values[mass_index]
            node_type[mass_value].append(h0[:,:,node,:].unsqueeze(dim=2))
            graph_type[mass_value].append(node)
        for p in range(self.num_patterns):
            if len(node_type[p])>0:
                node_type[p] = torch.cat(node_type[p],dim=2).to('cuda:0')
        return node_type,graph_type,h0
    
    def forward(self, history_data):
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat=self._prepare_inputs(history_data)
        # print('history_data.shape',history_data.shape)
        # batch_size,_,_,_ = history_data
        history_data = self.embedding(history_data)
        x = {}
        x[self.num_patterns-1] = history_data
        
        for i in range(self.num_patterns-1):
            x[i] = self.STD_Module[i](node_embedding_u,time_in_day_feat, day_in_week_feat, history_data)
            x[self.num_patterns-1] = x[self.num_patterns-1]-x[i]
            
        node_type,graph_type,h0 = self.Node_Clusterer(history_data,x)
        stg = self.DSTGNN_Module(time_in_day_feat,day_in_week_feat,graph_type)
        x_out = self.SIE_Module(history_data,node_type,graph_type,stg)
        x_out = self.RNN_Module(x_out)
        x_out = F.relu(self.end_conv_1(x_out))
        x_out = self.end_conv_2(x_out)
        x_out = self.matrics1(x_out)
        forecast_hidden = torch.cat([x_out,history_data,h0,time_in_day_feat,day_in_week_feat],dim=-1)
        forecast    = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast = self.end_conv_3(forecast)
        forecast    = forecast.transpose(1,2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        return forecast