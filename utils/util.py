import torch
import numpy as np
import torch
import pickle
import os
import torch.nn as nn
import math
import shutil
import torch.nn.functional as F
import time

def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

class StandardScaler():
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

#!/usr/bin/env python

def re_normalization(x, mean, std):
    r"""
    Standard re-normalization

    mean: float
        Mean of data
    std: float
        Standard of data
    """
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    r"""
    Max-min normalization

    _max: float
        Max
    _min: float
        Min
    """
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    r"""
    Max-min re-normalization

    _max: float
        Max
    _min: float
        Min
    """
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

class StandardScaler():
    r"""
    Description:
    -----------
    Standard the input.

    Args:
    -----------
    mean: float
        Mean of data.
    std: float
        Standard of data.

    Attributes:
    -----------
    Same as Args.
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    r"""
    Description:
    -----------
    Load pickle data.
    
    Parameters:
    -----------
    pickle_file: str
        File path.

    Returns:
    -----------
    pickle_data: any
        Pickle data.
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')
    
def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100



def masked_huber(preds, labels, null_val=np.nan):
    crit = torch.nn.SmoothL1Loss()
    # crit = torch.nn.MSELoss()
    return crit(preds, labels)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse

class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask   = model_args['adjs']
    
    def _mask(self, index, adj):
        mask = self.mask[index] + torch.ones_like(self.mask[index]) * 1e-7
        return mask.to(adj.device) * adj

    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result
    
class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        degree  = torch.sum(graph, dim=2)
        degree  = remove_nan_inf(1 / degree)
        degree  = torch.diag_embed(degree)
        normed_graph = torch.bmm(degree, graph)
        return normed_graph

    def forward(self, adj):
        return [self._norm(_) for _ in adj]

class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order  = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph               # 1 order
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order+1):     # e.g., order = 3, k=[2, 3]; order = 2, k=[2]
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]
    

class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.hidden_dim = model_args['num_hidden']
        self.node_dim   = model_args['node_hidden']
        self.time_slot_emb_dim  = self.hidden_dim
        self.input_seq_len      = 12
        # Time Series Feature Extraction
        self.dropout    = nn.Dropout(0.1)
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim= self.hidden_dim
        # Time Slot Embedding Extraction
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)
        # Distance Score
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim']*2
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim*2)

    def reset_parameters(self):
        for q_vec in self.q_vecs:
            nn.init.xavier_normal_(q_vec.data)
        for bias in self.biases:
            nn.init.zeros_(bias.data)

    def forward(self, X, E_d, E_u, T_D, D_W):
        # last pooling
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        # dynamic information
        X = X[:, :, :, 0].transpose(1, 2).contiguous()       # X->[batch_size, seq_len, num_nodes]->[batch_size, num_nodes, seq_len]
        [batch_size, num_nodes, seq_len] = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(self.dropout(self.bn(F.relu(self.fc_ts_emb1(X)))))     # [batchsize, num_nodes, hidden_dim]
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        # node embedding
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)
        # distance calculation
        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)                    # hidden state for calculating distance
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)                    # hidden state for calculating distance
#         print('x1:',X1.shape)
#         print('x2:',X2.shape)
        X  = [X1,X2]
        adjacent_list = []
        for _ in X:
            Q = self.WQ(_)
            K = self.WK(_)
            QKT = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            W = torch.softmax(QKT, dim=-1)
            adjacent_list.append(W)
        print('adjacent_list[0].shape:',adjacent_list[0].shape)
        return adjacent_list

class TrainLogger():
    r"""
    Description:
    -----------
    Logger class. Function:
    - print all training hyperparameter setting
    - print all model    hyperparameter setting
    - save all the python file of model

    Args:
    -----------
    path: str
        Log path
    """
    
    def __init__(self, model_name, dataset):
        path        = 'log/'
        cur_time    = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cur_time    = cur_time.replace(" ", "-")
        # mkdir
        os.makedirs(path + cur_time)
        # pwd = os.getcwd() + "/"
        # copy model files
        shutil.copytree('models',  path + cur_time + "/models")      # copy models
        shutil.copytree('configs',  path + cur_time + "/configs")      # copy models
        # shutil.copyfile('main.py',  path + cur_time + "/main.ipynb")      # copy models
        # backup model parameters
        try:
            shutil.copyfile('output/' + model_name + "_" + dataset + ".pt", path + cur_time + "/"  + model_name + "_" + dataset + ".pt")
            shutil.copyfile('output/' + model_name + "_" + dataset + "_resume" + ".pt", path + cur_time + "/" + model_name + "_" + dataset + "_resume.pt")
        except:
            # No model_para.pt
            pass
    def __print(self, dic, note=None, ban=[]):
        print("=============== " + note + " =================")
        for key,value in dic.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))
        print("--------------------------------------------")

    def print_model_args(self, model_args, ban=[]):
        self.__print(model_args, note='model args', ban=ban)

    def print_optim_args(self, optim_args, ban=[]):
        self.__print(optim_args, note='optim args', ban=ban)