import torch
import torch.nn as nn

class STD_Module(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(1 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_embedding_u, time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        decouple_feat = torch.cat([time_in_day_feat, day_in_week_feat, node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1)], dim=-1)
        hidden = self.fully_connected_layer_1(decouple_feat)
        hidden = self.activation(hidden)
        decouple_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))[:, -history_data.shape[1]:, :, :]
        history_data = history_data * decouple_gate
        return history_data
