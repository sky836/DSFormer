import numpy as np
import torch
from torch import nn


class FullAttention(nn.Module):
    def __init__(self, n_heads, d_feature, d_keys=None, d_values=None):
        super(FullAttention, self).__init__()
        self.n_heads = n_heads
        self.d_keys = d_keys

        self.query_projection = nn.Linear(d_feature, d_keys * n_heads)
        self.key_projection = nn.Linear(d_feature, d_keys * n_heads)
        self.value_projection = nn.Linear(d_feature, d_values * n_heads)

    def forward(self, Q, K, V, attn_mask=None):
        if attn_mask is not None:
            device = Q.device
            attn_mask = attn_mask.unsqueeze(2).repeat(1, 1, self.n_heads, 1, 1).to(device)
        batch_size, n_nodes, q_len, _ = Q.shape
        _, _, k_len, _ = K.shape

        # [batch_size, n_nodes, q_len, d_keys*n_heads] => [batch_size, n_nodes, n_heads, q_len, d_keys]
        queries = self.query_projection(Q).view(batch_size, n_nodes, q_len, self.n_heads, -1).transpose(2, 3)
        keys = self.key_projection(K).view(batch_size, n_nodes, k_len, self.n_heads, -1).transpose(2, 3)
        values = self.value_projection(V).view(batch_size, n_nodes, k_len, self.n_heads, -1).transpose(2, 3)

        # [batch_size, n, n_heads, q_len, k_len]
        attn_score = torch.matmul(queries, keys.transpose(3, 4)) / np.sqrt(self.d_keys)

        if attn_mask is not None:
            attn_score = attn_score.masked_fill(attn_mask, -1e9)
        # softmax
        # attn_score = torch.softmax(attn_score, dim=-1)

        return attn_score, values


class TimeAttentionLayer(nn.Module):
    def __init__(self, n_heads, time_d_model, target_d_model, dropout=0.1):
        super(TimeAttentionLayer, self).__init__()
        self.time_attn = FullAttention(n_heads=n_heads, d_feature=time_d_model, d_keys=time_d_model // n_heads, d_values=time_d_model // n_heads)
        self.target_attn = FullAttention(n_heads=n_heads, d_feature=target_d_model, d_keys=target_d_model // n_heads, d_values=target_d_model // n_heads)

        self.time_out_projection = nn.Linear(time_d_model, time_d_model)
        self.target_out_projection = nn.Linear(target_d_model, target_d_model)

        self.n_heads = n_heads
        self.merge_projection = nn.Linear(2, 1)

        self.Timedropout = nn.Dropout(dropout)
        self.Targetdropout = nn.Dropout(dropout)
        self.Timelayer_norm = nn.LayerNorm(time_d_model)
        self.Targetlayer_norm = nn.LayerNorm(target_d_model)

    def forward(self, time_features_Q, target_features_Q, time_features_K, target_features_K, attn_mask=None):
        """
        time_features:[batch_size, n_nodes, n_patches, d_model]
        target_features:[batch_size, n_nodes, n_patches, d_model]
        return:
        time_attn_features:[batch_size, n_nodes, n_patches, d_model]
        target_attn_features:[batch_size, n_nodes, n_patches, d_model]
        """
        batch_size, n_nodes, q_len, _ = time_features_Q.shape
        _, _, k_len, _ = time_features_K.shape
        # if attn_mask is not None:
        #     device = time_features_Q.device
        #     attn_mask = attn_mask.unsqueeze(2).repeat(1, 1, self.n_heads, 1, 1).to(device)
        # ===================================1 time process===========================================
        time_attn_score, time_values = self.time_attn(time_features_Q, time_features_K, time_features_K, attn_mask)
        time_attn_value = torch.matmul(time_attn_score, time_values).transpose(2, 3).reshape(batch_size, n_nodes, q_len, -1)
        time_attn_value = self.time_out_projection(time_attn_value)

        time_attn_value = self.Timedropout(time_attn_value)
        time_attn_value =self.Timelayer_norm(time_attn_value + time_features_Q)
        # ==================================2 target process============================================
        target_attn_score, target_values = self.target_attn(target_features_Q, target_features_K, target_features_K, attn_mask)
        # ==================================3 merge process============================================
        # 将两个attn score加权求和
        merge_attn_score = time_attn_score + target_attn_score
        # time_attn_score = time_attn_score.unsqueeze(3)
        # target_attn_score = target_attn_score.unsqueeze(3)
        # # merge_attn_score: [batch_size, n_nodes, n_heads, 2, q_len, k_len]
        # merge_attn_score = torch.cat((time_attn_score, target_attn_score), dim=3).permute(0, 1, 2, 4, 5, 3)
        # # merge_attn_score: [batch_size, n_nodes, n_heads, q_len, k_len]
        # merge_attn_score = self.merge_projection(merge_attn_score).squeeze(-1)

        # merge_attn_score还要进行mask
        if attn_mask is not None:
            device = merge_attn_score.device
            attn_mask = attn_mask.unsqueeze(2).repeat(1, 1, self.n_heads, 1, 1).to(device)
            merge_attn_score = merge_attn_score.masked_fill(attn_mask, -1e9)

        # 加权后再进行一次softmax
        merge_attn_score = torch.softmax(merge_attn_score, dim=-1)

        # merge_attn_score:[batch_size, n_nodes, n_heads, q_len, k_len]
        # target_values:[batch_size, n_nodes, n_heads, k_len, d_values]
        # =>[batch_size, n_nodes, q_len, d_v*n_heads]
        merge_attn_value = torch.matmul(merge_attn_score, target_values).transpose(2, 3).reshape(batch_size, n_nodes,
                                                                                                 q_len, -1)
        merge_attn_value = self.target_out_projection(merge_attn_value)

        merge_attn_value = self.Targetdropout(merge_attn_value)
        merge_attn_value = self.Targetlayer_norm(merge_attn_value + target_features_Q)

        return time_attn_score, target_attn_score, merge_attn_score, merge_attn_value, time_attn_value


class DecoderTimeAttention(nn.Module):
    def __init__(self, n_heads, decoder_time_d_model, decoder_target_d_model, dropout=0.1):
        super(DecoderTimeAttention, self).__init__()
        self.time_attn = FullAttention(n_heads=n_heads, d_feature=decoder_time_d_model, d_keys=decoder_time_d_model // n_heads, d_values=decoder_time_d_model // n_heads)
        self.outTime_projection = nn.Linear(decoder_time_d_model, decoder_time_d_model)
        self.outTarget_projection = nn.Linear(decoder_target_d_model, decoder_target_d_model)

        self.n_heads = n_heads

        self.Timedropout = nn.Dropout(dropout)
        self.Targetdropout = nn.Dropout(dropout)
        self.Timelayer_norm = nn.LayerNorm(decoder_time_d_model)
        self.Targetlayer_norm = nn.LayerNorm(decoder_target_d_model)

    def forward(self, time_features_Q, time_features_K, target_features_K=None):
        """
        time_features:[batch_size, n_nodes, n_patches, d_model]
        target_features:[batch_size, n_nodes, n_patches, d_model]
        return:
        time_attn_features:[batch_size, n_nodes, n_patches, d_model]
        target_attn_features:[batch_size, n_nodes, n_patches, d_model]
        """
        batch_size, n_nodes, q_len, _ = time_features_Q.shape
        _, _, k_len, _ = time_features_K.shape

        attn_score, values = self.time_attn(time_features_Q, time_features_K, time_features_K)
        time_value = torch.matmul(attn_score, values).transpose(2, 3).reshape(batch_size, n_nodes, q_len, -1)
        time_value = self.outTime_projection(time_value)

        time_value = self.Timedropout(time_value)
        time_value = self.Timelayer_norm(time_value + time_features_Q)
        if target_features_K is not None:
            target_features_K = target_features_K.view(batch_size, n_nodes, k_len, self.n_heads, -1).transpose(2, 3)
            cross_target = torch.matmul(attn_score, target_features_K).transpose(2, 3).reshape(batch_size, n_nodes,
                                                                                               q_len, -1)
            cross_target = self.outTarget_projection(cross_target)

            cross_target = self.Targetdropout(cross_target)
            cross_target = self.Targetlayer_norm(cross_target)
            return attn_score, time_value, cross_target
        return attn_score, time_value
