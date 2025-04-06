import numpy as np
import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F
from torch.autograd import Variable


class PatchEmbed(nn.Module):
    def __init__(self, seq_len=12*24, patch_size=3, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patches_resolution = seq_len // patch_size
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size, 1), stride=(patch_size, 1))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, model_dim)
        B, L, N, D = x.shape
        x = x.permute(0, 2, 3, 1).unsqueeze(-1)
        x = x.reshape(B*N, D, L, 1)

        x = self.proj(x)
        x = x.squeeze(-1)
        x = x.view(B, N, self.embed_dim, -1)
        x = x.transpose(-1, -2)
        if self.norm is not None:
            x = self.norm(x)
        return x.transpose(1, 2)


class CrossAttentionLayer(nn.Module):
    def __init__(
        self, time_dim, target_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.crossAtten = AttentionLayer(time_dim, num_heads, target_dim=target_dim)
        self.fc = FC(target_dim, feed_forward_dim, dropout)

        self.ln = nn.LayerNorm(target_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y_time, x_time, x_target):
        y_time, x_time, x_target = y_time.transpose(1, -2), x_time.transpose(1, -2), x_target.transpose(1, -2)
        residual_y, residual_x = y_time, x_time
        y_target, cross_attn_no_softmax, _ = self.crossAtten(y_time, x_time, x_target)
        y_target = self.dropout(y_target)
        y_target = self.ln(y_target)

        y_target = self.fc(y_target)
        y_target = y_target.transpose(1, -2)
        return y_target


class Decoder_layer(nn.Module):
    def __init__(self, time_dim, target_dim, supports, supports_len, num_heads, feed_forward_dim, dec_layers, dropout):
        super().__init__()
        self.time_dim = time_dim
        self.target_dim = target_dim
        self.dec_layers = dec_layers

        self.self_attn_time = nn.ModuleList(
            [
                SelfAttentionLayer(time_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(dec_layers)
            ]
        )

        self.cross_attn_time = nn.ModuleList(
            [
                SelfAttentionLayer(time_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(dec_layers)
            ]
        )

        self.cross_target = CrossAttentionLayer(time_dim, target_dim, feed_forward_dim, num_heads, dropout)

        # GCN special
        # self.supports = supports
        # self.supports_len = supports_len
        # self.gconvs = nn.ModuleList(
        #     [
        #         gcn(self.target_dim, self.target_dim, dropout, support_len=self.supports_len)
        #         for _ in range(dec_layers)
        #     ]
        # )
        # self.self_attn_layers_s = nn.ModuleList(
        #     [
        #         SelfAttentionLayer(target_dim, feed_forward_dim, num_heads, dropout)
        #         for _ in range(dec_layers)
        #     ]
        # )

    def forward(self, y_time, x_time, x_target):
        # y_time: (batch_size, in_steps, num_nodes, d_model)
        for i in range(self.dec_layers):
            y_time = self.self_attn_time[i](y_time, y_time, y_time, dim=1)
            # y_time = self.cross_attn_time[i](y_time, x_time, x_time, dim=1)

        y_target = self.cross_target(y_time, x_time, x_target)

        # for i in range(self.dec_layers):
        #     y_target = self.self_attn_layers_s[i](y_target, y_target, y_target, dim=2)
        #     y_target = self.gconvs[i](y_target, self.supports)

        return y_target


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        # self.mlp = linear(c_in,c_out)
        self.mlp = nn.Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        # x:[batch_size, seq_len, n_nodes, d_model]
        # 在seq_len维度进行gcn，而不是在d_model维度：[n_nodes, seq_len]*[n_nodes, n_nodes] => [n_nodes, seq_len]
        # x = x.transpose(1, 3)
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        # h = h.transpose(1, 3)
        return h


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden


class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False, target_dim=None):
        super().__init__()

        self.model_dim = model_dim
        self.target_dim = target_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads
        self.value_dim = self.head_dim

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)

        if target_dim is None:
            self.FC_V = nn.Linear(model_dim, model_dim)

            self.out_proj = nn.Linear(model_dim, model_dim)
        else:
            self.FC_V = nn.Linear(target_dim, target_dim)

            self.out_proj = nn.Linear(target_dim, target_dim)
            self.value_dim = target_dim // num_heads

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.value_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        attn_score_no_softmax = attn_score

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        if self.target_dim is not None:
            attn_score = attn_score.unsqueeze(1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out, attn_score, value


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, Q, K, V, dim=-2):
        Q = Q.transpose(dim, -2)
        K = K.transpose(dim, -2)
        V = V.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = Q
        out, _, _ = self.attn(Q, K, V)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class FC(nn.Module):
    def __init__(self, model_dim, feed_forward_dim, dropout):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.ln(x + residual)
        return x


class STNkd(nn.Module):
    def __init__(self, d=64, init_d=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(d, init_d, 1)
        self.conv2 = torch.nn.Conv1d(init_d, init_d*2, 1)
        # self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(init_d*2, init_d)
        self.fc2 = nn.Linear(init_d, init_d//2)
        self.fc3 = nn.Linear(init_d//2, 1)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(init_d)
        self.bn2 = nn.BatchNorm1d(init_d*2)
        # self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(init_d)
        self.bn5 = nn.BatchNorm1d(init_d//2)

        self.init_d = init_d

    def forward(self, x):
        # x:[B*N / B, D, T]
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.init_d*2)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # x = x + 0.5
        return x


class MergeAttentionLayer(nn.Module):
    def __init__(
        self, time_dim, target_dim, init_d, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn_time = AttentionLayer(time_dim, num_heads, mask)
        self.attn_target = AttentionLayer(target_dim, num_heads, mask)
        self.target_proj = nn.Linear(target_dim, target_dim)

        self.fc_time = FC(time_dim, feed_forward_dim, dropout)
        self.fc_target = FC(target_dim, feed_forward_dim, dropout)

        self.self_targetFC = FC(target_dim, feed_forward_dim, dropout)
        self.dropout_self = nn.Dropout(dropout)
        self.ln_self = nn.LayerNorm(target_dim)

        self.ln1 = nn.LayerNorm(time_dim)
        self.ln2 = nn.LayerNorm(target_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.W_t = STNkd(time_dim, init_d)
        self.W_s = STNkd(target_dim, init_d)
        self.h = num_heads

    def forward(self, time_features, target_features, dim=-2):
        batch_size = time_features.shape[0]  # [B, T, D]
        B, T, N, D = target_features.shape
        time_features, target_features = time_features.transpose(dim, -2), target_features.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual_time, residual_target = time_features, target_features
        time_features, attn_time_no_softmax, _ = self.attn_time(time_features, time_features, time_features)
        _, attn_target_no_softmax, value = self.attn_target(target_features, target_features, target_features)

        w_t = self.W_t(residual_time.transpose(1, 2)).unsqueeze(1).repeat(1, N, 1)  # [B, N, 1]
        w_s = self.W_s(residual_target.reshape(B*N, T, D).transpose(1, 2)).reshape(B, N, 1)  # [B, N, 1]
        w_m = torch.cat((w_t, w_s), dim=2)
        w_m = F.softmax(w_m, dim=2)
        w_t, w_s = torch.split(w_m, 1, dim=2)
        w_t, w_s = w_t.unsqueeze(-1).repeat(self.h, 1, T, T), w_s.unsqueeze(-1).repeat(self.h, 1, T, T)
        merge_attn = w_t * attn_time_no_softmax.unsqueeze(1) + w_s * attn_target_no_softmax
        
        target_features = merge_attn @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        target_features = torch.cat(
            torch.split(target_features, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        target_features = self.target_proj(target_features)

        time_features = self.dropout1(time_features)
        time_features = self.ln1(time_features + residual_time)
        target_features = self.dropout2(target_features)
        target_features = self.ln2(target_features + residual_target)

        time_features = self.fc_time(time_features)
        target_features = self.fc_target(target_features)

        time_features = time_features.transpose(dim, -2)
        target_features = target_features.transpose(dim, -2)
        return time_features, target_features


class STGCN(nn.Module):
    def __init__(self, time_dim, target_dim):
        super(STGCN, self).__init__()
        self.cheb_k = 2
        self.time_dim = time_dim
        self.target_dim = target_dim
        self.weights_pool = nn.init.xavier_normal_(
            nn.Parameter(
                torch.FloatTensor(self.time_dim, self.cheb_k * self.target_dim, self.target_dim)
            )
        )
        self.bias_pool = nn.init.xavier_normal_(
            nn.Parameter(torch.FloatTensor(self.time_dim, self.target_dim))
        )

    def forward(self, t, s, support):
        s_g = []

        if support.dim() == 2:
            graph_list = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                s_g.append(torch.einsum("nm,bmc->bnc", graph, s))
        elif support.dim() == 3:
            graph_list = [
                torch.eye(support.shape[1])
                .repeat(support.shape[0], 1, 1)
                .to(support.device),
                support,
            ]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                s_g.append(torch.einsum("bnm,bmc->bnc", graph, s))
        s_g = torch.cat(s_g, dim=-1)

        weights = torch.einsum(
            "bd,dio->bio", t, self.weights_pool
        )  # B, cheb_k*in_dim, out_dim
        bias = torch.matmul(t, self.bias_pool)  # B, T, out_dim
        s_gconv = (
                torch.einsum("bni,bio->bno", s_g, weights) + bias.unsqueeze(1)
        )  # B, T, N, out_dim
        return s_gconv


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (int): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x, dim=-2):
        """
        x: (batch_size, in_steps, num_nodes, model_dim)
        """
        x = x.transpose(dim, -2)

        x0 = x[..., 0::2, :]  # B N L/2 C
        x1 = x[..., 1::2, :]  # B N L/2 C
        x = torch.cat([x0, x1], -1)  # B N L/2 2*C

        x = self.reduction(x)
        x = self.norm(x)

        return x.transpose(dim, -2)


class Model(nn.Module):
    def __init__(self, configs, supports=None):
        super().__init__()
        self.num_nodes = configs.num_nodes
        self.in_steps = configs.seq_len
        self.out_steps = configs.pred_len
        self.steps_per_day = configs.steps_per_day
        self.input_dim = configs.input_dim
        self.output_dim = configs.output_dim
        self.input_embedding_dim = configs.input_embedding_dim
        self.tod_embedding_dim = configs.tod_embedding_dim
        self.dow_embedding_dim = configs.dow_embedding_dim
        self.spatial_embedding_dim = configs.spatial_embedding_dim
        self.feed_forward_dim = configs.feed_forward_dim
        self.pos_tdim = configs.pos_dim
        self.pos_sdim = configs.pos_dim
        self.time_dim = (configs.tod_embedding_dim + configs.dow_embedding_dim + configs.pos_dim)
        self.target_dim = (configs.input_embedding_dim + configs.spatial_embedding_dim + configs.pos_dim)
        self.num_heads = configs.n_heads
        self.num_layers = configs.num_layers
        self.use_mixed_proj = configs.use_mixed_proj
        self.patch_size = configs.patch_size
        self.num_nodes = configs.num_nodes
        self.supports = supports
        self.dropout = configs.dropout
        self.init_d = configs.init_d
        if self.supports is None:
            self.supports = []
        self.supports_len = 0

        self.patch_emb = PatchEmbed(seq_len=self.in_steps, patch_size=self.patch_size,
                                    in_chans=self.input_dim, embed_dim=self.input_embedding_dim, norm_layer=nn.LayerNorm)
        self.num_patches = self.patch_emb.num_patches
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        self.pos_time = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.num_patches, self.pos_tdim))
        )
        self.pos_series = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.num_patches, self.pos_sdim))
        )

        self.output_proj = nn.Linear(
            self.target_dim * self.num_patches, self.out_steps * self.output_dim
        )
        # self.output_proj1 = nn.Linear(self.target_dim * self.num_patches, self.target_dim)

        # ===================================encoding special=============================================
        self.merge_attn_layers = nn.ModuleList(
            [
                MergeAttentionLayer(self.time_dim, self.target_dim, self.init_d, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

        self.time_fc = nn.Linear(self.time_dim * self.num_patches, self.out_steps * (self.input_dim - self.output_dim))

    def encoding(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size, in_steps, num_nodes, _ = x.shape

        if self.tod_embedding_dim > 0:
            tod = x[..., -2]
            tod = tod[..., 0]
        if self.dow_embedding_dim > 0:
            dow = x[..., -1]
            dow = dow[..., 0]
        x = x[..., : self.input_dim]

        x = self.patch_emb(x)
        patch_size = self.patch_emb.patch_size
        target_features = [x]
        time_features = []
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long()) # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            time_features.append(tod_emb[:, ::patch_size])
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(dow.long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            time_features.append(dow_emb[:, ::patch_size])
        time_features.append(self.pos_time.expand(size=(batch_size, *self.pos_time.shape)))
        node_emb = self.node_emb.expand(size=(batch_size, self.num_patches, *self.node_emb.shape))
        target_features.append(node_emb)
        target_features = torch.cat(target_features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        time_features = torch.cat(time_features, dim=-1)  # (batch_size, in_steps, model_dim)
        pos_series = self.pos_series.expand(size=(batch_size, num_nodes, *self.pos_series.shape))
        target_features = target_features.transpose(1, 2)
        target_features = torch.cat([target_features, pos_series], dim=-1)
        target_features = target_features.transpose(1, 2)

        for i in range(self.num_layers):
            time_features, target_features = self.merge_attn_layers[i](time_features, target_features, dim=1)

        return time_features, target_features

    def forward(self, x, y):
        batch_size, _, num_nodes, _ = x.shape
        time_features, target_features = self.encoding(x)

        target_features = target_features.transpose(1, 2).reshape(batch_size, num_nodes, -1)

        out = target_features
        out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)

        time = time_features.reshape(batch_size, -1)
        time = self.time_fc(time).view(batch_size, self.out_steps, self.input_dim - self.output_dim)

        return out, time

