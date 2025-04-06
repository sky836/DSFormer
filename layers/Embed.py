import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size             # the L
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
                                        in_channel,
                                        embed_dim,
                                        kernel_size=(self.len_patch, 1),
                                        stride=(self.len_patch, 1))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        """
        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: patchified time series with shape [B, N, d, P]
        """

        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1) # B, N, C, L, 1
        # B*N,  C, L, 1
        long_term_history = long_term_history.reshape(batch_size*num_nodes, num_feat, len_time_series, 1)
        # B*N,  d, L/P, 1
        output = self.input_embedding(long_term_history)
        # norm
        output = self.norm_layer(output)
        # reshape
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)    # B, N, d, P
        assert output.shape[-1] == len_time_series / self.len_patch
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)

    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (tensor or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        input_data = input_data.view(batch_size*num_nodes, num_patches, num_feat)
        # positional encoding
        if index is None:
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
        else:
            index = index.view(batch_size*num_nodes, num_patches)
            pe = self.position_embedding[index]
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        # reshape
        input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        return input_data