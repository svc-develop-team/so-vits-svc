"""Building blocks for speech SSL models supporting pruning.

Originally from:
https://github.com/pytorch/audio/blob/main/torchaudio/models/wav2vec2/components.py

"""

import math
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Module

from .hardconcrete import HardConcrete
from .pruning_utils import (
    prune_conv1d_layer,
    prune_layer_norm,
    prune_linear_layer,
)


def _init_transformer_params(module):
    """
    Initialize the weights of Transformer module in Wav2Vec2/HuBERT.

    If the module is ``nn.Linear``, normalize the weight with mean 0 and standard deviation 0.02.
    If ``bias`` is set to ``True`` in the module, set ``bias`` to 0.

    If the module is ``nn.Embedding``, normalize the weight with mean 0 and standard deviation 0.02.
    If ``padding_idx`` is not None, set the weight of padding to 0.

    Note:
        Ths method corresponds to
        `init_bert_params
        <https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/transformer_sentence_encoder.py#L21>`__
        in the original ``fairseq`` implementation.
    """

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class LayerNorm(nn.LayerNorm):
    """Layer norm with transpose"""

    def forward(self, input: Tensor) -> Tensor:
        x = input.transpose(-2, -1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-2, -1)
        return x


class ConvLayerBlock(Module):
    """Convolution unit of FeatureExtractor"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        layer_norm: Optional[Module],
        prune_conv_channels: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

        if prune_conv_channels:
            self.hard_concrete = HardConcrete(n_in=out_channels, init_mean=0.01)
        else:
            self.hard_concrete = None

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape: ``[batch, in_channels, in_frame]``.
            length (Tensor or None, optional): Shape ``[batch, ]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
            Optional[Tensor]: Shape ``[batch, ]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = nn.functional.gelu(x)

        if self.hard_concrete is not None:
            channel_mask = self.hard_concrete()  # hard concrete mask, (out_channels,)
            x = x * channel_mask.unsqueeze(-1)

        if length is not None:
            length = torch.div(length - self.kernel_size, self.stride, rounding_mode="floor") + 1
            # When input length is 0, the resulting length can be negative. So fix it here.
            length = torch.max(torch.zeros_like(length), length)
        return x, length
    
    def get_num_params_and_out_channels(self, in_channels):
        if self.hard_concrete is not None:
            out_channels = self.hard_concrete.l0_norm()
        else:
            out_channels = self.conv.out_channels
        
        num_params = in_channels * out_channels * self.kernel_size
        if self.conv.bias is not None:
            num_params += out_channels
        if self.layer_norm is not None:
            num_params += out_channels * 2
        
        return num_params, out_channels


class FeatureExtractor(Module):
    """Extract features from audio

    Args:
        conv_layers (nn.ModuleList):
            convolution layers
    """

    def __init__(
        self,
        conv_layers: nn.ModuleList,
    ):
        super().__init__()
        self.conv_layers = conv_layers

        # NOTE: a dummy weight used to save the soft mask of the last conv layer
        self.dummy_weight = nn.Parameter(
            torch.ones(conv_layers[-1].conv.out_channels, dtype=torch.float32),
            requires_grad=False
        )

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor):
                Input Tensor representing a batch of audio,
                shape: ``[batch, time]``.
            length (Tensor or None, optional):
                Valid length of each input sample. shape: ``[batch, ]``.

        Returns:
            Tensor:
                The resulting feature, shape: ``[batch, frame, feature]``
            Optional[Tensor]:
                Valid length of each output sample. shape: ``[batch, ]``.
        """
        if x.ndim != 2:
            raise ValueError("Expected the input Tensor to be 2D (batch, time), " "but received {list(x.shape)}")

        x = x.unsqueeze(1)  # (batch, channel==1, frame)
        for layer in self.conv_layers:
            x, length = layer(x, length)  # (batch, feature, frame)
        x = x.transpose(1, 2)  # (batch, frame, feature)
        x = x * self.dummy_weight
        return x, length

    def get_num_params_and_final_out_channels(self):
        in_channels = 1
        num_params = 0
        for layer in self.conv_layers:
            layer_params, in_channels = layer.get_num_params_and_out_channels(in_channels)
            num_params += layer_params

        num_params += in_channels   # dummy weight
        
        return num_params, in_channels
    
    def prune(self):
        """"Prune conv layers and dummy weight based on hardconcrete parameters.
        This is an in-place operation.
        """
        new_config = []     # [(output_channel, kernel_size, stride), ...]
        for idx, layer in enumerate(self.conv_layers):
            if layer.hard_concrete is not None:
                assert not layer.hard_concrete.training
                mask = layer.hard_concrete()    # (out_features,)
                index = mask.nonzero().squeeze(-1)    # 2D -> 1D
                assert len(index) > 0, f"Conv channels pruned to zero at index {idx}"
                new_config.append(
                    (len(index), layer.kernel_size, layer.stride)
                )

                # prune the current layer
                prune_conv1d_layer(layer.conv, index, "output")
                if layer.layer_norm is not None:
                    prune_layer_norm(layer.layer_norm, index)

                # prune the next layer
                if idx == len(self.conv_layers) - 1:
                    self.dummy_weight.data *= mask
                    self.dummy_weight = nn.Parameter(
                        self.dummy_weight.index_select(0, index).clone().detach(), requires_grad=False
                    )
                else:
                    self.conv_layers[idx+1].conv.weight.data *= mask.unsqueeze(-1)
                    prune_conv1d_layer(self.conv_layers[idx+1].conv, index, dim="input")

                layer.hard_concrete = None
            else:
                new_config.append(
                    (layer.conv.out_channels, layer.kernel_size, layer.stride)
                )
                index = torch.arange(layer.conv.out_channels, dtype=torch.long)

        return new_config, index


class FeatureProjection(Module):
    """Layer that connects FeatureExtractor and Encoder

    Projects features to encoder dimension.

    Args:
        in_features (int): Input feature dim.
        out_features (int): Output feature dim.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(
            in_features,
            out_features,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor):
                Feature Tensor. shape: ``[batch, frame, in_feature]``
        Returns:
            Tensor: Projected features. ``[batch, frame, out_feature]``.
        """
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x
    
    def get_num_params(self, in_features):
        return in_features * 2 + (in_features + 1) * self.projection.out_features


class ConvolutionalPositionalEmbedding(Module):
    """Positional embedding which is placed at the beginning of Transformer.

    Args:
        embed_dim (int): Feature dimension of the input Tensor.
        kernel_size (int): The number of frames to be use.
        groups (int): The number of groups in feature dimensions.
    """

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int,
        groups: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )

        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    def __prepare_scriptable__(self):
        for hook in self.conv._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(self.conv)
        return self

    def forward(self, x):
        """
        Args:
            x (Tensor): shape ``[batch, frame, feature]``.

        Returns:
            Tensor: The resulting feature. Shape ``[batch, frame, feature]``.
        """
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., : -self.num_remove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)
        return x


class SelfAttention(Module):
    """Multihead Self Attention module

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probability on attn_output_weights. Default: ``0.0``
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        prune_heads: bool = False,  # whether to prune attention heads
        prune_layer: bool = False,  # whether to prune entire attention layers
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = torch.nn.Dropout(dropout)

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=True)

        if prune_heads:
            self.hard_concrete_for_heads = HardConcrete(n_in=num_heads, init_mean=0.01)
        else:
            self.hard_concrete_for_heads = None

        if prune_layer:
            self.hard_concrete_for_layer = HardConcrete(n_in=1, init_mean=0.01)
        else:
            self.hard_concrete_for_layer = None

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or ``None``, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``
            position_bias: Not used. Only for the compatibility with :py:class:`WavLMSelfAttention`.
            key_padding_mask (Tensor or ``None``): Not used. Only for the compatibility with
                :py:class:`WavLMSelfAttention`.
        Returns:
            (Tensor, ``None``): The resulting attention output and ``None`` (necessary for compatibility
                with :py:class:`WavLMSelAttention`).
                Attention output shape: ``[batch, sequence_length, embed_dim]``.
        """
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        
        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)  # B, nH, Hd, L
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        # scale down q to avoid value overflow.
        weights = (self.scaling * q) @ k  # B, nH, L, L
        if attention_mask is not None:
            weights += attention_mask
        # subtracting a constant value from the tensor won't change the output of softmax.
        # apply the subtraction to avoid value overflow in torch.nn.functional.softmax.
        # for more details, please see Equation 7 in https://arxiv.org/abs/2112.08778
        weights = weights - weights.max(dim=-1, keepdim=True)[0]

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v  # B, nH, L, Hd

        if self.hard_concrete_for_heads is not None:
            head_mask = self.hard_concrete_for_heads()  # (nH,)
            output = output * head_mask.unsqueeze(-1).unsqueeze(-1)

        output = output.transpose(2, 1).reshape(batch_size, length, self.num_heads * self.head_dim)

        output = self.out_proj(output)

        if self.hard_concrete_for_layer is not None:
            layer_mask = self.hard_concrete_for_layer() # (1,)
            output = output * layer_mask

        return output, None  # Necessary for compatibility with WavLMSelAttention

    def get_num_params(self):
        if self.hard_concrete_for_heads is not None:
            num_heads = self.hard_concrete_for_heads.l0_norm()
        else:
            num_heads = self.num_heads
        num_params = (self.embed_dim + 1) * num_heads * self.head_dim * 3 \
            + (num_heads * self.head_dim + 1) * self.embed_dim

        if self.hard_concrete_for_layer is not None:
            num_params *= self.hard_concrete_for_layer.l0_norm()
        
        return num_params

    def prune(self):
        new_config = {
            "use_attention": True,
            "num_heads": self.num_heads,
        }
        if self.hard_concrete_for_layer is not None:
            assert not self.hard_concrete_for_layer.training
            layer_mask = self.hard_concrete_for_layer() # (1,)
            self.out_proj.weight.data *= layer_mask
            self.out_proj.bias.data *= layer_mask
            if layer_mask == 0:
                new_config["use_attention"] = False
            self.hard_concrete_for_layer = None

        if self.hard_concrete_for_heads is not None:
            assert not self.hard_concrete_for_heads.training
            head_mask = self.hard_concrete_for_heads()  # (num_heads,)
            new_config["num_heads"] = len(head_mask.nonzero())
            if new_config["num_heads"] == 0:
                new_config["use_attention"] = False
            else:
                full_mask = head_mask.repeat_interleave(self.head_dim)
                full_index = full_mask.nonzero().squeeze(-1)  # 1D

                prune_linear_layer(self.k_proj, full_index, "output")
                prune_linear_layer(self.v_proj, full_index, "output")
                prune_linear_layer(self.q_proj, full_index, "output")

                self.out_proj.weight.data *= full_mask
                prune_linear_layer(self.out_proj, full_index, "input")
            self.hard_concrete_for_heads = None

        return new_config


class WavLMSelfAttention(SelfAttention):
    """Multi-headed self-attention for WavLM model :cite:`chen2022wavlm`.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional): Dropout probability on attn_output_weights. (Default: to ``0.0``)
        bias (bool, optional): If ``True``, add bias to input / output projection layers. (Default: ``True``)
        has_relative_attention_bias (bool, optional): If ``True``, apply relative position embedding.
            Necessary in the first encoder layer, but not in the subsequent ones. (Default: ``False``)
        num_buckets (int, optional): Number of buckets for relative position embedding. (Default: ``32``)
        max_distance (int, optional): Naximum distance for relative position embedding. (Default: ``128``)
        gru_rel_pos (bool, optional): If ``True``, apply gated relative position embedding. (Default: ``False``)
    """

    def __init__(
        self,
        embed_dim: int,
        total_num_heads: int,
        remaining_heads: Optional[List[int]] = None,
        dropout: float = 0.0,
        bias: bool = True,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        gru_rel_pos: bool = True,
        prune_heads: bool = False,
        prune_layer: bool = False,
    ):
        self.total_num_heads = total_num_heads
        if remaining_heads is None:
            self.remaining_heads = list(range(total_num_heads))
        else:
            self.remaining_heads = remaining_heads  # list of indices
        
        self.head_dim = embed_dim // total_num_heads

        super().__init__(embed_dim, len(self.remaining_heads), self.head_dim, dropout, prune_heads, prune_layer)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        if has_relative_attention_bias:
            self.rel_attn_embed = nn.Embedding(num_buckets, total_num_heads)
        else:
            self.rel_attn_embed = None

        # override linear layers to customize bias
        self.k_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(len(self.remaining_heads) * self.head_dim, embed_dim, bias=bias)

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)
            self.gru_rel_pos_const = nn.Parameter(torch.ones(1, total_num_heads, 1, 1))
        self.has_position_bias = True

    def compute_bias(self, query_length: int, key_length: int) -> Tensor:
        """Compute relative position embeddings for WavLM model.
        Args:
            query_length (int): Query position can take values between 0 and ``query_length - 1``.
            key_length (int): Key position can take values between 0 and ``key_length - 1``.
        Returns:
            Tensor of shape `(num_heads, query_length, key_length)`, relative positions embeddings
        """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # Shape (query_length, key_length)
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
        values = self.rel_attn_embed(relative_position_bucket)  # Shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1])
        return values

    def _relative_positions_bucket(self, relative_positions: Tensor, bidirectional: bool = True):
        """Compute relative position buckets for WavLM model. Computation similar to formula (5) in WavLM
           paper :cite:`chen2022wavlm`.
        Args:
            relative_positions (Tensor): Relative offsets between query and key positions,
                of shape ``(query_length, key_length)``.
            bidirectional (bool): If ``True``, values will be filled both above and below the diagonal in the resulting
                matrix. If ``False``, the elements above the diagonal (i.e. with negative relative offsets) will be set
                to zero. (Default ``True``)
        Returns:
            Tensor of shape ``(query_length, key_length)`` filled bucketed values of with relative positions.
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        # Shape (query_length, key_length)
        relative_buckets = torch.zeros_like(relative_positions, dtype=torch.long)

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def forward(
        self,
        query: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query (Tensor): Input of shape ``(batch_size, src_len, embed_dim)``.
            key_padding_mask (Tensor or None, optional): Mask to exclude keys that are pads, of shape
                `(batch, src_len)`, where padding elements are indicated by 1s. (Default: ``None``)
            attn_mask: Needs to be ``None``. The argument exists for compatibility with
                ``EncoderLayer``. (Default: ``None``)
            position_bias (Tensor or None, optional): Position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``. When used inside WavLM model encoder, will be
                generated in the first layer and then passed from each encoder layer to the next one.
                (Default: ``None``)
        Returns:
            attn_output (Tensor): Attention output of shape ``(batch_size, src_len, embed_dim)``.
            position_bias (Tensor or None): Position bias of shape ``(batch_size * num_heads, src_len, src_len)``.
        """
        bsz, seq_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert key_padding_mask is None

        # only for the first layer
        if self.rel_attn_embed is not None and position_bias is None:
            position_bias = self.compute_bias(seq_len, seq_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.total_num_heads, seq_len, seq_len)

        attn_mask_rel_pos: Optional[Tensor] = None
        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos:  # Apply gating on relative position bias
                query_layer = query.view(bsz, seq_len, self.total_num_heads, -1)
                query_layer = query_layer.permute(0, 2, 1, 3)

                gate_a, gate_b = torch.sigmoid(
                    self.gru_rel_pos_linear(query_layer).view(bsz, self.total_num_heads, seq_len, 2, 4).sum(-1, keepdim=False)
                ).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.view(bsz * self.total_num_heads, -1, 1) * position_bias

            attn_mask_rel_pos = attn_mask_rel_pos.view((-1, seq_len, seq_len))
            attn_mask_rel_pos = attn_mask_rel_pos.reshape(bsz, self.total_num_heads, seq_len, seq_len)[:, self.remaining_heads, :, :]

        attn_mask = attn_mask_rel_pos
        if attention_mask is not None:
            attn_mask = attn_mask + attention_mask
        if key_padding_mask is not None:
            attn_mask = attn_mask.masked_fill(
                key_padding_mask.reshape(bsz, 1, 1, seq_len),
                float("-inf")
            )
        attn_output, _ = super().forward(query, attention_mask=attn_mask)

        return attn_output, position_bias

    def prune(self):
        new_config = {
            "use_attention": True,
            "remaining_heads": self.remaining_heads,
        }
        if self.hard_concrete_for_layer is not None:
            assert not self.hard_concrete_for_layer.training
            layer_mask = self.hard_concrete_for_layer() # (1,)
            self.out_proj.weight.data *= layer_mask
            self.out_proj.bias.data *= layer_mask
            if layer_mask == 0:
                new_config["use_attention"] = False
            self.hard_concrete_for_layer = None

        if self.hard_concrete_for_heads is not None:
            assert not self.hard_concrete_for_heads.training
            head_mask = self.hard_concrete_for_heads()  # (num_heads,)
            new_config["remaining_heads"] = head_mask.nonzero().squeeze(-1).tolist()
            if len(new_config["remaining_heads"]) == 0:
                new_config["use_attention"] = False
            else:
                full_mask = head_mask.repeat_interleave(self.head_dim)
                full_index = full_mask.nonzero().squeeze(-1)  # 1D

                prune_linear_layer(self.k_proj, full_index, "output")
                prune_linear_layer(self.v_proj, full_index, "output")
                prune_linear_layer(self.q_proj, full_index, "output")

                self.out_proj.weight.data *= full_mask
                prune_linear_layer(self.out_proj, full_index, "input")
            self.hard_concrete_for_heads = None

        return new_config


class FeedForward(Module):
    """Layer that follows attention layer in encoder layer."""

    def __init__(
        self,
        io_features: int,
        intermediate_features: int,
        intermediate_dropout: float,
        output_dropout: float,
        prune_intermediate: bool = False,
        prune_layer: bool = False,
    ):
        super().__init__()
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = nn.Linear(intermediate_features, io_features)
        self.output_dropout = nn.Dropout(output_dropout)

        if prune_intermediate:
            self.hard_concrete_for_intermediate = HardConcrete(
                n_in=intermediate_features, init_mean=0.5
            )
        else:
            self.hard_concrete_for_intermediate = None
        
        if prune_layer:
            self.hard_concrete_for_layer = HardConcrete(n_in=1, init_mean=0.01)
        else:
            self.hard_concrete_for_layer = None

    def forward(self, x):
        """
        Args:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        Returns:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        """
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        if self.hard_concrete_for_intermediate is not None:
            intermediate_mask = self.hard_concrete_for_intermediate()   # (intermediate_features,)
            x = x * intermediate_mask

        x = self.output_dense(x)
        x = self.output_dropout(x)

        if self.hard_concrete_for_layer is not None:
            layer_mask = self.hard_concrete_for_layer()     # (1,)
            x = x * layer_mask

        return x
    
    def get_num_params(self):
        io_features = self.intermediate_dense.in_features
        if self.hard_concrete_for_intermediate is not None:
            intermediate_features = self.hard_concrete_for_intermediate.l0_norm()
        else:
            intermediate_features = self.intermediate_dense.out_features
        num_params = (io_features + 1) * intermediate_features + (intermediate_features + 1) * io_features

        if self.hard_concrete_for_layer is not None:
            num_params *= self.hard_concrete_for_layer.l0_norm()
        
        return num_params
    
    def prune(self):
        new_config = {
            "use_feed_forward": True,
            "ff_interm_features": self.intermediate_dense.out_features
        }
        if self.hard_concrete_for_layer is not None:
            assert not self.hard_concrete_for_layer.training
            layer_mask = self.hard_concrete_for_layer()
            self.output_dense.weight.data *= layer_mask
            self.output_dense.bias.data *= layer_mask
            if layer_mask == 0:
                new_config["use_feed_forward"] = False
            self.hard_concrete_for_layer = None

        if self.hard_concrete_for_intermediate is not None:
            assert not self.hard_concrete_for_intermediate.training
            interm_mask = self.hard_concrete_for_intermediate()
            interm_index = interm_mask.nonzero().squeeze(-1)    # NOTE: must specify dim=-1
            new_config["ff_interm_features"] = len(interm_index)
            if new_config["ff_interm_features"] == 0:
                new_config["use_feed_forward"] = False
            else:
                prune_linear_layer(self.intermediate_dense, interm_index, "output")

                self.output_dense.weight.data *= interm_mask
                prune_linear_layer(self.output_dense, interm_index, "input")
            self.hard_concrete_for_intermediate = None

        return new_config


class EncoderLayer(Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward."""

    def __init__(
        self,
        attention: Optional[Module],    # can be None if the entire layer is pruned
        dropout: float,
        layer_norm_first: bool,
        feed_forward: Optional[Module], # can be None if the entire layer is pruned
        embed_dim: int,
    ):
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Input of shape ``(batch, sequence_length, embed_dim)``.
            attention_mask (Tensor or ``None``, optional): attention mask
                of shape ``(batch, 1, sequence_length, sequence_length)``. (Default: ``None``)
            position_bias (Tensor or ``None``, optional): position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``.
                Only necessary for WavLM model, ``None`` otherwise. (Default: ``None``)
            key_padding_mask (Tensor or ``None``, optional): key padding mask of shape ``(batch_size, src_len)``.
                Only used for WavLM model, ignored otherwise. (Default: ``None``)
        Returns:
            (x, position_bias): Shapes are the same as in the input. Position bias is only relevant for WaLM model,
                ``None`` otherwise.
        """
        if self.attention is not None:
            residual = x

            if self.layer_norm_first:
                x = self.layer_norm(x)

            x, position_bias = self.attention(
                x, attention_mask=attention_mask, position_bias=position_bias, key_padding_mask=key_padding_mask
            )

            x = self.dropout(x)
            x = residual + x

        if self.layer_norm_first:
            if self.feed_forward is not None:
                x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            # NOTE: for post norm, the layer norms should always be applied even if the layers are pruned.
            x = self.layer_norm(x)
            if self.feed_forward is not None:
                x = x + self.feed_forward(x)
            x = self.final_layer_norm(x)
        return x, position_bias

    def get_num_params(self):
        num_params = self.embed_dim * 2 * 2     # two layer norms
        if self.attention is not None:
            num_params += self.attention.get_num_params()
        if self.feed_forward is not None:
            num_params += self.feed_forward.get_num_params()
        return num_params


class Transformer(Module):
    def __init__(
        self,
        pos_conv_embed: Module,
        dropout: float,
        layers: Module,
        layer_norm_first: bool,
        layer_drop: float,
    ):
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layers = layers

    def _preprocess(self, x: Tensor):
        x = x + self.pos_conv_embed(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout(x)
        return x

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
    ) -> Tensor:
        x = self._preprocess(x)
        for layer in self.layers:
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x, position_bias = layer(x, attention_mask, position_bias=position_bias)

        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x

    def get_intermediate_outputs(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
        position_bias: Optional[Tensor] = None,
    ) -> List[Tensor]:
        if num_layers is not None:
            if not 0 < num_layers <= len(self.layers):
                raise ValueError(f"`num_layers` must be between [1, {len(self.layers)}]")

        ret: List[Tensor] = []
        x = self._preprocess(x)
        for layer in self.layers:
            x, position_bias = layer(x, attention_mask, position_bias=position_bias)
            ret.append(x)
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret
    
    def get_num_params(self):
        # pos_conv_embed and layer_norm
        num_params = sum(p.numel() for p in self.pos_conv_embed.parameters()) + self.pos_conv_embed.embed_dim * 2
        for layer in self.layers:
            num_params += layer.get_num_params()
        return num_params
    
    def prune(self):
        new_config = defaultdict(list)
        for layer in self.layers:
            attention_config = layer.attention.prune()
            new_config["use_attention"].append(attention_config["use_attention"])
            if "remaining_heads" in attention_config:
                new_config["remaining_heads"].append(attention_config["remaining_heads"])
            else:
                new_config["num_heads"].append(attention_config["num_heads"])

            if not attention_config["use_attention"]:
                layer.attention = None
            
            ff_config = layer.feed_forward.prune()
            new_config["use_feed_forward"].append(ff_config["use_feed_forward"])
            new_config["ff_interm_features"].append(ff_config["ff_interm_features"])
            if not ff_config["use_feed_forward"]:
                layer.feed_forward = None
        
        return new_config


class Encoder(Module):
    def __init__(
        self,
        feature_projection: Module,
        transformer: Module,
    ):
        super().__init__()
        self.feature_projection = feature_projection
        self.transformer = transformer

    def _preprocess(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.feature_projection(features)

        mask: Optional[Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # create mask for padded elements and zero-out them
            mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            x[mask] = 0.0
            # extend the mask to attention shape and set weight
            mask = -10000.0 * mask[:, None, None, :].to(dtype=features.dtype)
            mask = mask.expand(batch_size, 1, max_len, max_len)
        return x, mask

    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        x, mask = self._preprocess(features, lengths)
        x = self.transformer(x, attention_mask=mask)
        return x

    def extract_features(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        x, masks = self._preprocess(features, lengths)
        interm = self.transformer.get_intermediate_outputs(x, attention_mask=masks, num_layers=num_layers)
        return [x] + interm
    
    def get_num_params(self, in_features):
        """Calculate the current model size."""
        feature_projection_size = self.feature_projection.get_num_params(in_features)
        transformer_size = self.transformer.get_num_params()
        return feature_projection_size + transformer_size
    
    def prune(self, conv_out_index):
        """In-place pruning of submodules."""
        prune_layer_norm(self.feature_projection.layer_norm, conv_out_index)
        prune_linear_layer(self.feature_projection.projection, conv_out_index, "input")
        transformer_config = self.transformer.prune()
        return transformer_config


################################################################################
def _get_feature_extractor(
    norm_mode: str,
    shapes: List[Tuple[int, int, int]],
    bias: bool,
    prune_conv_channels: bool = False,
) -> FeatureExtractor:
    """
    Args:
        norm_mode (str):
            Either "group_norm" or "layer_norm".
            If "group_norm", then a single normalization is applied
            in the first convolution block. Otherwise, all the convolution
            blocks will have layer normalization.
            This option corresponds to "extractor_mode" from fairseq.
            Expected values are "group_norm" for Base arch, and
            "layer_norm" for Large arch.
        shapes (list of tuple of int):
            Configuration of convolution layers. List of convolution configuration,
            i.e. ``[(output_channel, kernel_size, stride), ...]``
            This option corresponds to "conv_feature_layers" from fairseq.
            Expected values are
            ``[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2``
            for all the architectures.
        bias (bool):
            Whether to include bias term to each convolution operation.
            This option corresponds to "conv_bias" from fairseq.
            Expected values are False for Base arch, and True for Large arch.

    See Also:
        * Original implementation
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L666-L733
        * "extractor_mode"
          - Def and base:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L38-L45
          - Large:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L52
        * "conv_feature_layers"
          - Def, base and large:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L94-L100
        * "conv_bias"
          - Def and base:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L101-L103
          - Large:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L61
    """
    if norm_mode not in ["group_norm", "layer_norm"]:
        raise ValueError("Invalid norm mode")
    blocks = []
    in_channels = 1
    for i, (out_channels, kernel_size, stride) in enumerate(shapes):
        normalization = None
        if norm_mode == "group_norm" and i == 0:
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "layer_norm":
            normalization = LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        blocks.append(
            ConvLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
                prune_conv_channels=prune_conv_channels,
            )
        )
        in_channels = out_channels
    return FeatureExtractor(nn.ModuleList(blocks))


def _get_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    use_attention: List[bool],
    use_feed_forward: List[bool],
    num_heads: List[int],
    head_dim: int,
    attention_dropout: float,
    ff_interm_features: List[int],
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
    prune_attention_heads: bool = False,
    prune_attention_layer: bool = False,
    prune_feed_forward_intermediate: bool = False,
    prune_feed_forward_layer: bool = False,
) -> Encoder:
    """
    Args:
        in_features (int): The number of input features.
        embed_dim (int):
            The dimension of embedding.
            This option corresponds to "encoder_embed_dim" from fairseq.
            Expected values are 768 for Base arch, and 1024 for Large arch.
        dropout_input (float):
            The dropout probability applied after the input feature is projected
            to ``embed_dim``.
            This option corresponds to "dropout_input" from fairseq.
            Expected values are 0.1 for both Base and Large arch.
        pos_conv_kernel (int):
            The kernel size of convolutional positional embeddings.
            This option corresponds to "conv_pos" from fairseq.
            Expected values are 128 for both Base and Large arch.
        pos_conv_groups (int):
            The number of groups of convolutional positional embeddings.
            This option corresponds to "conv_pos_groups" from fairseq.
            Expected values are 16 for both Base and Large arch.
        num_layers (int):
            The number of self attention layers in transformer block.
            This option corresponds to "encoder_layers" from fairseq.
            Expected values are 12 for Base and 24 for Large arch.
        num_heads (int):
            The number of heads in self attention layers.
            This option corresponds to "encoder_attention_heads" from fairseq.
            Expected values are 12 for Base and 16 for Large arch.
        attention_dropout (float):
            The dropout probability applied after softmax in self-attention layer.
            This option corresponds to "attention_dropout" from fairseq.
            Expected values are 0.1 for Base and 0.0 for Large arch.
        ff_interm_features (int):
            The dimension of hidden features in feed forward layer.
            This option corresponds to "encoder_ffn_embed_dim" from fairseq.
            Expected values are 3072 for Base and 4096 for Large arch.
        ff_interm_dropout (float):
            The dropout probability applied in feedforward layer.
            This option correspinds to "activation_dropout" from fairseq.
            Expected values are 0.1 for both Base and Large arch.
        dropout (float):
            The dropout probability applied at the end of feed forward layer.
            This option corresponds to "dropout" from fairseq.
            Expected values are 0.1 for Base and 0.0 for Large arch.
        layer_norm_first (bool):
            Control the order of layer norm in transformer layer and each encoder layer.
            If True, in transformer layer, layer norm is applied before features are fed
            to encoder layers. In encoder layer, two layer norms are applied before and after
            self attention.
            If False, in transformer layer, layer norm is applied after features are fed
            to encoder layers. In encoder layer, two layer norms are applied after self
            attention, before and after feed forward.
            This option corresponds to "layer_norm_first" from fairseq.
            Expected values are False for Base and True for Large arch.
        layer_drop (float):
            Probability to drop each encoder layer during training.
            This option corresponds to "layerdrop" from fairseq.
            Expected values are 0.1 for both Base and Large arch.

    See Also:
        * "encoder_embed_dim"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L49-L51
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L64
        * "dropout_input"
          - Def, base and large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L75-L78
        * "conv_pos"
          - Def, base and large
            NOTE: The description is wrong.
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L204-L207
          - Usage
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L756
        * "conv_pos_groups"
          - Def, base and large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L208-L211
        * "encoder_layers"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L46-L48
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L63
        * "encoder_attention_heads"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L55-L57
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L66
        * "attention_dropout"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L66-L68
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L60
        * "encoder_ffn_embed_dim"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L52-L54
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L65
        * "activation_dropout"
          - Def
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L69-L71
          - Base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/base_960h.yaml#L55
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/vox_960h.yaml#L55
        * "dropout"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L63-L65
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L59
        * "layer_norm_first"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L91-L93
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L53
        * "layerdrop"
          - Def
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L72-L74
          - Base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/base_960h.yaml#L54
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/vox_960h.yaml#L54
    """
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    # Original impl
    # https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L768-L782
    encoder_layers = nn.ModuleList()
    for idx in range(num_layers):
        if use_attention[idx]:
            attention = SelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads[idx],
                head_dim=head_dim,
                dropout=attention_dropout,
                prune_heads=prune_attention_heads,
                prune_layer=prune_attention_layer,
            )
        else:
            attention = None
        if use_feed_forward[idx]:
            feed_forward = FeedForward(
                io_features=embed_dim,
                intermediate_features=ff_interm_features[idx],
                intermediate_dropout=ff_interm_dropout,
                output_dropout=dropout,
                prune_intermediate=prune_feed_forward_intermediate,
                prune_layer=prune_feed_forward_layer,
            )
        else:
            feed_forward = None
        encoder_layers.append(
            EncoderLayer(
                attention=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
                embed_dim=embed_dim,
            )
        )
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)


def _get_wavlm_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    use_attention: List[bool],
    use_feed_forward: List[bool],
    total_num_heads: List[int],
    remaining_heads: List[List[int]],
    num_buckets: int,
    max_distance: int,
    attention_dropout: float,
    ff_interm_features: List[int],
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
    prune_attention_heads: bool = False,
    prune_attention_layer: bool = False,
    prune_feed_forward_intermediate: bool = False,
    prune_feed_forward_layer: bool = False,
) -> Encoder:
    """
    Construct encoder for WavLM model :cite:`chen2022wavlm`. The structure of the encoder and most of the argments are
    the same as in :py:func:`_get_encoder` so refer there for documentation. The only difference from Wav2Vec2 encoder
    is usage of `WavLMSelfAttention` instead of `SelfAttention` and two additional parameters: `num_buckets` and
    `max_distance`.
    Args:
        in_features (int): See :py:func:`_get_encoder`.
        embed_dim (int): See :py:func:`_get_encoder`.
        dropout_input (float): See :py:func:`_get_encoder`.
        pos_conv_kernel (int): See :py:func:`_get_encoder`.
        pos_conv_groups (int): See :py:func:`_get_encoder`.
        num_layers (int): See :py:func:`_get_encoder`.
        num_heads (int): See :py:func:`_get_encoder`.
        num_buckets (int): Number of buckets for relative position embedding.
        max_distance (int): Maximum distance for relative position embedding.
        attention_dropout (float): See :py:func:`_get_encoder`.
        ff_interm_features (int): See :py:func:`_get_encoder`.
        ff_interm_dropout (float): See :py:func:`_get_encoder`.
        dropout (float): See :py:func:`_get_encoder`.
        layer_norm_first (bool): See :py:func:`_get_encoder`.
        layer_drop (float): See :py:func:`_get_encoder`.

    """
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    # Original impl
    # https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L768-L782
    encoder_layers = nn.ModuleList()
    for i in range(num_layers):
        if use_attention[i]:
            attention = WavLMSelfAttention(
                embed_dim=embed_dim,
                total_num_heads=total_num_heads[i],
                remaining_heads=remaining_heads[i],
                dropout=attention_dropout,
                has_relative_attention_bias=(i == 0),  # Position embedding is only necessary in the first layer.
                num_buckets=num_buckets,
                max_distance=max_distance,
                prune_heads=prune_attention_heads,
                prune_layer=prune_attention_layer,
            )
        else:
            attention = None
        if use_feed_forward[i]:
            feed_forward = FeedForward(
                io_features=embed_dim,
                intermediate_features=ff_interm_features[i],
                intermediate_dropout=ff_interm_dropout,
                output_dropout=dropout,
                prune_intermediate=prune_feed_forward_intermediate,
                prune_layer=prune_feed_forward_layer,
            )
        else:
            feed_forward = None
        encoder_layers.append(
            EncoderLayer(
                attention=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
                embed_dim=embed_dim,
            )
        )
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)


def _get_padding_mask(input: Tensor, lengths: Tensor) -> Tensor:
    """Generate the padding mask given the padded input and the lengths Tensors.
    Args:
        input (Tensor): The padded Tensor of dimension `[batch, max_len, frequency]`.
        lengths (Tensor): The lengths Tensor of dimension `[batch,]`.

    Returns:
        (Tensor): The padding mask.
    """
    batch_size, max_len, _ = input.shape
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
    return mask


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None
