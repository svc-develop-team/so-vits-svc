"""Speech SSL models supporting pruning.

Originally from:
https://github.com/pytorch/audio/blob/main/torchaudio/models/wav2vec2/model.py

"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from . import components


class Wav2Vec2Model(Module):
    """Acoustic model used in *wav2vec 2.0* :cite:`baevski2020wav2vec`.

    Note:
        To build the model, please use one of the factory functions.
        :py:func:`wav2vec2_model`, :py:func:`wav2vec2_base`, :py:func:`wav2vec2_large`,
        :py:func:`wav2vec2_large_lv60k`, :py:func:`hubert_base`, :py:func:`hubert_large`,
        and :py:func:`hubert_xlarge`.

    See Also:
        * :class:`torchaudio.pipelines.Wav2Vec2Bundle`: Pretrained models (without fine-tuning)
        * :class:`torchaudio.pipelines.Wav2Vec2ASRBundle`: ASR pipelines with pretrained models.

    Args:
        feature_extractor (torch.nn.Module):
            Feature extractor that extracts feature vectors from raw audio Tensor.

        encoder (torch.nn.Module):
            Encoder that converts the audio features into the sequence of probability
            distribution (in negative log-likelihood) over labels.

        aux (torch.nn.Module or None, optional):
            Auxiliary module. If provided, the output from encoder is passed to this module.
    """  # noqa: E501

    def __init__(
        self,
        normalize_waveform: bool,
        feature_extractor: Module,
        encoder: Module,
        aux: Optional[Module] = None,
    ):
        super().__init__()
        self.normalize_waveform = normalize_waveform
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.aux = aux

    @torch.jit.export
    def extract_features(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> Tuple[List[Tensor], Optional[Tensor]]:
        """Extract feature vectors from raw waveforms

        This returns the list of outputs from the intermediate layers of
        transformer block in encoder.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that the entire audio waveform
                length is valid.
            num_layers (int or None, optional):
                If given, limit the number of intermediate layers to go through.
                Providing `1` will stop the computation after going through one
                intermediate layers. If not given, the outputs from all the
                intermediate layers are returned.

        Returns:
            (List[Tensor], Optional[Tensor]):
            List of Tensors
                Features from requested layers.
                Each Tensor is of shape: `(batch, time frame, feature dimension)`
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of each feature Tensor.
        """
        if self.normalize_waveform:
            if lengths is not None:
                waveforms = [
                    F.layer_norm(wave[:length], (length,)) for wave, length in zip(waveforms, lengths)
                ]
                waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
            else:
                waveforms = F.layer_norm(waveforms, waveforms.shape[-1:])

        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder.extract_features(x, lengths, num_layers)   # (num_layers+1,), including the input
        return x, lengths
    
    def get_num_params(self):
        """Calculate the current size."""
        feature_extractor_size, encoder_in_features = self.feature_extractor.get_num_params_and_final_out_channels()
        encoder_size = self.encoder.get_num_params(encoder_in_features)
        return feature_extractor_size + encoder_size
    
    def prune(self):
        self.eval()     # must be in eval mode
        conv_config, conv_out_index = self.feature_extractor.prune()    # [(output_channel, kernel_size, stride), ...]
        transformer_config = self.encoder.prune(conv_out_index)     # NOTE: this is a defaultdict(list)
        use_attention = transformer_config["use_attention"]
        use_feed_forward = transformer_config["use_feed_forward"]
        num_heads = transformer_config["num_heads"]     # can be []
        remaining_heads = transformer_config["remaining_heads"]     # can be []
        ff_interm_features = transformer_config["ff_interm_features"]

        return conv_config, use_attention, use_feed_forward, num_heads, remaining_heads, ff_interm_features

    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute the sequence of probability distribution over labels.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that all the audio in ``waveforms``
                have valid length. Default: ``None``.

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor
                The sequences of probability distribution (in logit) over labels.
                Shape: `(batch, frames, num labels)`.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of the output Tensor.
        """
        if self.normalize_waveform:
            if lengths is not None:
                waveforms = [
                    F.layer_norm(wave[:length], (length,)) for wave, length in zip(waveforms, lengths)
                ]
                waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
            else:
                waveforms = F.layer_norm(waveforms, waveforms.shape[-1:])

        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder(x, lengths)
        if self.aux is not None:
            x = self.aux(x)
        return x, lengths


def wav2vec2_model(**configs) -> Wav2Vec2Model:
    """Wraps the original wav2vec2_model and wavlm_model."""

    if "encoder_remaining_heads" in configs:
        return wavlm_model(**configs)
    
    return wav2vec2_model_original(**configs)


def wav2vec2_model_original(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_use_attention: List[bool],
    encoder_use_feed_forward: List[bool],
    encoder_num_heads: List[int],
    encoder_head_dim: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: List[int],
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: Optional[int],
    normalize_waveform: bool,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds custom :class:`~torchaudio.models.Wav2Vec2Model`.

    Note:
        The "feature extractor" below corresponds to
        `ConvFeatureExtractionModel <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L736>`__
        in the original ``fairseq`` implementation.
        This is referred as "(convolutional) feature encoder" in the *wav2vec 2.0*
        :cite:`baevski2020wav2vec` paper.

        The "encoder" below corresponds to `TransformerEncoder <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L817>`__,
        and this is referred as "Transformer" in the paper.

    Args:
        extractor_mode (str): Operation mode of feature extractor.
            Valid values are ``"group_norm"`` or ``"layer_norm"``.
            If ``"group_norm"``, then a single normalization is applied
            in the first convolution block. Otherwise, all the convolution
            blocks will have layer normalization.

            This option corresponds to ``extractor_mode`` from ``fairseq``.
        extractor_conv_layer_config (list of integer tuples or None):
            Configuration of convolution layers in feature extractor.
            List of convolution configuration,
            i.e. ``[(output_channel, kernel_size, stride), ...]``

            If ``None`` is provided, then the following default value is used.

            .. code-block:: python

               [
                 (512, 10, 5),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 2, 2),
                 (512, 2, 2),
               ]

            This option corresponds to ``conv_feature_layers`` from ``fairseq``.

        extractor_conv_bias (bool):
            Whether to include bias term to each convolution operation.

            This option corresponds to ``conv_bias`` from ``fairseq``.

        encoder_embed_dim (int):
            The dimension of embedding in encoder.

            This option corresponds to ``encoder_embed_dim`` from ``fairseq``.

        encoder_projection_dropout (float):
            The dropout probability applied after the input feature is projected
            to ``encoder_embed_dim``.

            This option corresponds to ``dropout_input`` from ``fairseq``.

        encoder_pos_conv_kernel (int):
            The kernel size of convolutional positional embeddings.

            This option corresponds to ``conv_pos`` from ``fairseq``.

        encoder_pos_conv_groups (int):
            The number of groups of convolutional positional embeddings.

            This option corresponds to ``conv_pos_groups`` from ``fairseq``.

        encoder_num_layers (int):
            The number of self attention layers in transformer block.

            This option corresponds to ``encoder_layers`` from ``fairseq``.

        encoder_num_heads (int):
            The number of heads in self attention layers.

            This option corresponds to ``encoder_attention_heads`` from ``fairseq``.

        encoder_attention_dropout (float):
            The dropout probability applied after softmax in self-attention layer.

            This option corresponds to ``attention_dropout`` from ``fairseq``.

        encoder_ff_interm_features (int):
            The dimension of hidden features in feed forward layer.

            This option corresponds to ``encoder_ffn_embed_dim`` from ``fairseq``.

        encoder_ff_interm_dropout (float):
            The dropout probability applied in feedforward layer.

            This option correspinds to ``activation_dropout`` from ``fairseq``.

        encoder_dropout (float):
            The dropout probability applied at the end of feed forward layer.

            This option corresponds to ``dropout`` from ``fairseq``.

        encoder_layer_norm_first (bool):
            Control the order of layer norm in transformer layer and each encoder layer.
            If True, in transformer layer, layer norm is applied before features are fed
            to encoder layers. In encoder layer, two layer norms are applied before and after
            self attention.
            If False, in transformer layer, layer norm is applied after features are fed
            to encoder layers. In encoder layer, two layer norms are applied after self
            attention, before and after feed forward.

            This option corresponds to ``layer_norm_first`` from ``fairseq``.

        encoder_layer_drop (float):
            Probability to drop each encoder layer during training.

            This option corresponds to ``layerdrop`` from ``fairseq``.

        aux_num_out (int or None):
            When provided, attach an extra linear layer on top of encoder, which can be
            used for fine-tuning.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    feature_extractor = components._get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias, 
        prune_conv_channels=extractor_prune_conv_channels,
    )
    encoder = components._get_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        use_attention=encoder_use_attention,
        use_feed_forward=encoder_use_feed_forward,
        num_heads=encoder_num_heads,
        head_dim=encoder_head_dim,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
        prune_attention_heads=encoder_prune_attention_heads,
        prune_attention_layer=encoder_prune_attention_layer,
        prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)
    return Wav2Vec2Model(normalize_waveform, feature_extractor, encoder, aux)


def wav2vec2_base(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds "base" :class:`~torchaudio.models.Wav2Vec2Model` from *wav2vec 2.0* :cite:`baevski2020wav2vec`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=3072,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )


def wav2vec2_large(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds "large" :class:`~torchaudio.models.Wav2Vec2Model` from *wav2vec 2.0* :cite:`baevski2020wav2vec`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )


def wav2vec2_large_lv60k(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.0,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.0,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds "large lv-60k" :class:`~torchaudio.models.Wav2Vec2Model` from *wav2vec 2.0* :cite:`baevski2020wav2vec`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )


def hubert_base(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.05,
    aux_num_out: Optional[int] = None,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds "base" :class:`HuBERT <torchaudio.models.Wav2Vec2Model>` from *HuBERT* :cite:`hsu2021hubert`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_use_attention=[True] * 12,
        encoder_use_feed_forward=[True] * 12,
        encoder_num_heads=[12] * 12,
        encoder_head_dim=64,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=[3072] * 12,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )


def hubert_large(
    encoder_projection_dropout: float = 0.0,
    encoder_attention_dropout: float = 0.0,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.0,
    encoder_layer_drop: float = 0.0,
    aux_num_out: Optional[int] = None,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds "large" :class:`HuBERT <torchaudio.models.Wav2Vec2Model>` from *HuBERT* :cite:`hsu2021hubert`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )


def hubert_xlarge(
    encoder_projection_dropout: float = 0.0,
    encoder_attention_dropout: float = 0.0,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.0,
    encoder_layer_drop: float = 0.0,
    aux_num_out: Optional[int] = None,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds "extra large" :class:`HuBERT <torchaudio.models.Wav2Vec2Model>` from *HuBERT* :cite:`hsu2021hubert`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1280,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=48,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=5120,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )


def _init_hubert_pretrain_model(module):
    if isinstance(module, components.LayerNorm):
        torch.nn.init.kaiming_normal_(module.conv.weight)
    elif isinstance(module, components.ConvolutionalPositionalEmbedding):
        # normalize the weight to normal distribution.
        std = math.sqrt(4.0 / (module.embed_dim * module.kernel_size))
        torch.nn.init.normal_(module.conv.weight, mean=0.0, std=std)
        torch.nn.init.constant_(module.conv.bias, 0.0)
    elif isinstance(module, components.SelfAttention):
        # normalize the query, key, value, and out_proj parameters in self attention module.
        torch.nn.init.xavier_uniform_(module.k_proj.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(module.v_proj.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(module.q_proj.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(module.out_proj.weight)
        torch.nn.init.constant_(module.out_proj.bias, 0.0)
    elif isinstance(module, components.Transformer):
        module.apply(components._init_transformer_params)
    else:
        pass


def wavlm_model(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_use_attention: List[bool],
    encoder_use_feed_forward: List[bool],
    encoder_total_num_heads: List[int],
    encoder_remaining_heads: List[List[int]],
    encoder_num_buckets: int,
    encoder_max_distance: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: List[int],
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: Optional[int],
    normalize_waveform: bool,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds custom WaveLM model :cite:`chen2022wavlm`. The architecture is compatible
    with Wav2Vec2 model :cite:`baevski2020wav2vec`, and so the output object is
    :class:`~torchaudio.models.Wav2Vec2Model`. Most of the arguments have the same meaning
    as in :py:func:`wav2vec2_model` so please refer there for documentation.

    Args:
        extractor_mode (str): Operation mode of feature extractor.
            See :py:func:`wav2vec2_model`.

        extractor_conv_layer_config (list of integer tuples or None):
            See :py:func:`wav2vec2_model`.

        extractor_conv_bias (bool):
            See :py:func:`wav2vec2_model`.

        encoder_embed_dim (int):
            See :py:func:`wav2vec2_model`.

        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.

        encoder_pos_conv_kernel (int):
            See :py:func:`wav2vec2_model`.

        encoder_pos_conv_groups (int):
            See :py:func:`wav2vec2_model`.

        encoder_num_layers (int):
            See :py:func:`wav2vec2_model`.

        encoder_num_heads (int):
            See :py:func:`wav2vec2_model`.

        encoder_num_buckets (int):
            Number of buckets for relative position embedding.
        encoder_max_distance (int):
            Maximum distance for relative position embedding.

        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.

        encoder_ff_interm_features (int):
            See :py:func:`wav2vec2_model`.

        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.

        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.

        encoder_layer_norm_first (bool):
            See :py:func:`wav2vec2_model`.

        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.

        aux_num_out (int or None):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    feature_extractor = components._get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias,
        prune_conv_channels=extractor_prune_conv_channels,
    )
    encoder = components._get_wavlm_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        use_attention=encoder_use_attention,
        use_feed_forward=encoder_use_feed_forward,
        total_num_heads=encoder_total_num_heads,
        remaining_heads=encoder_remaining_heads,
        num_buckets=encoder_num_buckets,
        max_distance=encoder_max_distance,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
        prune_attention_heads=encoder_prune_attention_heads,
        prune_attention_layer=encoder_prune_attention_layer,
        prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)
    return Wav2Vec2Model(normalize_waveform, feature_extractor, encoder, aux)


def wavlm_base(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    """Builds "base" WaveLM model :cite:`chen2022wavlm`. The architecture is compatible
    with Wav2Vec2 model :cite:`baevski2020wav2vec`, and so the output class is
    :class:`~torchaudio.models.Wav2Vec2Model`.

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    return wavlm_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_num_buckets=320,
        encoder_max_distance=800,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=3072,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )


def wavlm_large(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    """Builds "large" WaveLM model :cite:`chen2022wavlm`. The architecture is compatible
    with Wav2Vec2 model :cite:`baevski2020wav2vec`, and so the output class is
    :class:`~torchaudio.models.Wav2Vec2Model`.

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    return wavlm_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_num_buckets=320,
        encoder_max_distance=800,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )
