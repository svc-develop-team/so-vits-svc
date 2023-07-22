import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from local_attention import LocalAttention
from torch import nn

#import fast_transformers.causal_product.causal_product_cuda

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape
    # (batch size, head, length, model_dim)

    # normalize model dim
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    # what is ration?, projection_matrix.shape[0] --> 266
    
    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    #data_dash = w^T x
    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    
    # diag_data = D**2 
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)
    
    #print ()
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data + eps))#- torch.max(data_dash)) + eps)

    return data_dash.type_as(data)

def orthogonal_matrix_chunk(cols, qr_uniform_q = False, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))

    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()
def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

class PCmer(nn.Module):
    """The encoder that is used in the Transformer model."""
    
    def __init__(self, 
                num_layers,
                num_heads,
                dim_model,
                dim_keys,
                dim_values,
                residual_dropout,
                attention_dropout):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout

        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])
        
    #  METHODS  ########################################################################################################
    
    def forward(self, phone, mask=None):
        
        # apply all layers to the input
        for (i, layer) in enumerate(self._layers):
            phone = layer(phone, mask)
        # provide the final sequence
        return phone


# ==================================================================================================================== #
#  CLASS  _ E N C O D E R  L A Y E R                                                                                   #
# ==================================================================================================================== #


class _EncoderLayer(nn.Module):
    """One layer of the encoder.
    
    Attributes:
        attn: (:class:`mha.MultiHeadAttention`): The attention mechanism that is used to read the input sequence.
        feed_forward (:class:`ffl.FeedForwardLayer`): The feed-forward layer on top of the attention mechanism.
    """
    
    def __init__(self, parent: PCmer):
        """Creates a new instance of ``_EncoderLayer``.
        
        Args:
            parent (Encoder): The encoder that the layers is created for.
        """
        super().__init__()
        
        
        self.conformer = ConformerConvModule(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)
        
        # selfatt -> fastatt: performer!
        self.attn = SelfAttention(dim = parent.dim_model,
                                  heads = parent.num_heads,
                                  causal = False)
        
    #  METHODS  ########################################################################################################

    def forward(self, phone, mask=None):
        
        # compute attention sub-layer
        phone = phone + (self.attn(self.norm(phone), mask=mask))
        
        phone = phone + (self.conformer(phone))
        
        return phone 

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            #nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

def linear_attention(q, k, v):
    if v is None:
        #print (k.size(), q.size())
        out = torch.einsum('...ed,...nd->...ne', k, q)
        return out

    else:
        k_cumsum = k.sum(dim = -2) 
        #k_cumsum = k.sum(dim = -2)
        D_inv = 1. / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + 1e-8)

        context = torch.einsum('...nd,...ne->...de', k, v)
        #print ("TRUEEE: ", context.size(), q.size(), D_inv.size())
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, qr_uniform_q = False, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)
    #print (nb_full_blocks)
    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        block_list.append(q)
    # block_list[n] is a orthogonal matrix ... (model_dim * model_dim)
    #print (block_list[0].size(), torch.einsum('...nd,...nd->...n', block_list[0], torch.roll(block_list[0],1,1)))
    #print (nb_rows, nb_full_blocks, nb_columns)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    #print (remaining_rows)
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        #print (q[:remaining_rows].size())
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)
    
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, no_projection = False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling, qr_uniform_q = qr_uniform_q)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal

    @torch.no_grad()
    def redraw_projection_matrix(self):
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)
        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        if v is None:
            out = attn_fn(q, k, None)
            return out
        else:
            out = attn_fn(q, k, v)
            return out
class SelfAttention(nn.Module):
    def __init__(self, dim, causal = False, heads = 8, dim_head = 64, local_heads = 0, local_window_size = 256, nb_features = None, feature_redraw_interval = 1000, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, dropout = 0., no_projection = False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, qr_uniform_q = qr_uniform_q, no_projection = no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        #print (heads, nb_features, dim_head)
        #name_embedding = torch.zeros(110, heads, dim_head, dim_head)
        #self.name_embedding = nn.Parameter(name_embedding, requires_grad=True)
        

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()
        #torch.nn.init.zeros_(self.name_embedding)
        #print (torch.sum(self.name_embedding))
    def forward(self, x, context = None, mask = None, context_mask = None, name=None, inference=False, **kwargs):
        _, _, _, h, gh = *x.shape, self.heads, self.global_heads
        
        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask
        #print (torch.sum(self.name_embedding))
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []
        #print (name)
        #print (self.name_embedding[name].size())
        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)
            if cross_attend:
                pass
                #print (torch.sum(self.name_embedding))
                #out = self.fast_attention(q,self.name_embedding[name],None)
                #print (torch.sum(self.name_embedding[...,-1:]))
            else:
                out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.dropout(out)