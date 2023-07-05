import torch.nn as nn
from torch.nn.utils import remove_weight_norm, weight_norm


class Depthwise_Separable_Conv1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        bias = True,
        padding_mode = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ):
      super().__init__()
      self.depth_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride = stride,padding=padding,dilation=dilation,bias=bias,padding_mode=padding_mode,device=device,dtype=dtype)
      self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias, device=device,dtype=dtype)
    
    def forward(self, input):
      return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
      self.depth_conv = weight_norm(self.depth_conv, name = 'weight')
      self.point_conv = weight_norm(self.point_conv, name = 'weight')

    def remove_weight_norm(self):
      self.depth_conv = remove_weight_norm(self.depth_conv, name = 'weight')
      self.point_conv = remove_weight_norm(self.point_conv, name = 'weight')

class Depthwise_Separable_TransposeConv1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0, 
        output_padding = 0,
        bias = True,
        dilation = 1,
        padding_mode = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ):
      super().__init__()
      self.depth_conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride = stride,output_padding=output_padding,padding=padding,dilation=dilation,bias=bias,padding_mode=padding_mode,device=device,dtype=dtype)
      self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias, device=device,dtype=dtype)
    
    def forward(self, input):
      return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
      self.depth_conv = weight_norm(self.depth_conv, name = 'weight')
      self.point_conv = weight_norm(self.point_conv, name = 'weight')

    def remove_weight_norm(self):
      remove_weight_norm(self.depth_conv, name = 'weight')
      remove_weight_norm(self.point_conv, name = 'weight')


def weight_norm_modules(module, name = 'weight', dim = 0):
    if isinstance(module,Depthwise_Separable_Conv1D) or isinstance(module,Depthwise_Separable_TransposeConv1D):
      module.weight_norm()
      return module
    else:
      return weight_norm(module,name,dim)

def remove_weight_norm_modules(module, name = 'weight'):
    if isinstance(module,Depthwise_Separable_Conv1D) or isinstance(module,Depthwise_Separable_TransposeConv1D):
      module.remove_weight_norm()
    else:
      remove_weight_norm(module,name)