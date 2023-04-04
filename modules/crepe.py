from typing import Optional,Union
try:
    from typing import Literal
except Exception as e:
    from typing_extensions import Literal
import numpy as np
import torch
import torchcrepe
from torch import nn
from torch.nn import functional as F
import scipy

#from:https://github.com/fishaudio/fish-diffusion

def repeat_expand(
    content: Union[torch.Tensor, np.ndarray], target_len: int, mode: str = "nearest"
):
    """Repeat content to target length.
    This is a wrapper of torch.nn.functional.interpolate.

    Args:
        content (torch.Tensor): tensor
        target_len (int): target length
        mode (str, optional): interpolation mode. Defaults to "nearest".

    Returns:
        torch.Tensor: tensor
    """

    ndim = content.ndim

    if content.ndim == 1:
        content = content[None, None]
    elif content.ndim == 2:
        content = content[None]

    assert content.ndim == 3

    is_np = isinstance(content, np.ndarray)
    if is_np:
        content = torch.from_numpy(content)

    results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

    if is_np:
        results = results.numpy()

    if ndim == 1:
        return results[0, 0]
    elif ndim == 2:
        return results[0]


class BasePitchExtractor:
    def __init__(
        self,
        hop_length: int = 512,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        keep_zeros: bool = True,
    ):
        """Base pitch extractor.

        Args:
            hop_length (int, optional): Hop length. Defaults to 512.
            f0_min (float, optional): Minimum f0. Defaults to 50.0.
            f0_max (float, optional): Maximum f0. Defaults to 1100.0.
            keep_zeros (bool, optional): Whether keep zeros in pitch. Defaults to True.
        """

        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.keep_zeros = keep_zeros

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        raise NotImplementedError("BasePitchExtractor is not callable.")

    def post_process(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = repeat_expand(f0, pad_to)

        if self.keep_zeros:
            return f0
        
        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
        
        # 去掉0频率, 并线性插值
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device),torch.zeros(pad_to, dtype=torch.float, device=x.device)

        if f0.shape[0] == 1:
            return torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0],torch.ones(pad_to, dtype=torch.float, device=x.device)
    
        # 大概可以用 torch 重写?
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        vuv_vector = vuv_vector.cpu().numpy()
        vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))
        
        return f0,vuv_vector


class MaskedAvgPool1d(nn.Module):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        """An implementation of mean pooling that supports masked values.

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """

        super(MaskedAvgPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x, mask=None):
        ndim = x.dim()
        if ndim == 2:
            x = x.unsqueeze(1)

        assert (
            x.dim() == 3
        ), "Input tensor must have 2 or 3 dimensions (batch_size, channels, width)"

        # Apply the mask by setting masked elements to zero, or make NaNs zero
        if mask is None:
            mask = ~torch.isnan(x)

        # Ensure mask has the same shape as the input tensor
        assert x.shape == mask.shape, "Input tensor and mask must have the same shape"

        masked_x = torch.where(mask, x, torch.zeros_like(x))
        # Create a ones kernel with the same number of channels as the input tensor
        ones_kernel = torch.ones(x.size(1), 1, self.kernel_size, device=x.device)

        # Perform sum pooling
        sum_pooled = nn.functional.conv1d(
            masked_x,
            ones_kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.size(1),
        )

        # Count the non-masked (valid) elements in each pooling window
        valid_count = nn.functional.conv1d(
            mask.float(),
            ones_kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.size(1),
        )
        valid_count = valid_count.clamp(min=1)  # Avoid division by zero

        # Perform masked average pooling
        avg_pooled = sum_pooled / valid_count

        # Fill zero values with NaNs
        avg_pooled[avg_pooled == 0] = float("nan")

        if ndim == 2:
            return avg_pooled.squeeze(1)

        return avg_pooled


class MaskedMedianPool1d(nn.Module):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        """An implementation of median pooling that supports masked values.

        This implementation is inspired by the median pooling implementation in
        https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """

        super(MaskedMedianPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x, mask=None):
        ndim = x.dim()
        if ndim == 2:
            x = x.unsqueeze(1)

        assert (
            x.dim() == 3
        ), "Input tensor must have 2 or 3 dimensions (batch_size, channels, width)"

        if mask is None:
            mask = ~torch.isnan(x)

        assert x.shape == mask.shape, "Input tensor and mask must have the same shape"

        masked_x = torch.where(mask, x, torch.zeros_like(x))

        x = F.pad(masked_x, (self.padding, self.padding), mode="reflect")
        mask = F.pad(
            mask.float(), (self.padding, self.padding), mode="constant", value=0
        )

        x = x.unfold(2, self.kernel_size, self.stride)
        mask = mask.unfold(2, self.kernel_size, self.stride)

        x = x.contiguous().view(x.size()[:3] + (-1,))
        mask = mask.contiguous().view(mask.size()[:3] + (-1,)).to(x.device)

        # Combine the mask with the input tensor
        #x_masked = torch.where(mask.bool(), x, torch.fill_(torch.zeros_like(x),float("inf")))
        x_masked = torch.where(mask.bool(), x, torch.FloatTensor([float("inf")]).to(x.device))

        # Sort the masked tensor along the last dimension
        x_sorted, _ = torch.sort(x_masked, dim=-1)

        # Compute the count of non-masked (valid) values
        valid_count = mask.sum(dim=-1)

        # Calculate the index of the median value for each pooling window
        median_idx = (torch.div((valid_count - 1), 2, rounding_mode='trunc')).clamp(min=0)

        # Gather the median values using the calculated indices
        median_pooled = x_sorted.gather(-1, median_idx.unsqueeze(-1).long()).squeeze(-1)

        # Fill infinite values with NaNs
        median_pooled[torch.isinf(median_pooled)] = float("nan")
        
        if ndim == 2:
            return median_pooled.squeeze(1)

        return median_pooled


class CrepePitchExtractor(BasePitchExtractor):
    def __init__(
        self,
        hop_length: int = 512,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        threshold: float = 0.05,
        keep_zeros: bool = False,
        device = None,
        model: Literal["full", "tiny"] = "full",
        use_fast_filters: bool = True,
    ):
        super().__init__(hop_length, f0_min, f0_max, keep_zeros)

        self.threshold = threshold
        self.model = model
        self.use_fast_filters = use_fast_filters
        self.hop_length = hop_length
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        if self.use_fast_filters:
            self.median_filter = MaskedMedianPool1d(3, 1, 1).to(device)
            self.mean_filter = MaskedAvgPool1d(3, 1, 1).to(device)

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        """Extract pitch using crepe.


        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).
        """

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."
        assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."

        x = x.to(self.dev)
        f0, pd = torchcrepe.predict(
            x,
            sampling_rate,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            pad=True,
            model=self.model,
            batch_size=1024,
            device=x.device,
            return_periodicity=True,
        )

        # Filter, remove silence, set uv threshold, refer to the original warehouse readme
        if self.use_fast_filters:
            pd = self.median_filter(pd)
        else:
            pd = torchcrepe.filter.median(pd, 3)

        pd = torchcrepe.threshold.Silence(-60.0)(pd, x, sampling_rate, 512)
        f0 = torchcrepe.threshold.At(self.threshold)(f0, pd)
        
        if self.use_fast_filters:
            f0 = self.mean_filter(f0)
        else:
            f0 = torchcrepe.filter.mean(f0, 3)

        f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)[0]

        return self.post_process(x, sampling_rate, f0, pad_to)
