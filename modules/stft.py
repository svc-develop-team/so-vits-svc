from librosa.util import pad_center, tiny
from scipy.signal import get_window
from torch import Tensor
from torch.autograd import Variable
from typing import Optional, Tuple

import librosa
import librosa.util as librosa_util
import math
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import warnings


def create_fb_matrix(
        n_freqs: int,
        f_min: float,
        f_max: float,
        n_mels: int,
        sample_rate: int,
        norm: Optional[str] = None
) -> Tensor:
    r"""Create a frequency bin conversion matrix.

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        norm (Optional[str]): If 'slaney', divide the triangular mel weights by the width of the mel band
        (area normalization). (Default: ``None``)

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * create_fb_matrix(A.size(-1), ...)``.
    """

    if norm is not None and norm != "slaney":
        raise ValueError("norm must be one of None or 'slaney'")

    # freq bins
    # Equivalent filterbank construction by Librosa
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
    m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    # create overlapping triangles
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.min(down_slopes, up_slopes)
    fb = torch.clamp(fb, 1e-6, 1)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)
    return fb


def lfilter(
        waveform: Tensor,
        a_coeffs: Tensor,
        b_coeffs: Tensor,
        clamp: bool = True,
) -> Tensor:
    r"""Perform an IIR filter by evaluating difference equation.

    Args:
        waveform (Tensor): audio waveform of dimension of ``(..., time)``.  Must be normalized to -1 to 1.
        a_coeffs (Tensor): denominator coefficients of difference equation of dimension of ``(n_order + 1)``.
                                Lower delays coefficients are first, e.g. ``[a0, a1, a2, ...]``.
                                Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (Tensor): numerator coefficients of difference equation of dimension of ``(n_order + 1)``.
                                 Lower delays coefficients are first, e.g. ``[b0, b1, b2, ...]``.
                                 Must be same size as a_coeffs (pad with 0's as necessary).
        clamp (bool, optional): If ``True``, clamp the output signal to be in the range [-1, 1] (Default: ``True``)

    Returns:
        Tensor: Waveform with dimension of ``(..., time)``.
    """
    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    assert (a_coeffs.size(0) == b_coeffs.size(0))
    assert (len(waveform.size()) == 2)
    assert (waveform.device == a_coeffs.device)
    assert (b_coeffs.device == a_coeffs.device)

    device = waveform.device
    dtype = waveform.dtype
    n_channel, n_sample = waveform.size()
    n_order = a_coeffs.size(0)
    n_sample_padded = n_sample + n_order - 1
    assert (n_order > 0)

    # Pad the input and create output
    padded_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)
    padded_waveform[:, (n_order - 1):] = waveform
    padded_output_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)

    # Set up the coefficients matrix
    # Flip coefficients' order
    a_coeffs_flipped = a_coeffs.flip(0)
    b_coeffs_flipped = b_coeffs.flip(0)

    # calculate windowed_input_signal in parallel
    # create indices of original with shape (n_channel, n_order, n_sample)
    window_idxs = torch.arange(n_sample, device=device).unsqueeze(0) + torch.arange(n_order, device=device).unsqueeze(1)
    window_idxs = window_idxs.repeat(n_channel, 1, 1)
    window_idxs += (torch.arange(n_channel, device=device).unsqueeze(-1).unsqueeze(-1) * n_sample_padded)
    window_idxs = window_idxs.long()
    # (n_order, ) matmul (n_channel, n_order, n_sample) -> (n_channel, n_sample)
    input_signal_windows = torch.matmul(b_coeffs_flipped, torch.take(padded_waveform, window_idxs))

    input_signal_windows.div_(a_coeffs[0])
    a_coeffs_flipped.div_(a_coeffs[0])
    for i_sample, o0 in enumerate(input_signal_windows.t()):
        windowed_output_signal = padded_output_waveform[:, i_sample:(i_sample + n_order)]
        o0.addmv_(windowed_output_signal, a_coeffs_flipped, alpha=-1)
        padded_output_waveform[:, i_sample + n_order - 1] = o0

    output = padded_output_waveform[:, (n_order - 1):]

    if clamp:
        output = torch.clamp(output, min=-1., max=1.)

    # unpack batch
    output = output.reshape(shape[:-1] + output.shape[-1:])

    return output



def biquad(
        waveform: Tensor,
        b0: float,
        b1: float,
        b2: float,
        a0: float,
        a1: float,
        a2: float
) -> Tensor:
    r"""Perform a biquad filter of input tensor.  Initial conditions set to 0.
    https://en.wikipedia.org/wiki/Digital_biquad_filter

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        b0 (float): numerator coefficient of current input, x[n]
        b1 (float): numerator coefficient of input one time step ago x[n-1]
        b2 (float): numerator coefficient of input two time steps ago x[n-2]
        a0 (float): denominator coefficient of current output y[n], typically 1
        a1 (float): denominator coefficient of current output y[n-1]
        a2 (float): denominator coefficient of current output y[n-2]

    Returns:
        Tensor: Waveform with dimension of `(..., time)`
    """

    device = waveform.device
    dtype = waveform.dtype

    output_waveform = lfilter(
        waveform,
        torch.tensor([a0, a1, a2], dtype=dtype, device=device),
        torch.tensor([b0, b1, b2], dtype=dtype, device=device)
    )
    return output_waveform



def _dB2Linear(x: float) -> float:
    return math.exp(x * math.log(10) / 20.0)


def highpass_biquad(
        waveform: Tensor,
        sample_rate: int,
        cutoff_freq: float,
        Q: float = 0.707
) -> Tensor:
    r"""Design biquad highpass filter and perform filtering.  Similar to SoX implementation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        cutoff_freq (float): filter cutoff frequency
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)

    Returns:
        Tensor: Waveform dimension of `(..., time)`
    """
    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2. / Q

    b0 = (1 + math.cos(w0)) / 2
    b1 = -1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)



def lowpass_biquad(
        waveform: Tensor,
        sample_rate: int,
        cutoff_freq: float,
        Q: float = 0.707
) -> Tensor:
    r"""Design biquad lowpass filter and perform filtering.  Similar to SoX implementation.

    Args:
        waveform (torch.Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        cutoff_freq (float): filter cutoff frequency
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`
    """
    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q

    b0 = (1 - math.cos(w0)) / 2
    b1 = 1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


class MelScale(torch.nn.Module):
    r"""Turn a normal STFT into a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    User can control which device the filter bank (`fb`) is (e.g. fb.to(spec_f.device)).

    Args:
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. Calculated from first input
            if None is given.  See ``n_fft`` in :class:`Spectrogram`. (Default: ``None``)
    """
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self,
                 n_mels: int = 128,
                 sample_rate: int = 24000,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 n_stft: Optional[int] = None) -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min

        assert f_min <= self.f_max, 'Require f_min: %f < f_max: %f' % (f_min, self.f_max)

        fb = torch.empty(0) if n_stft is None else create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate)
        self.register_buffer('fb', fb)

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """

        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        if self.fb.numel() == 0:
            tmp_fb = create_fb_matrix(specgram.size(1), self.f_min, self.f_max, self.n_mels, self.sample_rate)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(specgram.transpose(1, 2), self.fb).transpose(1, 2)

        # unpack batch
        mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])

        return mel_specgram


class TorchSTFT(torch.nn.Module):
    def __init__(self, fft_size, hop_size, win_size,
                 normalized=False, domain='linear',
                 mel_scale=False, ref_level_db=20, min_level_db=-100):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
        self.window = torch.hann_window(win_size)
        self.normalized = normalized
        self.domain = domain
        self.mel_scale = MelScale(n_mels=(fft_size // 2 + 1),
            n_stft=(fft_size // 2 + 1)) if mel_scale else None
        
    def transform(self, x):
        x_stft = torch.stft(x.to(torch.float32), self.fft_size, self.hop_size, self.win_size,
                            self.window.type_as(x), normalized=self.normalized)
        real = x_stft[..., 0]
        imag = x_stft[..., 1]
        mag = torch.clamp(real ** 2 + imag ** 2, min=1e-7)
        mag = torch.sqrt(mag)
        phase = torch.atan2(imag, real)

        if self.mel_scale is not None:
            mag = self.mel_scale(mag)

        if self.domain == 'log':
            mag = 20 * torch.log10(mag) - self.ref_level_db
            mag = torch.clamp((mag - self.min_level_db) / -self.min_level_db, 0, 1)
            return mag, phase
        elif self.domain == 'linear':
            return mag, phase
        elif self.domain == 'double':
            log_mag = 20 * torch.log10(mag) - self.ref_level_db
            log_mag = torch.clamp((log_mag - self.min_level_db) / -self.min_level_db, 0, 1)
            return torch.cat((mag, log_mag), dim=1), phase

    def complex(self, x):
        x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_size,
                            self.window.type_as(x), normalized=self.normalized)
        real = x_stft[..., 0]
        imag = x_stft[..., 1]
        return real, imag



class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

