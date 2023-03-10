import numpy as np
from numpy import linalg as LA
import librosa
from scipy.io import wavfile
import soundfile as sf
import librosa.filters


def load_wav(wav_path, raw_sr, target_sr=16000, win_size=800, hop_size=200):
    audio = librosa.core.load(wav_path, sr=raw_sr)[0]
    if raw_sr != target_sr:
        audio = librosa.core.resample(audio,
                                      raw_sr,
                                      target_sr,
                                      res_type='kaiser_best')
        target_length = (audio.size // hop_size +
                         win_size // hop_size) * hop_size
        pad_len = (target_length - audio.size) // 2
        if audio.size % 2 == 0:
            audio = np.pad(audio, (pad_len, pad_len), mode='reflect')
        else:
            audio = np.pad(audio, (pad_len, pad_len + 1), mode='reflect')
    return audio


def save_wav(wav, path, sample_rate, norm=False):
    if norm:
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, sample_rate, wav.astype(np.int16))
    else:
        sf.write(path, wav, sample_rate)


_mel_basis = None
_inv_mel_basis = None


def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sampling_rate // 2
    return librosa.filters.mel(hparams.sampling_rate,
                               hparams.n_fft,
                               n_mels=hparams.acoustic_dim,
                               fmin=hparams.fmin,
                               fmax=hparams.fmax)


def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _stft(y, hparams):
    return librosa.stft(y=y,
                        n_fft=hparams.n_fft,
                        hop_length=hparams.hop_length,
                        win_length=hparams.win_size)


def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _normalize(S, hparams):
    return hparams.max_abs_value * np.clip(((S - hparams.min_db) /
                                         (-hparams.min_db)), 0, 1)

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _stft(y, hparams):
    return librosa.stft(y=y,
                        n_fft=hparams.n_fft,
                        hop_length=hparams.hop_length,
                        win_length=hparams.win_size)


def _istft(y, hparams):
    return librosa.istft(y,
                         hop_length=hparams.hop_length,
                         win_length=hparams.win_size)


def melspectrogram(wav, hparams):
    D = _stft(wav, hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams),
                   hparams) - hparams.ref_level_db
    return _normalize(S, hparams)


