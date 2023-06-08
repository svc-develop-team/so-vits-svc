import numpy as np


def normalize_wav(audio, max_value=32768., scale=0.8):
    '''Normalize a mono wav
    '''
    if audio.ndim != 1:
        raise ValueError('Audio must be mono')
    if audio.shape[0] < 10:
        raise ValueError('Audio is too short')

    volume = 0.8 * max_value
    max_wav_value = max(np.mean(np.sort(np.abs(audio))[-10:]), 1e-8)
    norm_audio = audio * volume / max_wav_value
    norm_audio = np.clip(norm_audio, -volume, volume)
    return norm_audio
