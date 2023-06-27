import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import repeat_expand_2d


def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args, whole_audio=False):
    data_train = AudioDataset(
        filelists = args.data.training_files,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        spk=args.spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        unit_interpolate_mode = args.data.unit_interpolate_mode,
        use_aug=True)
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )
    data_valid = AudioDataset(
        filelists = args.data.validation_files,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        spk=args.spk,
        extensions=args.data.extensions,
        unit_interpolate_mode = args.data.unit_interpolate_mode,
        n_spk=args.model.n_spk)
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return loader_train, loader_valid 


class AudioDataset(Dataset):
    def __init__(
        self,
        filelists,
        waveform_sec,
        hop_size,
        sample_rate,
        spk,
        load_all_data=True,
        whole_audio=False,
        extensions=['wav'],
        n_spk=1,
        device='cpu',
        fp16=False,
        use_aug=False,
        unit_interpolate_mode = 'left'
    ):
        super().__init__()
        
        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.filelists = filelists
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer={}
        self.pitch_aug_dict = {}
        self.unit_interpolate_mode = unit_interpolate_mode
        # np.load(os.path.join(self.path_root, 'pitch_aug_dict.npy'), allow_pickle=True).item()
        if load_all_data:
            print('Load all the data filelists:', filelists)
        else:
            print('Load the f0, volume data filelists:', filelists)
        with open(filelists,"r") as f:
            self.paths = f.read().splitlines()
        for name_ext in tqdm(self.paths, total=len(self.paths)):
            path_audio = name_ext
            duration = librosa.get_duration(filename = path_audio, sr = self.sample_rate)
            
            path_f0 = name_ext + ".f0.npy"
            f0,_ = np.load(path_f0,allow_pickle=True)
            f0 = torch.from_numpy(np.array(f0,dtype=float)).float().unsqueeze(-1).to(device)
                
            path_volume = name_ext + ".vol.npy"
            volume = np.load(path_volume)
            volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)
            
            path_augvol = name_ext + ".aug_vol.npy"
            aug_vol = np.load(path_augvol)
            aug_vol = torch.from_numpy(aug_vol).float().unsqueeze(-1).to(device)
                        
            if n_spk is not None and n_spk > 1:
                spk_name = name_ext.split("/")[-2]
                spk_id = spk[spk_name] if spk_name in spk else 0
                if spk_id < 0 or spk_id >= n_spk:
                    raise ValueError(' [x] Muiti-speaker traing error : spk_id must be a positive integer from 0 to n_spk-1 ')
            else:
                spk_id = 0
            spk_id = torch.LongTensor(np.array([spk_id])).to(device)

            if load_all_data:
                '''
                audio, sr = librosa.load(path_audio, sr=self.sample_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio = torch.from_numpy(audio).to(device)
                '''
                path_mel = name_ext + ".mel.npy"
                mel = np.load(path_mel)
                mel = torch.from_numpy(mel).to(device)
                
                path_augmel = name_ext + ".aug_mel.npy"
                aug_mel,keyshift = np.load(path_augmel, allow_pickle=True)
                aug_mel = np.array(aug_mel,dtype=float)
                aug_mel = torch.from_numpy(aug_mel).to(device)
                self.pitch_aug_dict[name_ext] = keyshift

                path_units = name_ext + ".soft.pt"
                units = torch.load(path_units).to(device)
                units = units[0]  
                units = repeat_expand_2d(units,f0.size(0),unit_interpolate_mode).transpose(0,1)
                
                if fp16:
                    mel = mel.half()
                    aug_mel = aug_mel.half()
                    units = units.half()
                    
                self.data_buffer[name_ext] = {
                        'duration': duration,
                        'mel': mel,
                        'aug_mel': aug_mel,
                        'units': units,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id
                        }
            else:
                path_augmel = name_ext + ".aug_mel.npy"               
                aug_mel,keyshift = np.load(path_augmel, allow_pickle=True)
                self.pitch_aug_dict[name_ext] = keyshift
                self.data_buffer[name_ext] = {
                        'duration': duration,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id
                        }
           

    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]
        # check duration. if too short, then skip
        if data_buffer['duration'] < (self.waveform_sec + 0.1):
            return self.__getitem__( (file_idx + 1) % len(self.paths))
            
        # get item
        return self.get_data(name_ext, data_buffer)

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer['duration']
        waveform_sec = duration if self.whole_audio else self.waveform_sec
        
        # load audio
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)
        aug_flag = random.choice([True, False]) and self.use_aug
        '''
        audio = data_buffer.get('audio')
        if audio is None:
            path_audio = os.path.join(self.path_root, 'audio', name) + '.wav'
            audio, sr = librosa.load(
                    path_audio, 
                    sr = self.sample_rate, 
                    offset = start_frame * frame_resolution,
                    duration = waveform_sec)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            # clip audio into N seconds
            audio = audio[ : audio.shape[-1] // self.hop_size * self.hop_size]       
            audio = torch.from_numpy(audio).float()
        else:
            audio = audio[start_frame * self.hop_size : (start_frame + units_frame_len) * self.hop_size]
        '''
        # load mel
        mel_key = 'aug_mel' if aug_flag else 'mel'
        mel = data_buffer.get(mel_key)
        if mel is None:
            mel = name_ext + ".mel.npy"
            mel = np.load(mel)
            mel = mel[start_frame : start_frame + units_frame_len]
            mel = torch.from_numpy(mel).float() 
        else:
            mel = mel[start_frame : start_frame + units_frame_len]
            
        # load f0
        f0 = data_buffer.get('f0')
        aug_shift = 0
        if aug_flag:
            aug_shift = self.pitch_aug_dict[name_ext]
        f0_frames = 2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        
        # load units
        units = data_buffer.get('units')
        if units is None:
            path_units = name_ext + ".soft.pt"
            units = torch.load(path_units)
            units = units[0]  
            units = repeat_expand_2d(units,f0.size(0),self.unit_interpolate_mode).transpose(0,1)
            
        units = units[start_frame : start_frame + units_frame_len]

        # load volume
        vol_key = 'aug_vol' if aug_flag else 'volume'
        volume = data_buffer.get(vol_key)
        volume_frames = volume[start_frame : start_frame + units_frame_len]
        
        # load spk_id
        spk_id = data_buffer.get('spk_id')
        
        # load shift
        aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()
        
        return dict(mel=mel, f0=f0_frames, volume=volume_frames, units=units, spk_id=spk_id, aug_shift=aug_shift, name=name, name_ext=name_ext)

    def __len__(self):
        return len(self.paths)