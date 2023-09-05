import os
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=1
import numpy as np
import librosa
import numpy as np
import torch
import os
import torch
import torch.nn.functional as F
from torchvision.models import resnet152
import torch
import torch.nn as nn
from torch.nn import CosineSimilarity
import os
import librosa
import numpy as np
from scipy.ndimage.morphology import binary_dilation
import webrtcvad
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F


int16_max = (2 ** 15) - 1

class FeatureBank():
    def __init__(self, resnetckpt = None, max_audio_length=10, device = 'cuda', norm_volume=-10, ignore_silences=True):
        self.device = device
        self.cos = CosineSimilarity(dim = -1)
        if resnetckpt != None:
            self.resnetmodel = self.init_resnet152(resnetckpt)
        else:
            self.resnetmodel = None
        self.norm_volume = norm_volume
        self.max_audio_length = max_audio_length
        self.ignore_silences = ignore_silences
        
        self.silences_params = {
                "Network": {
                    "seed": 1
                },
                "nfft": 2048,
                "window": 0.025,
                "hop": 0.0125,
                "nmels": 128,
                "tisv_frame": 180,
                "preprocessing": {
                    "mel_window_length": 25,
                    "mel_window_step": 10,
                    "mel_n_channels": 40,
                    "partials_n_frames": 160,
                    "vad_window_length": 30,
                    "vad_moving_average_width": 8,
                    "vad_max_silence_length": 6,
                    "audio_norm_target_dBFS": -30
                }
            }
            

    def normalize_volume(self, wav, target_dBFS, increase_only=False, decrease_only=False):
        if increase_only and decrease_only:
                raise ValueError("Both increase only and decrease only are set")
        rms = np.sqrt(np.mean((wav * int16_max) ** 2))
        wave_dBFS = 20 * np.log10(rms / int16_max)
        dBFS_change = target_dBFS - wave_dBFS
        if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
                return wav
        return wav * (10 ** (dBFS_change / 20))
        
        
    def get_mfcc_db(self, file_path, offset=-1, duration=-1, random = 'False', sr=16000, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
        if offset != -1 and duration !=-1:
            wav,sr = librosa.load(file_path,sr=sr, offset=offset, duration=duration)
        else:
            wav,sr = librosa.load(file_path,sr=sr)
        wav = self.normalize_volume(wav, target_dBFS=self.norm_volume)
        if not self.ignore_silences:
            wav = self.trim_long_silences(wav, fs=sr)

        if wav.shape[0]<self.max_audio_length*sr:
                wav=np.pad(wav,int(np.ceil((10*sr-wav.shape[0])/2)),mode='reflect')
        else:
                wav=wav[:self.max_audio_length*sr]
        spec=librosa.feature.mfcc(wav, sr=sr, n_fft=n_fft,
                        hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
        spec_db=librosa.power_to_db(spec,top_db=top_db)
        return spec_db    
            
        
    def get_melspectrogram_db(self, file_path, offset = -1, duration = -1, random = 'False', sr=16000, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
        if offset != -1 and duration !=-1:
            wav,sr = librosa.load(file_path,sr=sr, offset=offset, duration=duration)
        else:
            wav,sr = librosa.load(file_path,sr=sr)
        wav = self.normalize_volume(wav, target_dBFS=self.norm_volume)
        if not self.ignore_silences:
            wav = self.trim_long_silences(wav, fs=sr)
        
        if wav.shape[0]<self.max_audio_length*sr:
                wav=np.pad(wav,int(np.ceil((10*sr-wav.shape[0])/2)),mode='reflect')
        else:
                wav=wav[:self.max_audio_length*sr]
        spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
                        hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
        spec_db=librosa.power_to_db(spec,top_db=top_db)
        return spec_db

    def spec_to_image(self, spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled

    def init_resnet152(self, resnetckpt):
        device = self.device
        num_classes = 1061
        model = resnet152(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
        model.conv1 = nn.Sequential(nn.Dropout2d(0.2), nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        model= nn.Sequential(*list(model.children())[:-1])
        model = model.to(device)
        model.load_state_dict(torch.load(resnetckpt))
        model.eval()
        return model


    def get_embedding(self, audio_path, offset=-1, duration=-1, interpolate=False, interpolate_size=192):
        temp_ads = self.spec_to_image(self.get_melspectrogram_db(audio_path, offset, duration))[np.newaxis,...]
        input_tensor = self.resnetmodel(torch.tensor([temp_ads]).to(self.device, dtype=torch.float32)).view(1, 1 ,2048)
            
        if interpolate:
            output_tensor = F.interpolate(input_tensor, size=interpolate_size, mode='linear')
            return output_tensor
            
        return input_tensor         


    def verify_speakers(self, audio_path1, audio_path2, interpolate=False, interpolate_size=192):
        embs1 = self.get_embedding(audio_path1, interpolate, interpolate_size)
        embs2 = self.get_embedding(audio_path2, interpolate, interpolate_size)
        return (self.cos(embs1, embs2).item()+1)/2
    
    
    def trim_long_silences(self, wav, fs=16000):
        samples_per_window = (self.silences_params['preprocessing']['vad_window_length'] *  fs) // 1000
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                            sample_rate= fs))
        voice_flags = np.array(voice_flags)
        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width

        audio_mask = moving_average(voice_flags, self.silences_params['preprocessing']['vad_moving_average_width'])
        audio_mask = np.round(audio_mask).astype(np.bool)

        # Dilate the voiced regions
        audio_mask = binary_dilation(audio_mask, np.ones(self.silences_params['preprocessing']['vad_max_silence_length'] + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)
        return wav[audio_mask == True]
        
    
if __name__ == '__main__':
    resnetckpt='/home/ubuntu/SpeakerIdentification/ResNet4SpeakerIdentification/train/VLSPArcFace_CombineAugment_23_adetector.pt'
    ResNet152FeatureBank = FeatureBank(resnetckpt=resnetckpt, ignore_silences=True)
    path1 = '/home/ubuntu/SpeakerIdentification/iCOMM/son.nguyen/IMG_6367.wav'
    path2 = '/home/ubuntu/SpeakerIdentification/iCOMM/son.nguyen/IMG_6368.wav'
    print(ResNet152FeatureBank.verify_speakers(path1, path2))    
