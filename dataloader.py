import os
import glob
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=1
import numpy as np
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import torchaudio

int16_max = 32767
def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def normalize_volume_torch(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = torch.sqrt(torch.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * torch.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))
# def get_melspectrogram_db(file_path, random = 'False', sr=16000, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80, volume_normalization = True):
#     wav,sr = librosa.load(file_path,sr=sr, duration=5)
#     if volume_normalization:
#         wav = normalize_volume(wav, target_dBFS=-10)
#     if wav.shape[0]<10*sr:
#       wav=np.pad(wav,int(np.ceil((10*sr-wav.shape[0])/2)),mode='reflect')
#     else:
#       wav=wav[:10*sr]
#     spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
#                 hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
#     spec_db=librosa.power_to_db(spec,top_db=top_db)
#     return spec_db
def get_melspectrogram_db(file_path,
                        sr=16000,
                        n_fft=2048,
                        hop_length=512,
                        n_mels=128,
                        fmin=20, fmax=8300, top_db=80, volume_normalization = True):
    wav,sr = librosa.load(file_path,sr=sr, duration=5)
    if volume_normalization:
        wav = normalize_volume(wav, target_dBFS=-10)
    if wav.shape[0]<10*sr:
      wav=np.pad(wav,int(np.ceil((10*sr-wav.shape[0])/2)),mode='reflect')
    else:
      wav=wav[:10*sr]
    spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
                hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    return spec_db


def get_mfcc_db(file_path, random = 'False', sr=16000, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    wav,sr = librosa.load(file_path,sr=sr)
    wav = normalize_volume(wav, target_dBFS=-10)

    if wav.shape[0]<10*sr:
      wav=np.pad(wav,int(np.ceil((10*sr-wav.shape[0])/2)),mode='reflect')
    else:
      wav=wav[:10*sr]
    spec=librosa.feature.mfcc(wav, sr=sr, n_fft=n_fft,
                hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    return spec_db

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


class AudioDataset(Dataset):
    def __init__(self,
                 root,
                 batch_size=32,
                 num_workers=2,
                 classnames=None,
                 total_classes = None,
                 audio_duration = 10,
                 augment_volume = True,
                 augment_noise  = True,
                 is_train = True):
        """
            Audio dataloader
        """
        super(AudioDataset, self).__init__()
        self.data = open(root).read().strip('\n').split('\n')
        self.audio_duration = audio_duration
        self.augment_volume = augment_volume
        self.augment_volume_range = [-40, 40]
        self.augment_noise = augment_noise
        self.mixup = 0.0
        self.total_classes = total_classes
        self.freqm = 10
        self.timem = 10
        self.skip_norm = True
        self.noise = True
        self.norm_mean = None
        self.norm_std = None
        self.melbins = 128
        self.target_length = 1024 # ~ s
        self.is_train = is_train
    
    def __len__(self):
        return len(self.data)

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        
        # Extract fbank
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        # librosa_audio, librosa_sr = librosa.load(filename, sr = None)
        # print(librosa_audio.shape[0]/16000, fbank.shape, librosa_sr)
        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup and self.is_train:
            datum = self.data[index]
            datum_wav, datum_labels = datum.split(",")
            datum_labels = np.array(int(datum_labels), dtype = np.int16)
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum_wav, mix_datum_labels = mix_datum.split(",")
            mix_datum_labels = np.array(int(mix_datum_labels), dtype = np.int16)
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(datum_wav, mix_datum_wav)
            # # initialize the label
            # label_indices = np.zeros(self.total_classes)
            # # add sample 1 labels
            # label_indices[datum_labels] += mix_lambda
            # # add sample 2 labels
            # label_indices[mix_datum_labels] += 1.0-mix_lambda
            # label_indices = torch.FloatTensor(label_indices)
            label_indices = torch.from_numpy(datum_labels)
        # if not do mixup
        else:
            datum = self.data[index]
            datum_wav, datum_labels = datum.split(",")
            datum_labels = np.array(int(datum_labels), dtype = np.int16)
            fbank, mix_lambda = self._wav2fbank(datum_wav)
            # label_indices = np.zeros(self.label_num)
            # label_indices[datum_labels] = 1.0
            # label_indices = torch.FloatTensor(label_indices)
            label_indices = torch.from_numpy(datum_labels)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0 and self.is_train:
            fbank = freqm(fbank)
        if self.timem != 0 and self.is_train:
            fbank = timem(fbank)
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise and self.is_train:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # print(fbank.shape, label_indices)
        # mix_ratio = min(mix_lambda, 1-mix_lambda) / max(mix_lambda, 1-mix_lambda)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128] -> unsqueeze to [1, 1024, 128]
        return fbank.unsqueeze(0), label_indices.to(torch.int64)

