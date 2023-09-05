import os
import glob
from sklearn.utils import shuffle


lst_dataset = ['/home/ubuntu/dataset/speaker_recognition/vox1_converted',
                '/home/ubuntu/dataset/speaker_recognition/EVN+iComm',
                '/home/ubuntu/dataset/speaker_recognition/CN-Celeb-converted/train',
                '/home/ubuntu/dataset/speaker_recognition/VIVOS',
                '/home/ubuntu/SpeakerIdentification/VLSP2021/vietnamese-speaker-verification/data/data/train']
lst_audio_train = []
lst_audio_val = []
speaker_id = -1
total_sample = 0
for dataset in lst_dataset:
    lst_id = glob.glob(dataset + '/*')
    for spkf in lst_id:
        if 'VLSP' in dataset:
            lst_audio = glob.glob(spkf + '/*/*.wav')
        else:
            lst_audio = glob.glob(spkf + '/*.wav')
        total_audio = len(lst_audio)
        if total_audio < 10:
            continue
        speaker_id += 1
        total_sample  += total_audio
        lst_audio = shuffle(lst_audio)
        for ix, audio in enumerate(lst_audio):
            if ix < 5:
                lst_audio_val.append("{},{}".format(audio, speaker_id))
            else:
                lst_audio_train.append("{},{}".format(audio, speaker_id))

lst_audio_train = shuffle(lst_audio_train)
lst_audio_val = shuffle(lst_audio_val)

print(speaker_id, total_sample)
with open('lst_combine_train.txt', 'w+') as f:
    for line in lst_audio_train:
        f.write(line + '\n')


with open('lst_combine_val.txt', 'w+') as f:
    for line in lst_audio_val:
        f.write(line + '\n')