import os
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import librosa
import random
import numpy as np
from pesq import pesq


class NISQA_Corpus_Dataset(Dataset):
    def __init__(self,
                csv_path="/work/data/speech_metrics_eval/NISQA_Corpus/NISQA_TRAIN_SIM/NISQA_TRAIN_SIM_file_pesq_si_sdr.csv",
                base_path="/work/data/speech_metrics_eval/NISQA_Corpus/",
                sampling_rate=16000,
                clip_sec=3,
                use_precalculated_metrics = True):
        self.entries = pd.read_csv(csv_path)
        self.base_path = base_path
        self.sampling_rate = sampling_rate
        self.clip_sec = clip_sec
        self.use_precalculated_metrics = use_precalculated_metrics
        if self.use_precalculated_metrics:
            self.entries = self.entries[self.entries["pesq"].notna()]


    def audio_loading(self,path,sampling_rate=16000):

        audio, fs = librosa.load(path, sr=None)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        if fs != sampling_rate:
            audio = librosa.resample(audio,fs,sampling_rate)

        return audio

    def clip_audio(self,audio,clip_sec=3,sampling_rate=16000):
        window_size = clip_sec*sampling_rate
        start = random.randrange(0, len(audio)-window_size)
        return audio[start:start+window_size]

    def clip_2_audios(self,audio_1,audio_2,clip_sec=3,sampling_rate=16000):
        window_size = clip_sec*sampling_rate
        start = random.randrange(0, len(audio_1)-window_size)
        return audio_1[start:start+window_size],audio_2[start:start+window_size]

    def si_sdr(self,deg_audio,reference_audio):
        eps = np.finfo(deg_audio.dtype).eps
        reference = reference_audio.reshape(reference_audio.size, 1)
        estimate = deg_audio.reshape(deg_audio.size, 1)
        Rss = np.dot(reference.T, reference)

        a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

        e_true = a * reference
        e_res = estimate - e_true

        Sss = (e_true**2).sum()
        Snn = (e_res**2).sum()

        return 10 * np.log10((eps+ Sss)/(eps + Snn))


    def __len__(self,):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries.iloc[idx]
        deg_file = os.path.join(self.base_path, entry.filepath_deg)
        ref_file = os.path.join(self.base_path, entry.filepath_ref)
        deg_audio = self.audio_loading(deg_file,self.sampling_rate)
        ref_audio = self.audio_loading(ref_file,self.sampling_rate)
        cliped_deg_audio, cliped_ref_audio = self.clip_2_audios(deg_audio,ref_audio, self.clip_sec, self.sampling_rate)
        if self.use_precalculated_metrics:
            pesq_val, si_sdr_val = entry.pesq, entry.si_sdr
        else:
            pesq_val = pesq(self.sampling_rate, cliped_ref_audio, cliped_deg_audio, 'wb')
            si_sdr_val = self.si_sdr(cliped_deg_audio,cliped_ref_audio)


        return cliped_deg_audio, entry.mos, pesq_val, si_sdr_val