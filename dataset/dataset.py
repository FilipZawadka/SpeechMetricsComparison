import os
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import librosa


class NISQA_Corpus_Dataset(Dataset):
    def __init__(self,
                csv_path="/work/data/speech_metrics_eval/NISQA_Corpus/NISQA_TRAIN_SIM/NISQA_TRAIN_SIM_file.csv",
                base_path="/work/data/speech_metrics_eval/NISQA_Corpus/"):
        self.entries = pd.read_csv(csv_path)
        self.base_path = base_path


    def audio_loading(self,path,sampling_rate=16000):

        audio, fs = librosa.load(path)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        if fs != sampling_rate:
            audio = librosa.resample(audio,fs,sampling_rate)

        return audio

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries.iloc[idx]
        deg_file = os.path.join(self.base_path, entry.filepath_deg)
        deg_audio = self.audio_loading(deg_file)

        return deg_audio, entry.mos