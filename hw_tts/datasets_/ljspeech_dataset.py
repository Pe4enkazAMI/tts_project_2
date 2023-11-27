import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path
import random
import torchaudio
ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent
import numpy as np
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", 
}


class LJspeechDataset():
    def __init__(self, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
            self._data_dir = data_dir
            self._load_dataset()
        self._data_dir = data_dir
        self.data = os.listdir(self._data_dir)
        self.max_audio_length = kwargs["max_audio_length"]
        self.data = self.data[:kwargs["limit"]]

    def _load_dataset(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = 22050
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor    


    def __getitem__(self, id):
        audio_path = self.data[id]
        audio_wave = self.load_audio(Path(self._data_dir) / Path(audio_path))
        if audio_wave.shape[-1] > self.max_audio_length:
            random_start = np.random.randint(0, audio_wave.shape[-1] - self.max_audio_length + 1)
            audio_wave = audio_wave[..., random_start:random_start+self.max_audio_length]

        return {
            "audio": audio_wave,
            "audio_length": audio_wave.shape[-1],
        }
    
    def __len__(self):
        return len(self.data)
