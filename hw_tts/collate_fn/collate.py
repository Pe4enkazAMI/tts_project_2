import logging
from typing import List
from hw_tts.preproc import MelSpectrogram
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]) -> dict:
    audio = [elem['audio'].reshape(-1, 1) for elem in dataset_items]
    batch_audio = pad_sequence(audio, batch_first=True, padding_value=0).squeeze(-1)
    melspec_transform = MelSpectrogram()
    batch_spectrogram = melspec_transform(batch_audio)
    result_batch = {}
    result_batch['spectrogram'] = batch_spectrogram
    result_batch['real_audio'] = batch_audio.unsqueeze(1)
    return result_batch