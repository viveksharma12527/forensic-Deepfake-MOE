import os
import yaml
import random
import librosa
import numpy as np
import torch
import torch.nn as nn


def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.01)


def read_yaml(config_path):
    """
    Read YAML file.

    :param config_path: path to the YAML config file.
    :type config_path: str
    :return: dictionary correspondent to YAML content
    :rtype dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def read_audio(audio_path, fs=16000, trim=False, int_type=False):
    x, fs = librosa.load(audio_path, sr=fs, mono=True)
    if trim:
        x = librosa.effects.trim(x, top_db=20)[0]
    if x.size == 0:
        raise ValueError(f"Audio is empty after loading/trimming: {audio_path}")
    if int_type:
        x = (x * 32768).astype(np.int32)
    return x, fs


def fix_length(x, max_len: int = 64600):
    """Adjust a waveform to the target length by repetition or truncation."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def seed_everything(seed: int):
    """
    Set seed for everything.
    :param seed: seed value
    :type seed: int
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class EpochMetrics:
    def __init__(self, device, num_losses):
        self.totals = torch.zeros(num_losses + 1, device=device, dtype=torch.float64)
        self.num_total = 0

    def update(self, logits, labels, *losses):
        batch_size = labels.numel()
        correct = (logits.detach().argmax(dim=1) == labels).sum()
        values = torch.stack([loss.detach() * batch_size for loss in losses] + [correct])
        self.totals.add_(values)
        self.num_total += batch_size

    def compute(self):
        values = (self.totals / self.num_total).tolist()
        return values[:-1], values[-1] * 100
