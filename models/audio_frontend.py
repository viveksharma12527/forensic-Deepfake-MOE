import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class PreEmphasis(nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x, self.flipped_filter).squeeze(1)


class AudioFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.fbank = nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,
                hop_length=160,
                f_min=20,
                f_max=7600,
                window_fn=torch.hamming_window,
                n_mels=80,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            x = self.fbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
        return x