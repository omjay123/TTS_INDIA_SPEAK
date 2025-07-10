import numpy as np
import torch
import torchaudio

def griffin_lim_vocoder(mel_spec, n_iter=50, sr=24000, n_fft=1024, hop_length=256, win_length=1024):
    mel_spec = torch.tensor(mel_spec).unsqueeze(0)
    mel_spec = torchaudio.functional.db_to_amplitude(mel_spec)
    
    mel_transform = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=80,
        sample_rate=sr
    )
    linear_spec = mel_transform(mel_spec)
    
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_iter=n_iter
    )
    waveform = griffin_lim(linear_spec)
    return waveform.squeeze().numpy()
