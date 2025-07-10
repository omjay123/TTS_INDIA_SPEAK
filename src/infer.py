import torch
import torchaudio
import numpy as np
import os
from src.model import TacotronLite
from src.data_loader import TTSData
from src.utils import griffin_lim_vocoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer(text, model, vocab, mel_len=160, sampling_rate=24000):
    token_to_id = {ch: idx for idx, ch in enumerate(vocab)}
    text_ids = [token_to_id.get(c, token_to_id["<unk>"]) for c in text.strip()]
    text_tensor = torch.tensor([text_ids], dtype=torch.long).to(device)

    mel_input = torch.zeros(1, mel_len, 80).to(device)

    with torch.no_grad():
        mel_out = model(text_tensor, mel_input)  # [1, T, 80]

    mel = mel_out[0].cpu().numpy().T  # [80, T]
    wav = griffin_lim_vocoder(mel, n_iter=60, sr=sampling_rate)

    return wav, mel

def save_wav(wav, path, sampling_rate=24000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    waveform = torch.tensor(wav).unsqueeze(0)  # [1, T]
    torchaudio.save(path, waveform, sampling_rate)

if __name__ == "__main__":
    checkpoint_path = "outputs/checkpoints/model_epoch_500.pt"
    csv_path = "data/toy_tts_parallel_data.csv"
    sampling_rate = 24000

    # Load model
    dataset = TTSData(csv_path)
    vocab_size = len(dataset.vocab)
    model = TacotronLite(vocab_size).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    input_text = "நீங்கள் எப்படி இருக்கிறீர்கள்?" 
    wav, mel = infer(input_text, model, dataset.vocab, mel_len=160, sampling_rate=sampling_rate)

    save_wav(wav, "outputs/audio_output/namaste.wav", sampling_rate)
    print("Audio saved to: outputs/audio_output/namaste.wav")
