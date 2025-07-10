import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

class TTSData(Dataset):
    def __init__(self, csv_path, max_text_len=150):
        self.data = pd.read_csv(csv_path)
        self.vocab = self.build_vocab(self.data['normalized_text'])
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.max_text_len = max_text_len

    def build_vocab(self, texts):
        chars = set()
        for txt in texts:
            chars.update(txt.strip())
        vocab = [PAD_TOKEN, UNK_TOKEN] + sorted(list(chars))
        return vocab

    def __len__(self):
        return len(self.data)

    def tokenize(self, text):
        return [self.token_to_id.get(char, self.token_to_id[UNK_TOKEN]) for char in text.strip()]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['normalized_text']
        mel = json.loads(row['mel_spectrogram'])  # 80 x 160 = 12800 floats
        text_ids = self.tokenize(text)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).view(80, -1)  # [80, 160]
        return torch.tensor(text_ids, dtype=torch.long), mel_tensor

def collate_fn(batch):
    text_seqs, mel_specs = zip(*batch)
    text_lens = [len(seq) for seq in text_seqs]
    max_len = max(text_lens)

    padded_texts = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(text_seqs):
        padded_texts[i, :len(seq)] = seq

    mel_tensors = torch.stack(mel_specs)  # [B, 80, T]

    return padded_texts, torch.tensor(text_lens), mel_tensors

def get_dataloader(csv_path, batch_size=4, shuffle=True):
    dataset = TTSData(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
