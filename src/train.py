import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader, TTSData
from model import TacotronLite
import os
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, num_epochs=500, save_dir="outputs/checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for text, text_lens, mel in dataloader:
            B, mel_dim, mel_T = mel.size()

            # Shift mel for teacher forcing
            mel_input = mel[:, :, :-1].transpose(1, 2).to(device) 
            mel_target = mel[:, :, 1:].transpose(1, 2).to(device)
            text = text.to(device)

            optimizer.zero_grad()
            mel_output = model(text, mel_input)

            # Pad target to match output length
            min_len = min(mel_output.size(1), mel_target.size(1))
            loss = criterion(mel_output[:, :min_len], mel_target[:, :min_len])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss:.4f}")

        # Save checkpoint every 50 epochs
        if epoch % 50 == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

            save_mel_output_plot(mel_output[0].detach().cpu().numpy(), f"outputs/mel_outputs/mel_epoch_{epoch}.png")

def save_mel_output_plot(mel, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.T, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar()
    plt.title("Mel Output")
    plt.savefig(path)
    plt.close()

if __name__ == "__main__":
    csv_path = "data/toy_tts_parallel_data.csv"
    dataloader = get_dataloader(csv_path, batch_size=4)

    dummy_dataset = TTSData(csv_path)
    vocab_size = len(dummy_dataset.vocab)

    model = TacotronLite(vocab_size).to(device)
    train(model, dataloader)
