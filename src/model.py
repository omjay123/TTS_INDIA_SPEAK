import torch
import torch.nn as nn
import torch.nn.functional as F

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super().__init__()
        layers = []
        for size in sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            in_dim = size
        self.prenet = nn.Sequential(*layers)

    def forward(self, x):
        return self.prenet(x)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.embedding(x)
        outputs, _ = self.bilstm(x)
        return outputs 

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.query_layer = nn.Linear(decoder_dim, encoder_dim)
        self.energy = nn.Linear(encoder_dim, 1)

    def forward(self, query, encoder_outputs):
        query = self.query_layer(query).unsqueeze(1)
        energy = self.energy(torch.tanh(query + encoder_outputs))
        attn_weights = F.softmax(energy.squeeze(-1), dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, encoder_dim, mel_dim, prenet_sizes=[128, 64], decoder_dim=256):
        super().__init__()
        self.prenet = Prenet(mel_dim, prenet_sizes)
        self.lstm = nn.LSTM(prenet_sizes[-1] + encoder_dim, decoder_dim, batch_first=True)
        self.linear = nn.Linear(decoder_dim, mel_dim)
        self.attn = Attention(encoder_dim, decoder_dim)

    def forward(self, mel_input, encoder_outputs):
        B, T_mel, mel_dim = mel_input.size()
        mel_input = self.prenet(mel_input)
        outputs = []
        hidden = None

        for t in range(T_mel):
            x_t = mel_input[:, t, :]
            decoder_input = x_t.unsqueeze(1)

            query = hidden[0][-1] if hidden is not None else torch.zeros(B, self.lstm.hidden_size, device=mel_input.device)

            context, _ = self.attn(query, encoder_outputs)
            context = context.unsqueeze(1)

            lstm_input = torch.cat([decoder_input, context], dim=-1)
            out, hidden = self.lstm(lstm_input, hidden) 
            mel_out = self.linear(out.squeeze(1))
            outputs.append(mel_out)

        return torch.stack(outputs, dim=1)

class TacotronLite(nn.Module):
    def __init__(self, vocab_size, mel_dim=80, embed_dim=128, enc_hidden_dim=128, dec_hidden_dim=256):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, enc_hidden_dim)
        self.decoder = Decoder(enc_hidden_dim * 2, mel_dim, decoder_dim=dec_hidden_dim)

    def forward(self, text, mel_input):
        encoder_outputs = self.encoder(text)
        mel_out = self.decoder(mel_input, encoder_outputs)
        return mel_out
