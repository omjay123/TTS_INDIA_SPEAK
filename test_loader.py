from src.data_loader import get_dataloader

dataloader = get_dataloader("data/toy_tts_parallel_data.csv")

for batch in dataloader:
    text, text_lens, mel = batch
    print("Text shape:", text.shape)
    print("Mel shape:", mel.shape)
    break