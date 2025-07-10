# 🇮🇳 IndiaSpeaks Tamil TTS – Proof of Concept

A lightweight **Text-to-Speech (TTS)** engine generating **natural Tamil speech** for **low-power mobile devices** using a Tacotron-style model trained on a toy dataset.

---

## Objectives

- **Sampling Rate**: 24 kHz  
- **Model Size**: ≤ 6 MB  
- **Latency**: ≤ 120 ms per sentence on Snapdragon 855  

---

## Folder Structure



TTS\_India\_Speak/
├── data/
│   └── toy\_tts\_parallel\_data.csv       # Toy parallel dataset
├── outputs/
│   ├── checkpoints/                    # Model checkpoints (e.g., model\_epoch\_500.pt)
│   ├── mel\_outputs/                    # Mel plots from training
│   ├── audio\_output/                   # Final WAV files
│   └── optimized\_model/                # ONNX optimized model
├── src/
│   ├── data\_loader.py                  # Dataset and DataLoader
│   ├── model.py                        # TacotronLite model
│   ├── train.py                        # Training loop
│   ├── infer.py                        # Inference pipeline
│   ├── utils.py                        # Griffin-Lim vocoder
│   ├── optimization\_model.py           # Full optimization pipeline
│   └── onnx\_export.py                  # ONNX export logic (called by optimization\_model)
├── README.md                           
```

---

## Setup Instructions

### 1. Create Environment & Install Dependencies

```bash
conda create -n tts_india_speak python=3.10
conda activate tts_india_speak

pip install -r requirements.txt
````

---

## Pipeline: Train → Infer → Optimize → Re-Infer

### Step 1️⃣: Train the Model

Train the TacotronLite model on the toy dataset.

```bash
python src/train.py
```

* Checkpoints saved to `outputs/checkpoints/`
* Mel spectrogram plots saved to `outputs/mel_outputs/`

---

### Step 2️⃣: Run Inference (Before Optimization)

Generate audio from a Tamil phrase (in Latin script) using the trained model.

```bash
python src/infer.py
```

Change the text input in `infer.py`:

```python
input_text = "நீங்கள் எப்படி இருக்கிறீர்கள்?"
```

* Output WAV saved to: `outputs/audio_output/tamil_audio.wav`

---

### Step 3️⃣: Optimize the Model (Quantization + ONNX Export)

This step uses `optimization_model.py`, which internally calls `onnx_export.py`.

```bash
python src/optimization_model.py
```

* Optimized model is saved to: `outputs/optimized_model/tts_model.onnx`

---

### Step 4️⃣: Run Inference with Optimized Model

Use ONNX Runtime or a separate script to load and run the optimized model. You can replace the `infer.py` pipeline with ONNX-based inference if targeting mobile or edge devices.


## Optimization Techniques Used

| Technique            | Purpose                      |
| -------------------- | ---------------------------- |
| Dynamic Quantization | Reduce model size + latency  |
| ONNX Export          | Make model portable          |
| TorchScript Ready    | Mobile deployment compatible |
| Dim Reduction        | Keep model under 6 MB        |

---

## Evaluation Metrics (Internal)

| Metric         | Description                        |
| -------------- | ---------------------------------- |
| MCD            | Mel-Cepstral Distortion            |
| MOS            | Mean Opinion Score (audio quality) |
| Inference Time | Must be ≤ 120 ms on Snapdragon 855 |

---

## Future Improvements

* Replace Griffin-Lim for better quality.
* Add multilingual and streaming support.
* Build Web UI using Streamlit or Flask.

---

## Maintained by

**Om Prakash -> Voice AI Team @ IndiaSpeaks**
Building lightweight multilingual speech systems for India.

---