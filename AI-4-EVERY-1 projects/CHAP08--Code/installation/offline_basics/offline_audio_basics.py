from transformers import pipeline
import torch
 
model_path = ("../../Models/models--openai--whisper-medium/"
              "snapshots/18530d7c5ce1083f21426064b85fbd1e24bd1858")
 
device = "cuda" if torch.cuda.is_available() else "cpu"
 
 
pipe = pipeline("automatic-speech-recognition",
                model=model_path, chunk_length_s=30, device=device)
 
out = pipe("../../Resources/test.mp3")["text"]
print(out)