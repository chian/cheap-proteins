import os
import sys
import torch
import numpy as np
import subprocess

# Set device once and use everywhere
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Parameters
input_csv = "test_sequence_pairs.csv"
embedding_file = "test_embedding_pairs.pt"
embedding_dim = 64  # Should match the CHEAP encoder used
fixed_sequence_length = 256  # Should match sequence_to_embedding_pairs.py

# 1. Encode sequence pairs to embeddings
print("Encoding sequence pairs to embeddings...")
subprocess.run([
    sys.executable, "sequence_to_embedding_pairs.py",
    input_csv, embedding_file, str(embedding_dim)
], check=True)

# 2. Train the GRU on the embedding pairs
print("Training GRU on embedding pairs...")
from gru_predictor import train_gru
from gru_predictor import compute_saliency_map, plot_saliency_map

# Train for a small number of epochs for testing
model = train_gru(
    data_file=embedding_file,
    embedding_dim=embedding_dim,
    batch_size=2,
    num_epochs=3,
    lr=1e-3,
    device=device
)

print("Test pipeline complete.")

# 3. Decode and print predicted and true sequences for the first test pair
print("\nDecoding predicted and true sequences for the first test pair:")
from cheap_encoder_decoder import CheapEncoderDecoder

# Load embeddings and original sequences
data = torch.load(embedding_file, weights_only=False)
emb_A = torch.tensor(data['emb_A']).to(device)
emb_B = torch.tensor(data['emb_B']).to(device)

# Predict embedding_B from embedding_A using the trained model
model.eval()
with torch.no_grad():
    pred_B = model(emb_A[:1])  # [1, seq_len, embedding_dim]

# Initialize decoder on the same device
ced = CheapEncoderDecoder(device=device)
mask = torch.ones((1, fixed_sequence_length), dtype=torch.bool, device=device)

# CheapEncoderDecoder.decode expects [batch, seq_len, embedding_dim] for a batch
pred_seq = ced.decode(pred_B[:1], mask)
true_seq = ced.decode(emb_B[:1], mask)

print(f"Predicted sequence:\n{pred_seq[0] if isinstance(pred_seq, list) else pred_seq}")
print(f"True sequence:\n{true_seq[0] if isinstance(true_seq, list) else true_seq}")

# Compute and plot saliency map for the first test pair
saliency_map = compute_saliency_map(model, emb_A[:1], device=device)
plot_saliency_map(saliency_map, title="Saliency Map for First Test Pair") 