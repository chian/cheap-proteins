import torch
import numpy as np
import pandas as pd
import sys
from cheap_encoder_decoder import CheapEncoderDecoder

# Set the fixed sequence length for all sequences
fixed_sequence_length = 256  # <-- MODIFY THIS VALUE AS NEEDED

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python sequence_to_embedding_pairs.py <input.csv|tsv> <output.pt|npy> <embedding_dim>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    embedding_dim = int(sys.argv[3])

    # Detect delimiter
    delimiter = ',' if input_file.endswith('.csv') else '\t'
    df = pd.read_csv(input_file, delimiter=delimiter)
    assert 'seqA' in df.columns and 'seqB' in df.columns, "Input file must have 'seqA' and 'seqB' columns."

    encoder_decoder = CheapEncoderDecoder()
    emb_A_list, emb_B_list = [], []

    def process_seq(seq):
        # Truncate or pad the sequence to fixed_sequence_length
        seq = seq[:fixed_sequence_length]
        if len(seq) < fixed_sequence_length:
            seq = seq + 'A' * (fixed_sequence_length - len(seq))  # Pad with 'A' (alanine)
        return seq

    for idx, row in df.iterrows():
        seqA = process_seq(row['seqA'])
        seqB = process_seq(row['seqB'])
        embA, _ = encoder_decoder.encode([seqA])  # [1, seq_len, emb_dim]
        embB, _ = encoder_decoder.encode([seqB])
        embA = embA.squeeze(0)  # [seq_len, emb_dim]
        embB = embB.squeeze(0)
        emb_A_list.append(embA.cpu().numpy())
        emb_B_list.append(embB.cpu().numpy())
        if (idx+1) % 100 == 0:
            print(f"Processed {idx+1} pairs...")

    emb_A_arr = np.stack(emb_A_list)
    emb_B_arr = np.stack(emb_B_list)

    if output_file.endswith('.pt'):
        torch.save({'emb_A': emb_A_arr, 'emb_B': emb_B_arr}, output_file)
    elif output_file.endswith('.npy'):
        np.save(output_file, (emb_A_arr, emb_B_arr))
    else:
        raise ValueError('Output file must be .pt or .npy')
    print(f"Saved {len(emb_A_arr)} pairs to {output_file}") 