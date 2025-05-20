import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class EmbeddingPairDataset(Dataset):
    def __init__(self, file_path):
        # file_path should point to a .pt or .npy file containing a dict or tuple/list of (embedding_A, embedding_B)
        if file_path.endswith('.pt'):
            data = torch.load(file_path, weights_only=False)
            self.embedding_A = torch.tensor(data['emb_A']) if not isinstance(data['emb_A'], torch.Tensor) else data['emb_A']
            self.embedding_B = torch.tensor(data['emb_B']) if not isinstance(data['emb_B'], torch.Tensor) else data['emb_B']
        elif file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True)
            self.embedding_A = torch.tensor(data[0]) if not isinstance(data[0], torch.Tensor) else data[0]
            self.embedding_B = torch.tensor(data[1]) if not isinstance(data[1], torch.Tensor) else data[1]
        else:
            raise ValueError('Unsupported file type')
        assert self.embedding_A.shape[0] == self.embedding_B.shape[0]

    def __len__(self):
        return self.embedding_A.shape[0]

    def __getitem__(self, idx):
        return self.embedding_A[idx], self.embedding_B[idx]

class GRUPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        # x: (batch, seq_len, embedding_dim)
        out, _ = self.gru(x)  # out: (batch, seq_len, hidden_dim)
        out = self.fc(out)    # out: (batch, seq_len, embedding_dim)
        return out

def train_gru(
    data_file,
    embedding_dim,
    batch_size=32,
    num_epochs=10,
    lr=1e-3,
    device=None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = EmbeddingPairDataset(data_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GRUPredictor(embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for emb_A, emb_B in loader:
            # emb_A, emb_B: (batch, seq_len, embedding_dim)
            emb_A = emb_A.to(device)
            emb_B = emb_B.to(device)
            pred_B = model(emb_A)  # (batch, seq_len, embedding_dim)
            loss = criterion(pred_B, emb_B)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * emb_A.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")
    return model

def compute_saliency_map(model, input_embedding, target_residue_idx=None, device=None):
    """
    Computes a saliency map for a given input embedding and trained model.
    Args:
        model: Trained GRUPredictor model.
        input_embedding: torch.Tensor of shape (1, seq_len, embedding_dim)
        target_residue_idx: int or None. If int, computes saliency for that output residue index. If None, computes for all.
        device: torch.device
    Returns:
        saliency: torch.Tensor of shape (seq_len,) if target_residue_idx is int, else (seq_len, seq_len)
    """
    model.eval()
    input_embedding = input_embedding.clone().detach().to(device)
    input_embedding.requires_grad_(True)
    output = model(input_embedding)  # (1, seq_len, embedding_dim)
    if target_residue_idx is not None:
        # Compute saliency for a specific output residue
        target = output[0, target_residue_idx, :].sum()
        model.zero_grad()
        target.backward(retain_graph=True)
        saliency = input_embedding.grad.abs().sum(dim=-1).squeeze(0)  # (seq_len,)
        input_embedding.grad.zero_()
        return saliency.cpu().numpy()
    else:
        # Compute saliency for each output residue
        seq_len = output.shape[1]
        saliency_map = []
        for t in range(seq_len):
            model.zero_grad()
            if input_embedding.grad is not None:
                input_embedding.grad.zero_()
            target = output[0, t, :].sum()
            target.backward(retain_graph=True)
            saliency = input_embedding.grad.abs().sum(dim=-1).squeeze(0)  # (seq_len,)
            saliency_map.append(saliency.cpu().numpy())
        saliency_map = np.stack(saliency_map, axis=0)  # (seq_len, seq_len)
        return saliency_map

def plot_saliency_map(saliency_map, input_labels=None, output_labels=None, title="Saliency Map"):
    """
    Plots the saliency map as a heatmap.
    Args:
        saliency_map: np.ndarray of shape (seq_len,) or (seq_len, seq_len)
        input_labels: list of str, optional
        output_labels: list of str, optional
        title: str
    """
    plt.figure(figsize=(8, 6))
    if saliency_map.ndim == 1:
        plt.plot(saliency_map)
        plt.xlabel("Input Residue Index")
        plt.ylabel("Saliency")
    else:
        plt.imshow(saliency_map, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label="Saliency")
        plt.xlabel("Input Residue Index")
        plt.ylabel("Output Residue Index")
        if input_labels is not None:
            plt.xticks(ticks=np.arange(len(input_labels)), labels=input_labels, rotation=90)
        if output_labels is not None:
            plt.yticks(ticks=np.arange(len(output_labels)), labels=output_labels)
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage: python gru_predictor.py data.pt 64
    import sys
    if len(sys.argv) < 3:
        print("Usage: python gru_predictor.py <data_file.pt|npy> <embedding_dim>")
        sys.exit(1)
    data_file = sys.argv[1]
    embedding_dim = int(sys.argv[2])
    train_gru(data_file, embedding_dim) 