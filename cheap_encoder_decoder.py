import torch
from cheap.pretrained import CHEAP_shorten_1_dim_64, get_pipeline
from cheap.proteins import LatentToSequence

class CheapEncoderDecoder:
    def __init__(self, device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        model = CHEAP_shorten_1_dim_64(return_pipeline=False)
        self.pipeline = get_pipeline(model, device=device)
        self.decoder = LatentToSequence().to(device)

    def encode(self, sequences):
        """
        Encode protein sequences to compressed embeddings.
        Args:
            sequences (str or list of str): Protein sequence(s).
        Returns:
            emb (torch.Tensor): Compressed embedding(s).
            mask (torch.Tensor): Mask(s) for valid positions.
        """
        emb, mask = self.pipeline(sequences)
        return emb, mask

    def decode(self, emb, mask=None):
        """
        Decode compressed embeddings back to protein sequences.
        Args:
            emb (torch.Tensor): Compressed embedding(s).
            mask (torch.Tensor, optional): Mask(s) for valid positions.
        Returns:
            sequence_str (list of str): Decoded protein sequence(s).
        """
        latent = self.pipeline.decode(emb, mask)
        _, _, sequence_str = self.decoder.to_sequence(latent)
        return sequence_str 