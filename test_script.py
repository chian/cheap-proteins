import torch
from cheap.pretrained import CHEAP_shorten_1_dim_64, get_pipeline
from cheap.proteins import LatentToSequence

pipeline = CHEAP_shorten_1_dim_64(return_pipeline=False)  # get the model only
device = "mps" if torch.backends.mps.is_available() else "cpu"
pipeline = get_pipeline(pipeline, device=device)  # create pipeline on correct device

sequences = [
    "AYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAV",
    "VFGRCELAAAMRHGLDNYRGYSLGNWVCAAFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKIVSDGNGMNAWVAWRNRCGTDVQAWIRGCRL",
    "RTDCYGNVNRIDTTGASCKTAKPEGLSYCGVSASKKIAERDLQAMDRYKTIIKKVGEKLCVEPAVIAGIISRESHAGKVLKNGWGDRGNGFGLMQVDKRSHKPQGTWNGEVHITQGTTILINFIKTIQKKFPSWTKDQQLKGGISAYNAGAGNVRSYARMDIGTTHDDYANDVVARAQYYKQHGY",
]

emb, mask = pipeline(sequences)
print("Embedding shape:", emb.shape)
print("Mask shape:", mask.shape)

# Decode: get latent from compressed embedding, then sequence from latent
latent = pipeline.decode(emb, mask)
decoder = LatentToSequence()
decoder = decoder.to(device)
sequence_probs, sequence_idx, sequence_str = decoder.to_sequence(latent)

print("Decoded sequences:")
for s in sequence_str:
    print(s)