# Orthogonality analysis

The CPC-big model and k-means checkpoints used in [Analyzing Speaker Information in Self-Supervised Models to Improve Zero-Resource Speech Processing](https://arxiv.org/abs/2108.00917).

## Basic Usage

```python
import torch, torchaudio
from sklearn.preprocessing import StandardScaler

# Load model checkpoints
cpc = torch.hub.load("bshall/cpc:main", "cpc").cuda()

# Load audio
wav, sr = torchaudio.load("path/to/wav")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

x = cpc.encode(wav).squeeze().cpu().numpy()  # Encode
x = StandardScaler().fit_transform(x)  # Speaker normalize
```

Note that the `encode` function is stateful (keeps the hidden state of the LSTM from previous calls).

## Encode an Audio Dataset

Clone the repo and use the `encode.py` script:

```
usage: encode.py [-h] in_dir out_dir

Encode an audio dataset using CPC-big (with speaker normalization and discretization).

positional arguments:
  in_dir      Path to the directory to encode.
  out_dir     Path to the output directory.

optional arguments:
  -h, --help  show this help message and exit
```
