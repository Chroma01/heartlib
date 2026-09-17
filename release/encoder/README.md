---
license: apache-2.0
tags:
- audio
- heartcodec
- encoder
---

# HeartCodec OSS Encoder

Encoder-only weights for HeartCodec audio tokenization. Pair this checkpoint
with [HeartCodec-oss-20260123](https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123),
which supplies the shared RVQ quantizer and audio decoder. Together they support
audio → eight token streams at 12.5 Hz → 48 kHz stereo reconstruction.

The matching code and reconstruction examples are in
[HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib).

You can now encode your own audio into tokens for fine-tuning HeartMuLa.
We hope this contribution will be useful to the music research community.

## Installation and reconstruction

Use Python 3.10, a CUDA-compatible PyTorch installation, and `ffmpeg` for MP3
output. Install `heartlib` in a separate environment.

```bash
git clone https://github.com/HeartMuLa/heartlib.git
cd heartlib
pip install -e .
hf download HeartMuLa/HeartCodec-oss-encoder --local-dir ./ckpt/HeartCodec-oss-encoder
hf download HeartMuLa/HeartCodec-oss-20260123 --local-dir ./ckpt/HeartCodec-oss-20260123
python examples/run_music_reconstruction.py \
  --encoder_path ./ckpt/HeartCodec-oss-encoder \
  --decoder_path ./ckpt/HeartCodec-oss-20260123 \
  --input_path ./assets/reference.mp3 \
  --save_path ./reconstructed.mp3
```

Replace `--input_path` to use your own authorized audio. MP3 is encoded directly from float32 output at
320 kbps, without an intermediate PCM WAV. WAV output is available by selecting
a `.wav` output path. Output is trimmed to the input duration.

## Python API

```python
import torch
from heartlib.heartcodec.modeling_heartcodec import HeartCodec

codec = HeartCodec.from_encoder_decoder_pretrained(
    "HeartMuLa/HeartCodec-oss-20260123",
    "HeartMuLa/HeartCodec-oss-encoder",
    dtype=torch.float32,
).to("cuda").eval()

# waveform: float32 tensor [samples], [1, samples], or [2, samples]
# sample_rate: positive integer sampling rate of the input waveform
# tokens = codec.tokenize(waveform, sample_rate, batch_size=1)
# reconstructed = codec.detokenize(tokens, num_steps=10, guidance_scale=1.25)
```

Do not pass this encoder-only repository to `HeartCodec.from_pretrained()`.
The existing decoder-only loading API remains supported. Encoder and decoder
weights are strictly checked by the split-checkpoint loader.

## Files

- `encoder.safetensors`: 1,006 encoder tensors, approximately 2.15 GB;
  original `encoder.*` names, shapes, dtypes and values are preserved.
- `encoder_config.json`: feature extractor and query encoder configuration.
- `SHA256SUMS`: weight-file SHA-256 checksum.
- `LICENSE`: the source checkpoint's Apache-2.0 license.

No decoder or RVQ weights are included here. The shared quantizer belongs to
the paired decoder checkpoint; loading two independently trained quantizers
is not supported.

## Validation

On an NVIDIA B300, the split encoder and published decoder completed a
239.293-second stereo audio round trip using FP32, batch size 1, seed 42,
10 decoder steps and guidance scale 1.25. Tokens had shape `[8, 2992]`.
Output was finite, non-silent, 48 kHz stereo and matched the input duration.
PyTorch peak allocated memory was approximately 12.65 GiB; peak reserved
memory was approximately 15.41 GiB. These are measurements for one input and
configuration, not minimum-memory requirements or an audio-quality benchmark.

All exported encoder tensor bytes match the source checkpoint. The paired
decoder's 818 tensors match the source decoder numerically; eight RVQ
initialization flags differ only in stored shape (scalar versus `[1]`).

## License

Apache-2.0, as declared by the source checkpoint. See `LICENSE`.
