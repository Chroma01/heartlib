"""Feature extraction, query downsampling, and waveform tokenization."""

from copy import deepcopy
from numbers import Integral

import torch
from torch import nn

from .feature_extractors import HeartCodecFeatureExtractors, default_feature_extractor_config
from .query_transformer import HeartCodecQueryEncoder


def default_encoder_config():
    """Settings of the original ``config_context_sq_ft.yaml`` encoder."""
    return {
        "version": "vqv12",
        "chunk_samples": 714240,  # 29.76 seconds at 24 kHz
        "feature_extractors": default_feature_extractor_config(),
        "query": {"dim": 512, "num_layers": 6, "head_dim": 128, "interval": 2},
    }


class HeartCodecEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = deepcopy(config)
        if config.get("version") != "vqv12":
            raise ValueError("HeartCodec encoding supports the vqv12 configuration.")
        self.chunk_samples = config["chunk_samples"]
        if self.chunk_samples != 714240:
            raise ValueError("The vqv12 encoder uses 29.76-second (714240-sample) chunks.")
        self.feature_extractors = HeartCodecFeatureExtractors(config["feature_extractors"])
        self.query = HeartCodecQueryEncoder(**config["query"])
        self.requires_grad_(False)

    def forward(self, audio_24k):
        features = self.feature_extractors(audio_24k)
        parameter = next(self.query.parameters())
        features = tuple(feature.to(parameter.dtype) for feature in features)
        return self.query(*features)

    @torch.inference_mode()
    def tokenize(self, waveform, sample_rate, quantizer, *, batch_size=1):
        """Encode one waveform into CPU int64 tokens of shape ``[num_quantizers, frames]``.

        ``waveform`` is floating point, shaped ``[samples]`` or
        ``[channels, samples]`` (one or two channels). ``sample_rate`` is its
        rate in Hz. Mono is duplicated to stereo. Chunks are processed in
        batches without moving the entire recording to the model's device.
        Each token frame represents 80 ms; trim reconstructed padding to the
        original waveform duration when saving audio.

        ``quantizer`` is the decoder's shared RVQ, passed explicitly so its
        weights are registered only once. Requires ``eval()`` on both modules.
        """
        if self.training or quantizer.training:
            raise RuntimeError("Call HeartCodec.eval() before tokenizing audio.")
        if not isinstance(waveform, torch.Tensor) or not waveform.is_floating_point():
            raise TypeError("waveform must be a floating-point torch.Tensor.")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
            raise ValueError("waveform must have shape [samples], [1, samples], or [2, samples].")
        if waveform.shape[-1] == 0 or not torch.isfinite(waveform).all():
            raise ValueError("waveform must contain nonempty, finite audio samples.")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer in Hz.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        from torchaudio.functional import resample
        from torch.nn import functional as F

        waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
        if waveform.shape[0] == 1:
            waveform = waveform.expand(2, -1)
        if sample_rate != 24000:
            waveform = resample(waveform, sample_rate, 24000)

        # Match wrapper.sound2code, including its final partial-frame rule.
        output_frames = int(waveform.shape[-1] / 24000 * 12.5) + 1
        chunk_samples = self.chunk_samples
        chunk_count = (waveform.shape[-1] + chunk_samples - 1) // chunk_samples
        waveform = F.pad(waveform, (0, chunk_count * chunk_samples - waveform.shape[-1]))
        chunks = waveform.reshape(2, chunk_count, chunk_samples).permute(1, 0, 2)
        encoder_parameter = next(self.parameters())
        code_batches = []
        for start in range(0, chunk_count, batch_size):
            chunk_batch = F.pad(chunks[start : start + batch_size], (0, 240))
            chunk_batch = chunk_batch.to(device=encoder_parameter.device, dtype=torch.float32)
            features = self(chunk_batch)
            quantizer_parameter = next(quantizer.parameters())
            features = features.to(device=quantizer_parameter.device, dtype=quantizer_parameter.dtype)
            _, indices, _ = quantizer(features)
            code_batches.append(indices.reshape(-1, quantizer.num_quantizers).cpu())
        return torch.cat(code_batches, dim=0)[:output_frames].transpose(0, 1).contiguous().long()
