"""Pretrained feature extractors for the vqv12 HeartCodec encoder.

The MuEncoder frontend is adapted from MusicFM, Copyright 2023 ByteDance Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torchaudio
from transformers import (
    Wav2Vec2ConformerConfig,
    WavLMConfig,
    WavLMModel,
    WhisperConfig,
    WhisperFeatureExtractor,
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerEncoder,
)
from transformers.models.whisper.modeling_whisper import WhisperEncoder


def default_feature_extractor_config() -> Dict[str, Any]:
    """Return the fixed pretrained architectures used by config_context_sq_ft."""
    return {
        "whisper_config": {
            "d_model": 768,
            "encoder_layers": 12,
            "encoder_attention_heads": 12,
            "encoder_ffn_dim": 3072,
            "num_mel_bins": 80,
            "max_source_positions": 1500,
            "activation_function": "gelu",
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "encoder_layerdrop": 0.0,
            "scale_embedding": False,
        },
        "wavlm_config": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "feat_proj_dropout": 0.1,
            "layerdrop": 0.05,
            "layer_norm_eps": 1e-5,
            "feat_extract_norm": "group",
            "feat_extract_activation": "gelu",
            "conv_dim": [512] * 7,
            "conv_stride": [5, 2, 2, 2, 2, 2, 2],
            "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
            "conv_bias": False,
            "num_conv_pos_embeddings": 128,
            "num_conv_pos_embedding_groups": 16,
            "num_buckets": 320,
            "max_bucket_distance": 800,
            "do_stable_layer_norm": False,
            "add_adapter": False,
        },
        "muencoder_config": {
            "sample_rate": 24000,
            "n_fft": 2048,
            "hop_length": 240,
            "n_mels": 128,
            "conv_dim": 512,
            "mean": 6.768444971712967,
            "std": 18.417922652295623,
            "conformer_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "hidden_act": "swish",
                "position_embeddings_type": "rotary",
                "rotary_embedding_base": 10000,
                "max_source_positions": 5000,
                "layer_norm_eps": 1e-5,
                "conv_depthwise_kernel_size": 31,
                "num_conv_pos_embeddings": 128,
                "num_conv_pos_embedding_groups": 16,
                "hidden_dropout": 0.1,
                "attention_dropout": 0.1,
                "activation_dropout": 0.1,
                "conformer_conv_dropout": 0.1,
                "layerdrop": 0.0,
            },
        },
    }


def _merge_config(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class MelSTFT(nn.Module):
    """MusicFM's power mel spectrogram, preserving checkpoint buffer names."""

    def __init__(self, sample_rate=24000, n_fft=2048, hop_length=240, n_mels=128):
        super().__init__()
        # Torchaudio validates filter values during construction, which cannot
        # run on meta tensors used by from_pretrained's low-memory loading.
        with torch.device("cpu"):
            self.mel_stft = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform: Tensor) -> Tensor:
        # STFT and logarithms use float32 even when model weights are half precision.
        return self.amplitude_to_db(self.mel_stft.float()(waveform.float()))


class Res2dModule(nn.Module):
    def __init__(self, idim: int, odim: int, stride=(2, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(idim, odim, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(odim)
        self.conv2 = nn.Conv2d(odim, odim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(odim)
        self.relu = nn.ReLU()
        self.diff = idim != odim or stride[0] > 1
        if self.diff:
            self.conv3 = nn.Conv2d(idim, odim, 3, padding=1, stride=stride)
            self.bn3 = nn.BatchNorm2d(odim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        if self.diff:
            x = self.bn3(self.conv3(x))
        return self.relu(x + out)


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim: int, hdim: int, odim: int, n_bands: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            Res2dModule(idim, hdim, (2, 2)),
            Res2dModule(hdim, hdim, (2, 2)),
        )
        self.linear = nn.Linear(hdim * n_bands // 4, odim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x.unsqueeze(1))
        x = x.permute(0, 3, 1, 2).flatten(2)
        return self.linear(x)


class MuEncoder(nn.Module):
    """Inference subset of MusicFM25Hz; no fairseq or training heads required."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = _merge_config(
            default_feature_extractor_config()["muencoder_config"], config or {}
        )
        self.mean = config["mean"]
        self.std = config["std"]
        self.preprocessor_melspec_2048 = MelSTFT(
            **{key: config[key] for key in ("sample_rate", "n_fft", "hop_length", "n_mels")}
        )
        conformer_config = Wav2Vec2ConformerConfig(**config["conformer_config"])
        self.conv = Conv2dSubsampling(
            1, config["conv_dim"], conformer_config.hidden_size, config["n_mels"]
        )
        self.conformer = Wav2Vec2ConformerEncoder(conformer_config)

    def forward(self, waveform: Tensor) -> Tuple[Tensor, Tensor]:
        # The original fairseq wrapper discarded incomplete 25 Hz input frames.
        samples = waveform.shape[-1] // 960 * 960
        mel = self.preprocessor_melspec_2048(waveform[..., :samples])[..., :-1]
        mel = ((mel - self.mean) / self.std).to(self.conv.linear.weight.dtype)
        hidden_states = self.conformer(
            self.conv(mel), output_hidden_states=True, return_dict=True
        ).hidden_states
        return hidden_states[2], hidden_states[11]


class HeartCodecFeatureExtractors(nn.Module):
    """Extract aligned stereo acoustic, semantic, Whisper, and WavLM features.

    Input contains a multiple of 80 ms at 24 kHz followed by the original
    240-sample alignment tail. Outputs have shape ``[batch, frames_25hz, dim]``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = _merge_config(default_feature_extractor_config(), config or {})
        whisper_config = WhisperConfig(**config["whisper_config"])
        whisper_config._attn_implementation = "eager"
        self.whisper_encoder = WhisperEncoder(whisper_config)
        self.whisper_processor = WhisperFeatureExtractor(
            feature_size=whisper_config.num_mel_bins,
            sampling_rate=16000,
            hop_length=160,
            chunk_length=30,
            n_fft=400,
            padding_value=0.0,
            return_attention_mask=False,
        )
        self.wavlm_encoder = WavLMModel(WavLMConfig(**config["wavlm_config"]))
        self.muencoder = MuEncoder(config["muencoder_config"])
        # Match the original cached CPU-built filter. Building the filter on
        # CUDA introduces small differences that can change RVQ token indices.
        with torch.device("cpu"):
            self.resample_24k_to_16k = torchaudio.transforms.Resample(24000, 16000)
        self.resample_24k_to_16k.register_buffer(
            "kernel", self.resample_24k_to_16k.kernel, persistent=False
        )
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def _stereo_features(features: Tensor, batch_size: int) -> Tensor:
        _, frames, dim = features.shape
        return features.reshape(batch_size, 2, frames, dim).permute(0, 2, 1, 3).reshape(
            batch_size, frames, 2 * dim
        )

    @torch.no_grad()
    def forward(self, audio_24k: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if audio_24k.ndim != 3 or audio_24k.shape[1] != 2:
            raise ValueError("Feature extraction expects stereo audio shaped [batch, 2, samples].")
        content_samples = audio_24k.shape[-1] - 240
        if content_samples <= 0 or content_samples % 1920:
            raise ValueError("Each 24 kHz segment must contain a multiple of 1920 samples plus a 240-sample tail.")
        if audio_24k.shape[-1] > 30 * 24000:
            raise ValueError("Each feature extraction segment must fit in Whisper's 30-second window.")
        batch_size = audio_24k.shape[0]
        waveform = audio_24k.flatten(0, 1).float()
        acoustic, semantic = self.muencoder(waveform)

        waveform_16k = self.resample_24k_to_16k.float()(waveform)
        wavlm_states = self.wavlm_encoder(
            waveform_16k.to(self.wavlm_encoder.feature_extractor.conv_layers[0].conv.weight.dtype),
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states
        wavlm = torch.stack(wavlm_states[6:10], dim=1).mean(dim=1)

        mel = self.whisper_processor(
            waveform_16k.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            do_normalize=False,
        ).input_features
        whisper = self.whisper_encoder(
            mel.to(device=audio_24k.device, dtype=self.whisper_encoder.conv1.weight.dtype),
            return_dict=True,
        ).last_hidden_state
        frames_50hz = content_samples // 480
        whisper = whisper[:, :frames_50hz]
        if wavlm.shape[1] != frames_50hz:
            raise ValueError("WavLM frame count does not match the 24 kHz alignment contract.")
        whisper = whisper.reshape(batch_size * 2, -1, 2, whisper.shape[-1]).mean(dim=2)
        wavlm = wavlm.reshape(batch_size * 2, -1, 2, wavlm.shape[-1]).mean(dim=2)
        expected_frames = content_samples // 960
        features = (acoustic, semantic, whisper, wavlm)
        if any(feature.shape[1] != expected_frames for feature in features):
            raise ValueError("Pretrained feature streams have inconsistent frame counts.")
        return tuple(self._stereo_features(feature, batch_size) for feature in features)
