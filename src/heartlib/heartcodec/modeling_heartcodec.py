"""Hugging Face interface for HeartCodec waveform encoding and decoding."""

from transformers.modeling_utils import PreTrainedModel
import json
from pathlib import Path
from copy import deepcopy

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .configuration_heartcodec import HeartCodecConfig
from .models.decoder import HeartCodecDecoder, _load_legacy_decoder_state_dict
from .models.encoder import HeartCodecEncoder


class HeartCodec(PreTrainedModel):
    config_class = HeartCodecConfig

    def __init__(self, config: HeartCodecConfig):
        super().__init__(config)
        self.decoder = HeartCodecDecoder(config)
        self.encoder = (
            HeartCodecEncoder(config.encoder_config)
            if config.encoder_config is not None else None
        )
        self.sample_rate = config.sample_rate
        self.register_load_state_dict_pre_hook(_load_legacy_decoder_state_dict)
        self.post_init()

    @property
    def flow_matching(self):
        """Compatibility alias for the released decoder interface."""
        return self.decoder.flow_matching

    @property
    def scalar_model(self):
        """Compatibility alias for the released decoder interface."""
        return self.decoder.scalar_model

    @classmethod
    def from_encoder_decoder_pretrained(
        cls, decoder_path, encoder_path, *, encoder_revision=None,
        revision=None, token=None, local_files_only=False, **kwargs
    ):
        """Load the released decoder and a separate encoder-only checkpoint.

        Paths may be local directories or Hugging Face model IDs. The encoder
        shares the decoder's RVQ; no second quantizer is created. Move the returned
        model to the desired device after loading. Use FP32 for reconstruction.
        """
        if kwargs.pop("output_loading_info", False):
            raise ValueError("This loader validates loading internally and returns a model.")
        model, info = cls.from_pretrained(
            decoder_path, revision=revision, token=token,
            local_files_only=local_files_only, output_loading_info=True, **kwargs
        )
        if any(info.get(key) for key in
               ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Decoder checkpoint did not load strictly: {info}")
        if model.encoder is not None:
            raise ValueError("decoder_path must point to a decoder-only checkpoint.")

        def resolve(filename):
            directory = Path(encoder_path)
            if directory.is_dir():
                path = directory / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
                return str(path)
            return hf_hub_download(
                str(encoder_path), filename, revision=encoder_revision,
                token=token, local_files_only=local_files_only,
            )

        with open(resolve("encoder_config.json"), encoding="utf-8") as stream:
            encoder_config = json.load(stream)
        state = load_file(resolve("encoder.safetensors"), device="cpu")
        if not state or any(not key.startswith("encoder.") for key in state):
            raise ValueError("Encoder checkpoint must contain only encoder.* tensors.")
        encoder = HeartCodecEncoder(encoder_config)
        encoder.load_state_dict(
            {key[len("encoder."):]: value for key, value in state.items()}, strict=True
        )
        decoder_parameter = next(model.decoder.parameters())
        model.encoder = encoder.to(device=decoder_parameter.device, dtype=decoder_parameter.dtype)
        model.config.encoder_config = deepcopy(encoder_config)
        return model.eval()

    @staticmethod
    def _fix_state_dict_key_on_load(key):
        # Existing packaged checkpoints stored the decoder at the model root.
        key, changed = PreTrainedModel._fix_state_dict_key_on_load(key)
        if key.startswith(("flow_matching.", "scalar_model.")):
            return "decoder." + key, True
        return key, changed

    def tokenize(self, waveform, sample_rate, *, batch_size=1):
        """Encode float audio ``[samples]`` or ``[channels, samples]`` into RVQ tokens.

        Mono and stereo inputs at any positive integer sample rate are accepted.
        Returns CPU int64 ``[num_quantizers, frames]`` at 12.5 Hz. Requires a complete
        checkpoint in evaluation mode; ``batch_size`` controls encoder chunks.
        """
        if self.encoder is None:
            raise RuntimeError(
                "This HeartCodec checkpoint only supports decoding. "
                "Use HeartCodec.from_encoder_decoder_pretrained() with "
                "HeartCodec-oss-encoder to tokenize waveforms."
            )
        return self.encoder.tokenize(
            waveform, sample_rate, self.decoder.quantizer, batch_size=batch_size
        )

    def detokenize(
        self,
        codes,
        duration=29.76,
        num_steps=10,
        disable_progress=False,
        guidance_scale=1.25,
    ):
        """Decode ``[num_quantizers, frames]`` tokens into a CPU stereo waveform."""
        return self.decoder.detokenize(
            codes,
            duration=duration,
            num_steps=num_steps,
            disable_progress=disable_progress,
            guidance_scale=guidance_scale,
        )
