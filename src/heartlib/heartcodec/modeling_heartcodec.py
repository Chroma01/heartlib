"""Hugging Face interface for HeartCodec waveform encoding and decoding."""

from transformers.modeling_utils import PreTrainedModel

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
                "Load a complete HeartCodec checkpoint with encoder weights "
                "to tokenize waveforms."
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
