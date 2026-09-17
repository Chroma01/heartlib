"""Encode a waveform into HeartCodec tokens, then decode it back to audio."""

import argparse
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf
import torch

from heartlib.heartcodec.modeling_heartcodec import HeartCodec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder_path", default="./ckpt/HeartCodec-oss-20260123",
                        help="Decoder checkpoint directory or Hugging Face model ID.")
    parser.add_argument("--encoder_path", default="./ckpt/HeartCodec-oss-encoder",
                        help="Encoder-only checkpoint directory or Hugging Face model ID.")
    parser.add_argument("--model_path", default=None,
                        help="Legacy complete checkpoint; cannot be combined with separate paths.")
    parser.add_argument("--input_path", required=True,
                        help="Input mono or stereo audio file.")
    parser.add_argument("--save_path", default="./assets/recon.mp3",
                        help="Output MP3 or WAV file.")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device, for example cuda:0 or cpu.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of audio chunks encoded together.")
    parser.add_argument("--num_steps", type=int, default=10,
                        help="Number of decoder flow-matching steps.")
    parser.add_argument("--guidance_scale", type=float, default=1.25,
                        help="Decoder classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for waveform decoding.")
    args = parser.parse_args()

    if args.batch_size < 1 or args.num_steps < 1:
        parser.error("--batch_size and --num_steps must be positive integers")
    if not np.isfinite(args.guidance_scale):
        parser.error("--guidance_scale must be finite")
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(f"Invalid --device: {error}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; install a CUDA-enabled PyTorch build or use --device cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            parser.error(f"CUDA device {device.index} is unavailable")

    save_path = Path(args.save_path)
    if save_path.suffix.lower() not in {".mp3", ".wav"}:
        parser.error("--save_path must end in .mp3 or .wav")
    if save_path.suffix.lower() == ".mp3" and shutil.which("ffmpeg") is None:
        parser.error("MP3 output requires ffmpeg on PATH")

    torch.manual_seed(args.seed)
    if args.model_path:
        import sys
        if any(arg.split('=')[0] in {'--encoder_path', '--decoder_path'} for arg in sys.argv[1:]):
            parser.error("--model_path cannot be combined with --encoder_path or --decoder_path")
        model = HeartCodec.from_pretrained(args.model_path, dtype=torch.float32)
    else:
        model = HeartCodec.from_encoder_decoder_pretrained(
            args.decoder_path, args.encoder_path, dtype=torch.float32
        )
    model = model.to(device).eval()
    audio, sample_rate = sf.read(args.input_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())

    tokens = model.tokenize(waveform, sample_rate, batch_size=args.batch_size)
    print(f"Encoded {len(audio) / sample_rate:.3f}s into {tuple(tokens.shape)} tokens.")
    reconstructed = model.detokenize(
        tokens, num_steps=args.num_steps, guidance_scale=args.guidance_scale
    )
    output_frames = round(len(audio) * model.sample_rate / sample_rate)
    if (reconstructed.ndim != 2 or reconstructed.shape[0] != 2
            or reconstructed.shape[1] < output_frames):
        raise RuntimeError(
            f"Expected stereo reconstruction with at least {output_frames} frames; "
            f"received shape {tuple(reconstructed.shape)}"
        )
    output = reconstructed[:, :output_frames].float().cpu().T.contiguous().numpy()
    if not np.isfinite(output).all():
        raise RuntimeError("HeartCodec returned non-finite audio samples")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.suffix.lower() == ".mp3":
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "f32le", "-ar", str(model.sample_rate),
                "-ac", str(output.shape[1]), "-i", "pipe:0",
                "-c:a", "libmp3lame", "-b:a", "320k", str(save_path),
            ],
            input=output.astype(np.dtype("<f4"), copy=False).tobytes(),
            check=True,
        )
    else:
        sf.write(save_path, output, model.sample_rate, subtype="PCM_24")
    print(f"Reconstructed audio saved to {save_path}")


if __name__ == "__main__":
    main()
