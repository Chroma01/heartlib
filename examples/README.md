# 🎤 Lyrics Transcription

Download checkpoint using any of the following command:
```
hf download --local_dir './ckpt/HeartTranscriptor-oss' 'HeartMuLa/HeartTranscriptor-oss' 
modelscope download --model 'HeartMuLa/HeartTranscriptor-oss' --local_dir './ckpt/HeartTranscriptor-oss'
```

```
python ./examples/run_lyrics_transcription.py --model_path=./ckpt
```

By default this command will load the generated music file at `./assets/output.mp3` and print the transcribed lyrics. Use `--music_path` to specify the path to the music file.

Note that our HeartTranscriptor is trained on separated vocal tracks. In this example usage part, we directly demonstrate on unseparated music tracks, which is purely for simplicity of illustration. We recommend using source separation tools like demucs to separate the tracks before transcribing lyrics to achieve better results.

# Music Reconstruction

[HeartCodec-full-oss](https://huggingface.co/HeartMuLa/HeartCodec-full-oss) includes the encoder and decoder. It converts mono or stereo audio into eight token streams at 12.5 Hz and reconstructs stereo audio at 48 kHz.

Follow the [environment setup instructions](../README.md), then run these commands from the repository root:

```bash
hf download --local-dir './ckpt/HeartCodec-full' 'HeartMuLa/HeartCodec-full-oss'
python ./examples/run_music_reconstruction.py \
  --model_path ./ckpt/HeartCodec-full \
  --input_path ./assets/reference.mp3 \
  --save_path ./assets/recon.mp3
```

The example uses the bundled reference audio at `./assets/reference.mp3`. Use `--input_path` to select your own audio file. The example uses CUDA and saves a 320 kbps MP3 by default; MP3 output requires `ffmpeg` on PATH. Use `--save_path ./assets/recon.wav` for 24-bit PCM WAV output, or `--device cpu` to run on CPU. Mono input is duplicated to stereo, and the reconstructed audio is trimmed to the input duration.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--model_path` | `./ckpt/HeartCodec-full` | Complete model directory. |
| `--input_path` | Required | Input mono or stereo audio file. |
| `--save_path` | `./assets/recon.mp3` | Output MP3 or WAV file. |
| `--device` | `cuda` | PyTorch device, such as `cuda:0` or `cpu`. |
| `--batch_size` | `1` | Audio chunks encoded together; larger values use more GPU memory. |
| `--num_steps` | `10` | Decoder flow-matching steps. |
| `--guidance_scale` | `1.25` | Decoder guidance strength. |
| `--seed` | `42` | Random seed for waveform decoding. |

The decoder samples audio conditioned on the tokens. The seed controls this sampling; reconstruction is not a bit-exact copy of the input.
