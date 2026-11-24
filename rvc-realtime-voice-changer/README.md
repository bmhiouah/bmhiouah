# RVC Realtime Voice Changer (V2)

Minimal foundation for **real-time** Retrieval-based Voice Conversion (**RVC V2**) inference.

- **No training** — use ready-made V2 `.pth` (+ optional `.index`) models only
- **No WhatsApp** or messaging integrations
- Built to grow step-by-step: audio I/O first, then plug in real RVC infer

## Plan (step-by-step)

1. **Audio loop (done in skeleton)** — capture mic → optional process → play speakers via `sounddevice`
2. **Model path wiring** — point `MODEL_PATH` / config at a local V2 checkpoint (sample EN feminine zips under `models/`)
3. **Inference hook** — replace `infer_rvc_v2()` passthrough with real HuBERT + RVC V2 synthesizer
4. **Low-latency tuning** — block size, sample rate (often 40k/48k), buffering, optional `.index` retrieval
5. **Later (out of scope now)** — GUI, VST, virtual cable routing, multi-speaker switching

## Stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.10+ |
| Audio I/O | `sounddevice` (+ PortAudio) |
| Arrays | `numpy` |
| Inference runtime | PyTorch (`torch`) |
| RVC / content encoder | RVC V2 weights + HuBERT (via `fairseq` or community forks; see notes in `requirements.txt`) |

Reference upstream: [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

Realtime-oriented peers for later ideas (not required here): [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

## Where to get ready-made RVC V2 models

**Seven free English feminine V2 zips ship under `models/`** (see [`models/INDEX.md`](models/INDEX.md) for exact filenames, byte sizes, and sources). Unzipped `.pth` / `.index` stay local (gitignored). Extra models: Hugging Face, [voice-models.com](https://voice-models.com), or [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) for HuBERT.

You typically need:

- **Generator**: `*.pth` (RVC V2 checkpoint)
- **Optional**: `*.index` (FAISS feature index for quality; skippable for first wiring)
- **Shared**: HuBERT / content encoder weights (from the VoiceConversionWebUI hub or your RVC install)

## Weights layout

```
models/
  INDEX.md                 # catalog: filenames + sizes
  GirlVoice.zip … Frida.zip  # Git LFS archives (each contains .pth [+ .index])
  _tmp/                    # local extract dir (ignored)
```

```bash
git lfs pull   # after clone
unzip models/FurinaEN.zip -d models/_tmp/
export RVC_MODEL_PATH=models/_tmp/furina-EN_v2_48k_e60/Furina_EN_e60_v2_48k.pth
python -m src.realtime_vc
```

Until a real checkpoint is loaded, the starter script runs **mic → speaker passthrough** and logs where inference would run.

## Quick start

```bash
cd rvc-realtime-voice-changer
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# List devices, then run passthrough loop
python -m src.realtime_vc --list-devices
python -m src.realtime_vc
```

Requires a working mic and speakers (or headphones). On Linux you may need system PortAudio (`libportaudio2`).

## Layout

```
rvc-realtime-voice-changer/
  README.md
  requirements.txt
  configs/
    default.yaml          # sample rate, block size, model path placeholders
  models/
    INDEX.md              # shipped zip catalog
    *.zip                 # free EN feminine V2 weights (LFS)
  src/
    __init__.py
    realtime_vc.py        # mic → infer_rvc_v2() hook → speaker
```

## Status

- [x] Folder structure
- [x] Runnable realtime audio passthrough + `infer_rvc_v2()` TODO hook
- [x] Free EN feminine V2 model zips + `models/INDEX.md` (no inference yet)
- [ ] Load real RVC V2 + HuBERT and convert in-loop
- [ ] Optional FAISS index / pitch / formant controls
- [ ] Latency / quality tuning guide
