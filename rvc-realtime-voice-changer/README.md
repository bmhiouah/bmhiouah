# RVC Realtime Voice Changer (V2)

Minimal foundation for **real-time** Retrieval-based Voice Conversion (**RVC V2**) inference.

- **No training** — use ready-made V2 `.pth` (+ optional `.index`) models only
- **No WhatsApp** or messaging integrations
- Built to grow step-by-step: audio I/O first, then plug in real RVC infer

## Plan (step-by-step)

1. **Audio loop (done in skeleton)** — capture mic → optional process → play speakers via `sounddevice`
2. **Model path wiring** — point `MODEL_PATH` / config at a local V2 checkpoint (you download; not shipped here)
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

**Do not download weights into this repo.** Place files you obtain yourself under `models/` (gitignored).

Common sources:

| Source | Notes |
| --- | --- |
| [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) | Official-ish HF hub: pretrained assets, HuBERT, etc. |
| Community HF spaces / model cards | Search Hugging Face for `RVC V2` + character/speaker name; prefer V2 `.pth` |
| [voice-models.com](https://voice-models.com) | Catalog of shared community voice models |
| RVC Discord / model-sharing boards | Often link HF mirrors; verify license and consent |

You typically need:

- **Generator**: `*.pth` (RVC V2 checkpoint)
- **Optional**: `*.index` (FAISS feature index for quality; skippable for first wiring)
- **Shared**: HuBERT / content encoder weights (from the VoiceConversionWebUI hub or your RVC install)

## Weights placeholder

```
models/
  .gitkeep                 # keep folder in git
  YOUR_SPEAKER_v2.pth      # ← put your V2 model here (local only)
  YOUR_SPEAKER_v2.index    # ← optional
```

Set the path via env or edit `configs/default.yaml`:

```bash
export RVC_MODEL_PATH=models/YOUR_SPEAKER_v2.pth
python -m src.realtime_vc
```

Until a real file exists, the starter script runs **mic → speaker passthrough** and logs where inference would run.

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
    .gitkeep              # weights go here; not committed
  src/
    __init__.py
    realtime_vc.py        # mic → infer_rvc_v2() hook → speaker
```

## Status

- [x] Folder structure
- [x] Runnable realtime audio passthrough + `infer_rvc_v2()` TODO hook
- [ ] Load real RVC V2 + HuBERT and convert in-loop
- [ ] Optional FAISS index / pitch / formant controls
- [ ] Latency / quality tuning guide
