# RVC Realtime Voice Changer (V2)

Real-time Retrieval-based Voice Conversion (**RVC V2**) for macOS / Linux:

- Mic → **ContentVec (HuBERT)** → **RVC V2 generator** → speakers
- Default voice: **Frida** (`models/Frida.zip` → `models/_tmp/frida/frida.pth`)
- **No training** — ready-made V2 `.pth` (+ optional `.index`) only
- **AirPods auto-select** for output when connected; input stays on default/built-in mic

## Quick start

```bash
cd rvc-realtime-voice-changer
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# One-shot: resolve Frida + ContentVec, load weights, synthetic infer (no mic needed)
python -m src.realtime_vc --dry-run --download-hubert

# List devices, then run (AirPods/EarPods out when present)
python -m src.realtime_vc --list-devices
python -m src.realtime_vc

# Browser dashboard (Start/Stop, voice + device pickers) — idle until Start
python -m src.web_dashboard
# → http://127.0.0.1:8765/
```

On first real run the script will:

1. Unzip `models/Frida.zip` → `models/_tmp/frida/frida.pth` if needed (`git lfs pull` first)
2. Download [lj1995 `hubert_base.pt`](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt) (ContentVec) into `models/hubert/` and convert it to Transformers format (no fairseq)
3. Open a duplex stream at the **model sample rate** (Frida = **40 kHz**)

## HuBERT / ContentVec download

RVC’s “HuBERT” file is **ContentVec** (500-class), published as:

| Item | Path / URL |
| --- | --- |
| Source | https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt |
| Local fairseq pt | `models/hubert/hubert_base.pt` |
| Converted (auto) | `models/hubert/contentvec_hf/` |

Manual download:

```bash
mkdir -p models/hubert
curl -L -o models/hubert/hubert_base.pt \
  "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
python -m src.realtime_vc --dry-run --download-hubert
```

## Audio devices

| Role | Policy |
| --- | --- |
| Output | Prefer a device whose name contains `AirPods` / `EarPods`; else system default |
| Input | Prefer built-in / MacBook mic; else system default input (not AirPods/EarPods mic) |
| Override | `--input-device N --output-device M`, `--list-devices`, or the web dashboard |

## Web dashboard

```bash
source .venv/bin/activate
python -m src.web_dashboard          # http://127.0.0.1:8765/  (foreground; PID /tmp/rvc-web-dashboard.pid)
# Force-kill leftovers (dashboard + audio children) if anything survived a crash:
python -m src.web_dashboard --stop-all
# same helper on the CLI entry:
python -m src.realtime_vc --stop-all
```

Exit (Ctrl+C / SIGTERM / SIGHUP / process exit) always stops capture+playback: signal + atexit hooks, `sd.stop()`, and audio-worker escalate to SIGKILL. `--daemon` is ignored (no orphan detach).

- **Voice** dropdown lists every `models/*.zip` plus extracted `.pth` under `models/_tmp/` (unavailable LFS placeholders are disabled)
- Choose input (default: MacBook / built-in mic) and output (default: EarPods / AirPods)
- **Start speaking** / **Stop** — Start loads the *selected* RVC voice; stream idle until then
- UI polls status/logs only (4s); voice/device catalogs load once (no full re-render spam)
- Realtime quality: block 20480, CPU, protect=0.33, RMS gain 1.25, block crossfade (not TTS)
- Reuses `load_rvc_v2` / `infer_audio` / duplex loop from `src.realtime_vc`
- Stdlib HTTP only (no FastAPI/Flask)

```bash
python -m src.realtime_vc --list-devices
python -m src.realtime_vc --output-device 2 --input-device 0
```

Apple Silicon uses **MPS** when available; otherwise **CPU**. Force with `--device cpu|mps|cuda`.

## Layout

```
rvc-realtime-voice-changer/
  README.md
  requirements.txt
  configs/default.yaml
  models/
    INDEX.md + *.zip     # EN feminine V2 archives (Git LFS)
    hubert/              # ContentVec (gitignored weights)
    _tmp/frida/          # extracted Frida .pth / .index
  src/
    realtime_vc.py       # CLI + duplex loop + AirPods/EarPods select
    web_dashboard.py     # Local browser UI (stdlib HTTP, Start/Stop)
    rvc_engine.py        # load + infer (16 kHz → model SR)
    rvc_pack/            # vendored RVC synthesizer (MIT, from rvc-python/RVC)
```

## Inference contract

`infer` path (see `src/rvc_engine.py`):

1. Resample input to **16 kHz mono**
2. ContentVec layer **12** → 768-D features (×2 time interpolate)
3. Optional FAISS `.index` blend
4. F0 via parselmouth (`pm`)
5. `SynthesizerTrnMs768NSFsid` → float waveform at **model SR** (Frida 40 kHz)

## Status

- [x] Realtime audio I/O + AirPods/EarPods output auto-select
- [x] Frida auto-unzip + ContentVec (lj1995) without fairseq
- [x] Real RVC V2 infer (vendored synthesizer pack)
- [x] `--dry-run` for headless import/path checks
- [x] Local web dashboard (Start/Stop, voice + devices)
- [ ] SOLA / crossfade low-latency tuning
- [ ] YAML config auto-load
- [ ] WhatsApp / virtual-cable routing

Reference upstream: [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).  
Synthesizer code vendored from the MIT [daswer123/rvc-python](https://github.com/daswer123/rvc-python) infer pack.
