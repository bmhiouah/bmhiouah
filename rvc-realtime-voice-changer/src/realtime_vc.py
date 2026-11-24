#!/usr/bin/env python3
"""
Realtime mic → RVC V2 → speaker loop.

Defaults to Frida (``models/_tmp/frida/frida.pth``); unzips ``models/Frida.zip``
if needed. ContentVec/HuBERT is loaded from ``models/hubert/`` (auto-download
from lj1995 hub — see README).

Device policy:
  - Output: prefer AirPods when the name matches via sounddevice
  - Input: system default / built-in mic (never force AirPods mic)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):  # pragma: no cover — missing package or PortAudio
    sd = None  # type: ignore[assignment]

from .rvc_engine import (
    DEFAULT_FRIDA_INDEX,
    DEFAULT_FRIDA_PTH,
    HUBERT_HF_DIR,
    HUBERT_PT,
    HUBERT_SR,
    HUBERT_URL,
    ensure_frida_pth,
    ensure_hubert_weights,
    infer_audio,
    load_rvc_v2,
    pick_torch_device,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.environ.get("RVC_MODEL_PATH", str(DEFAULT_FRIDA_PTH.relative_to(ROOT)))
DEFAULT_BLOCK = 4096  # larger blocks help RVC quality / CPU headroom
# Prefer AirPods / EarPods for output when connected
AIRPODS_TOKENS = ("airpods", "airpod", "earpods", "earpod")
BUILTIN_MIC_TOKENS = (
    "built-in",
    "builtin",
    "macbook",
    "internal microphone",
    "internal mic",
    "microphone array",
)


def resolve_model_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _require_sounddevice() -> None:
    if sd is None:
        raise RuntimeError(
            "sounddevice/PortAudio unavailable. "
            "Install with: pip install -r requirements.txt "
            "(on macOS: brew install portaudio). "
            "Use --dry-run to verify models without audio hardware."
        )


def list_devices() -> None:
    _require_sounddevice()
    print(sd.query_devices())
    print(f"Default input/output: {sd.default.device}")


def _device_name(index: int) -> str:
    try:
        return str(sd.query_devices(index)["name"])
    except Exception:  # noqa: BLE001
        return f"#{index}"


def auto_select_devices(
    input_device: Optional[int],
    output_device: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    """
    Prefer AirPods for *output* when connected; keep input on default/built-in mic.

    Explicit ``--input-device`` / ``--output-device`` always win.
    """
    if input_device is not None and output_device is not None:
        return input_device, output_device
    if sd is None:
        return input_device, output_device

    try:
        devices = sd.query_devices()
        default_in, default_out = sd.default.device
    except Exception as exc:  # noqa: BLE001
        print(f"[audio] Device query failed ({exc}); using explicit/defaults only")
        return input_device, output_device

    if not isinstance(default_in, int):
        default_in = None
    if not isinstance(default_out, int):
        default_out = None

    airpods_out: Optional[int] = None
    builtin_in: Optional[int] = None

    for i, dev in enumerate(devices):
        name = str(dev.get("name", "")).lower()
        max_in = int(dev.get("max_input_channels", 0) or 0)
        max_out = int(dev.get("max_output_channels", 0) or 0)
        if airpods_out is None and max_out > 0 and any(t in name for t in AIRPODS_TOKENS):
            airpods_out = i
        if builtin_in is None and max_in > 0 and any(t in name for t in BUILTIN_MIC_TOKENS):
            # Skip AirPods mic even if it says microphone
            if not any(t in name for t in AIRPODS_TOKENS):
                builtin_in = i

    chosen_out = output_device
    if chosen_out is None:
        chosen_out = airpods_out if airpods_out is not None else default_out
        if airpods_out is not None:
            print(f"[audio] AirPods output → {_device_name(airpods_out)} (id={airpods_out})")
        elif chosen_out is not None:
            print(f"[audio] Default output → {_device_name(chosen_out)} (id={chosen_out})")

    chosen_in = input_device
    if chosen_in is None:
        chosen_in = builtin_in if builtin_in is not None else default_in
        # Never silently pick AirPods as mic when we preferred built-in
        if chosen_in is not None:
            in_name = _device_name(chosen_in).lower()
            if any(t in in_name for t in AIRPODS_TOKENS) and builtin_in is not None:
                chosen_in = builtin_in
            print(f"[audio] Input mic → {_device_name(chosen_in)} (id={chosen_in})")

    return chosen_in, chosen_out


def _match_length(audio: np.ndarray, frames: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.shape[0] == frames:
        return x
    if x.shape[0] > frames:
        return x[:frames]
    out = np.zeros(frames, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def run_stream(
    model: dict[str, Any],
    sample_rate: int,
    block_size: int,
    input_device: Optional[int],
    output_device: Optional[int],
    *,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Open a duplex stream: capture → RVC V2 → play.

    Pass ``stop_event`` to allow a clean stop from another thread (web UI).
    Without it, runs until KeyboardInterrupt / process kill.
    """
    _require_sounddevice()

    # Stream at model SR so playback matches generator output without extra resample.
    stream_sr = int(model.get("tgt_sr") or sample_rate)
    engine = model.get("engine")
    # Callback runs RVC on each block. For low latency on-device, raise --block-size.

    def callback(indata, outdata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        try:
            # Resample stream_sr → 16 kHz inside engine, generate at tgt_sr
            converted = infer_audio(mono, model, stream_sr)
            # If lengths diverge (hop framing), fit to callback frames
            if converted.shape[0] != frames:
                # Ratio tgt_sr/stream_sr should be ~1 when stream_sr == tgt_sr
                converted = _match_length(converted, frames)
        except Exception as exc:  # noqa: BLE001
            print(f"[infer] {exc}", file=sys.stderr)
            converted = mono
        outdata[:, 0] = converted

    print(
        f"[audio] duplex {stream_sr} Hz, block={block_size}, "
        f"in={input_device}, out={output_device} — Ctrl+C to stop"
    )
    if engine is not None:
        print(
            f"[audio] RVC {model.get('version')} on {model.get('device')} "
            f"(HuBERT @ {HUBERT_SR} Hz → out @ {model.get('tgt_sr')} Hz)"
        )

    with sd.Stream(
        device=(input_device, output_device),
        samplerate=stream_sr,
        blocksize=block_size,
        dtype="float32",
        channels=1,
        callback=callback,
    ):
        if stop_event is None:
            while True:
                time.sleep(0.5)
        else:
            while not stop_event.is_set():
                time.sleep(0.1)


class RealtimeSession:
    """Start/stop wrapper around ``load_rvc_v2`` + ``run_stream`` for the web UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._error: Optional[str] = None
        self.running = False
        self.model_path: Optional[str] = None
        self.input_device: Optional[int] = None
        self.output_device: Optional[int] = None
        self.device: Optional[str] = None
        self.block_size = DEFAULT_BLOCK

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "model_path": self.model_path,
                "input_device": self.input_device,
                "output_device": self.output_device,
                "device": self.device,
                "block_size": self.block_size,
                "error": self._error,
            }

    def start(
        self,
        model_path: str | Path,
        *,
        index_path: Optional[str | Path] = None,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        block_size: int = DEFAULT_BLOCK,
        device: Optional[str] = None,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
    ) -> dict[str, Any]:
        with self._lock:
            if self.running:
                raise RuntimeError("Already speaking — stop first")
            self._error = None
            self._stop.clear()
            path = resolve_model_path(str(model_path))
            self.model_path = str(path)
            self.input_device = input_device
            self.output_device = output_device
            self.block_size = int(block_size)
            self.device = device
            self.running = True

            def worker() -> None:
                try:
                    if "frida" in path.name.lower():
                        ensure_frida_pth(path)
                    if not path.is_file():
                        raise FileNotFoundError(f"Model not found: {path}")
                    model = load_rvc_v2(
                        path,
                        index_path=index_path,
                        device=device,
                        download_hubert=True,
                        f0_up_key=f0_up_key,
                        index_rate=index_rate,
                    )
                    in_dev, out_dev = auto_select_devices(input_device, output_device)
                    with self._lock:
                        self.input_device = in_dev
                        self.output_device = out_dev
                        self.device = str(model.get("device"))
                    stream_sr = int(model["tgt_sr"])
                    run_stream(
                        model=model,
                        sample_rate=stream_sr,
                        block_size=self.block_size,
                        input_device=in_dev,
                        output_device=out_dev,
                        stop_event=self._stop,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._error = str(exc)
                    print(f"[session] {exc}", file=sys.stderr)
                finally:
                    with self._lock:
                        self.running = False

            self._thread = threading.Thread(target=worker, name="rvc-realtime", daemon=True)
            self._thread.start()
            return self.status()

    def stop(self, timeout: float = 5.0) -> dict[str, Any]:
        with self._lock:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            self.running = False
            self._thread = None
            return self.status()


def dry_run(model_path: Path, *, download_hubert: bool = False) -> int:
    """Imports + path resolve only; no audio stream. Safe on headless hosts."""
    print("[dry-run] Checking imports…")
    import torch  # noqa: F401
    import librosa  # noqa: F401
    import parselmouth  # noqa: F401
    from transformers import HubertModel  # noqa: F401
    from .rvc_pack.models import SynthesizerTrnMs768NSFsid  # noqa: F401

    print(f"[dry-run] torch {torch.__version__} device={pick_torch_device()}")
    print(f"[dry-run] ROOT={ROOT}")

    try:
        pth = ensure_frida_pth(model_path)
        print(f"[dry-run] Frida .pth OK → {pth} ({pth.stat().st_size} bytes)")
    except FileNotFoundError as exc:
        print(f"[dry-run] Frida path error: {exc}")
        return 1

    idx = DEFAULT_FRIDA_INDEX
    if idx.is_file():
        print(f"[dry-run] Index OK → {idx}")
    else:
        print(f"[dry-run] Index not found (optional): {idx}")

    print(f"[dry-run] HuBERT expected pt: {HUBERT_PT}")
    print(f"[dry-run] HuBERT HF dir:     {HUBERT_HF_DIR}")
    print(f"[dry-run] HuBERT download:   {HUBERT_URL}")
    try:
        hub = ensure_hubert_weights(download=download_hubert)
        print(f"[dry-run] HuBERT/ContentVec OK → {hub}")
    except FileNotFoundError as exc:
        print(f"[dry-run] HuBERT not ready (download with --download-hubert): {exc}")
        # Still success for path/import check if Frida resolved
        return 0

    # Optional: load weights if hubert ready (proves checkpoint wiring)
    try:
        model = load_rvc_v2(pth, download_hubert=False)
        print(
            f"[dry-run] Loaded RVC {model['version']} tgt_sr={model['tgt_sr']} "
            f"device={model['device']}"
        )
        # Tiny synthetic infer
        fake = np.zeros(HUBERT_SR // 2, dtype=np.float32)
        fake[1000:3000] = 0.1 * np.sin(2 * np.pi * 220 * np.arange(2000) / HUBERT_SR)
        out = infer_audio(fake, model, HUBERT_SR)
        print(f"[dry-run] Synthetic infer OK → out samples={out.shape[0]} sr={model['tgt_sr']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[dry-run] Model load/infer error: {exc}")
        return 1

    # Audio hardware is optional on headless hosts
    if sd is None:
        print("[dry-run] sounddevice/PortAudio unavailable — skipping device list")
    else:
        try:
            list_devices()
        except Exception as exc:  # noqa: BLE001
            print(f"[dry-run] No audio hardware / PortAudio query failed: {exc}")

    print("[dry-run] OK")
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime RVC V2 voice changer")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to RVC V2 .pth (default: Frida)",
    )
    p.add_argument(
        "--index",
        default=None,
        help="Optional FAISS .index path (default: next to Frida)",
    )
    p.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Stream sample rate override (default: model tgt_sr, e.g. 40000 for Frida)",
    )
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK, help="Frames per block")
    p.add_argument("--input-device", type=int, default=None, help="sounddevice input id")
    p.add_argument("--output-device", type=int, default=None, help="sounddevice output id")
    p.add_argument("--list-devices", action="store_true", help="Print audio devices and exit")
    p.add_argument(
        "--device",
        default=None,
        help="Torch device: cpu | mps | cuda (default: mps if available else cpu)",
    )
    p.add_argument("--f0-up-key", type=int, default=0, help="Pitch shift in semitones")
    p.add_argument("--index-rate", type=float, default=0.5, help="FAISS blend rate 0..1")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Imports + path resolve (+ optional load); no audio stream",
    )
    p.add_argument(
        "--download-hubert",
        action="store_true",
        help="Download lj1995 hubert_base.pt and convert if missing",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_devices:
        list_devices()
        return 0

    model_path = resolve_model_path(args.model)

    if args.dry_run:
        return dry_run(model_path, download_hubert=args.download_hubert)

    try:
        ensure_frida_pth(model_path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        model = load_rvc_v2(
            model_path,
            index_path=args.index,
            device=args.device,
            download_hubert=True,
            f0_up_key=args.f0_up_key,
            index_rate=args.index_rate,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[model] Failed to load RVC: {exc}", file=sys.stderr)
        return 1

    in_dev, out_dev = auto_select_devices(args.input_device, args.output_device)
    stream_sr = args.sr or int(model["tgt_sr"])

    try:
        run_stream(
            model=model,
            sample_rate=stream_sr,
            block_size=args.block_size,
            input_device=in_dev,
            output_device=out_dev,
        )
    except KeyboardInterrupt:
        print("\n[audio] stopped")
    except Exception as exc:  # noqa: BLE001
        # Headless / no PortAudio devices
        print(f"[audio] Stream failed: {exc}", file=sys.stderr)
        print(
            "[audio] Tip: use --dry-run on machines without audio hardware; "
            "on a Mac, grant mic permission and check --list-devices.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
