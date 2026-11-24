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
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Spawn isolates native audio/torch crashes from the HTTP server process.
_MP = mp.get_context("spawn")

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
# Larger blocks give RVC headroom on CPU and avoid PortAudio underruns.
DEFAULT_BLOCK = 16384
# Realtime defaults to CPU — MPS can segfault / stall inside the audio callback.
DEFAULT_TORCH_DEVICE = os.environ.get("RVC_TORCH_DEVICE", "cpu")
# FAISS IVF search has segfaulted on this Mac (faiss-cpu + numpy); skip by default.
DEFAULT_INDEX_RATE = float(os.environ.get("RVC_INDEX_RATE", "0"))
# Prefer a CoreAudio-native rate for mic/EarPods; RVC still runs at model SR via resample.
PREFERRED_STREAM_RATES = (48000, 44100, 40000, 32000, 16000)
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


def _stream_channel_counts(
    input_device: Optional[int],
    output_device: Optional[int],
) -> tuple[int, int]:
    """Mono mic in; up to stereo out so EarPods / AirPods get both channels."""
    in_ch, out_ch = 1, 1
    try:
        if input_device is not None:
            max_in = int(sd.query_devices(input_device)["max_input_channels"] or 0)
            in_ch = 1 if max_in >= 1 else max(1, max_in)
        if output_device is not None:
            max_out = int(sd.query_devices(output_device)["max_output_channels"] or 0)
            if max_out >= 2:
                out_ch = 2
            elif max_out >= 1:
                out_ch = 1
    except Exception as exc:  # noqa: BLE001
        print(f"[audio] Channel query fallback ({exc})", flush=True)
    return in_ch, out_ch


def _pick_stream_sample_rate(
    input_device: Optional[int],
    output_device: Optional[int],
    preferred: Optional[int] = None,
) -> int:
    """Choose a sample rate both devices can open (EarPods often prefer 44100)."""
    candidates: list[int] = []
    if preferred:
        candidates.append(int(preferred))
    for dev in (input_device, output_device):
        if dev is None:
            continue
        try:
            rate = int(float(sd.query_devices(dev).get("default_samplerate") or 0))
            if rate > 0:
                candidates.append(rate)
        except Exception:  # noqa: BLE001
            pass
    candidates.extend(PREFERRED_STREAM_RATES)
    seen: set[int] = set()
    ordered: list[int] = []
    for rate in candidates:
        if rate not in seen:
            seen.add(rate)
            ordered.append(rate)

    in_ch, out_ch = _stream_channel_counts(input_device, output_device)
    for rate in ordered:
        try:
            with sd.InputStream(
                device=input_device,
                samplerate=rate,
                channels=in_ch,
                dtype="float32",
                blocksize=512,
            ), sd.OutputStream(
                device=output_device,
                samplerate=rate,
                channels=out_ch,
                dtype="float32",
                blocksize=512,
            ):
                return rate
        except Exception:  # noqa: BLE001
            continue
    return ordered[0] if ordered else 48000


def run_stream(
    model: dict[str, Any],
    sample_rate: int,
    block_size: int,
    input_device: Optional[int],
    output_device: Optional[int],
    *,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Open separate capture/playback streams: mic → RVC V2 → speakers.

    Pass ``stop_event`` to allow a clean stop from another thread (web UI).
    Without it, runs until KeyboardInterrupt / process kill.

    Uses separate InputStream/OutputStream (not one duplex Stream) because
    mic + EarPods are different CoreAudio devices; blocking duplex read can hang.
    """
    _require_sounddevice()

    from .rvc_engine import _resample  # local helper; keeps playback at stream SR

    model_sr = int(model.get("tgt_sr") or sample_rate)
    stream_sr = _pick_stream_sample_rate(input_device, output_device, preferred=sample_rate)
    engine = model.get("engine")
    in_ch, out_ch = _stream_channel_counts(input_device, output_device)
    # Blocking read/write (not a PortAudio callback): RVC on CPU can take longer
    # than one block; a callback would stall PortAudio and hang Stop.

    stats = {"n": 0, "peak_in": 0.0, "peak_out": 0.0}

    print(
        f"[audio] streams {stream_sr} Hz (model {model_sr} Hz), block={block_size}, "
        f"channels=({in_ch},{out_ch}), in={input_device}, out={output_device} "
        f"— Ctrl+C to stop",
        flush=True,
    )
    if engine is not None:
        print(
            f"[audio] RVC {model.get('version')} on {model.get('device')} "
            f"(HuBERT @ {HUBERT_SR} Hz → model @ {model_sr} Hz → play @ {stream_sr} Hz)",
            flush=True,
        )

    try:
        with sd.InputStream(
            device=input_device,
            samplerate=stream_sr,
            blocksize=block_size,
            dtype="float32",
            channels=in_ch,
            latency="high",
        ) as istream, sd.OutputStream(
            device=output_device,
            samplerate=stream_sr,
            blocksize=block_size,
            dtype="float32",
            channels=out_ch,
            latency="high",
        ) as ostream:
            print(
                f"[audio] duplex OPEN active=({istream.active},{ostream.active}) "
                f"in={_device_name(input_device) if input_device is not None else 'default'} "
                f"out={_device_name(output_device) if output_device is not None else 'default'}",
                flush=True,
            )
            while stop_event is None or not stop_event.is_set():
                indata, overflowed = istream.read(block_size)
                if overflowed:
                    print(f"[audio] input overflow={overflowed}", file=sys.stderr, flush=True)
                mono = np.asarray(indata[:, 0], dtype=np.float32)
                t0 = time.perf_counter()
                try:
                    converted = infer_audio(mono, model, stream_sr)
                    # Model may emit at tgt_sr (e.g. 40 kHz); resample to stream SR.
                    if int(model_sr) != int(stream_sr):
                        converted = _resample(converted, model_sr, stream_sr)
                    if converted.shape[0] != block_size:
                        converted = _match_length(converted, block_size)
                except Exception as exc:  # noqa: BLE001
                    print(f"[infer] {exc}", file=sys.stderr, flush=True)
                    converted = mono
                infer_ms = (time.perf_counter() - t0) * 1000.0

                outdata = np.zeros((block_size, out_ch), dtype=np.float32)
                outdata[:, 0] = converted
                if out_ch > 1:
                    outdata[:, 1] = converted
                under = ostream.write(outdata)
                if under:
                    print(f"[audio] output underflow={under}", file=sys.stderr, flush=True)

                rms_in = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                rms_out = (
                    float(np.sqrt(np.mean(np.square(converted)))) if converted.size else 0.0
                )
                stats["n"] += 1
                stats["peak_in"] = max(stats["peak_in"], rms_in)
                stats["peak_out"] = max(stats["peak_out"], rms_out)
                n = stats["n"]
                if n <= 8 or n % 20 == 0:
                    print(
                        f"[audio] block={n} rms_in={rms_in:.5f} rms_out={rms_out:.5f} "
                        f"peak_in={stats['peak_in']:.5f} peak_out={stats['peak_out']:.5f} "
                        f"infer_ms={infer_ms:.0f}",
                        flush=True,
                    )
            print(
                f"[audio] duplex CLOSED after {stats['n']} blocks "
                f"peak_in={stats['peak_in']:.5f} peak_out={stats['peak_out']:.5f}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[audio] Stream open/run failed: {exc}", file=sys.stderr, flush=True)
        raise


def _audio_worker_main(
    model_path: str,
    index_path: Optional[str],
    input_device: Optional[int],
    output_device: Optional[int],
    block_size: int,
    device: str,
    f0_up_key: int,
    index_rate: float,
    stop_event: Any,
) -> None:
    """Child process entry: load RVC + run streams. Crashes here leave the dashboard alive."""
    try:
        path = Path(model_path)
        if "frida" in path.name.lower():
            ensure_frida_pth(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        print(
            f"[session] Loading RVC on {device} (block={block_size}, index_rate={index_rate})…",
            flush=True,
        )
        model = load_rvc_v2(
            path,
            index_path=index_path,
            device=device,
            download_hubert=True,
            f0_up_key=f0_up_key,
            index_rate=index_rate,
        )
        in_dev, out_dev = auto_select_devices(input_device, output_device)
        stream_sr = int(model["tgt_sr"])
        print(
            f"[session] Starting duplex in={in_dev} out={out_dev} sr={stream_sr}",
            flush=True,
        )
        run_stream(
            model=model,
            sample_rate=stream_sr,
            block_size=block_size,
            input_device=in_dev,
            output_device=out_dev,
            stop_event=stop_event,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[session] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


class RealtimeSession:
    """Start/stop wrapper for the web UI.

    Audio runs in a **spawned child process** (not only a thread): a native
    segfault in PortAudio/torch kills the child, while the HTTP dashboard stays up.
    A small watcher thread joins the child and updates ``running`` / ``error``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[mp.Process] = None
        self._watcher: Optional[threading.Thread] = None
        self._stop: Optional[Any] = None
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
        index_rate: float = DEFAULT_INDEX_RATE,
    ) -> dict[str, Any]:
        # IMPORTANT: never call status() while holding this lock (non-reentrant).
        torch_device = device or DEFAULT_TORCH_DEVICE
        use_index_rate = float(index_rate)
        use_index_path = str(index_path) if index_path and use_index_rate > 0 else None
        path = resolve_model_path(str(model_path))

        with self._lock:
            if self.running:
                raise RuntimeError("Already speaking — stop first")
            self._error = None
            self.model_path = str(path)
            self.input_device = input_device
            self.output_device = output_device
            self.block_size = int(block_size)
            self.device = torch_device
            self.running = True
            stop_event = _MP.Event()
            self._stop = stop_event
            proc = _MP.Process(
                target=_audio_worker_main,
                name="rvc-audio",
                args=(
                    str(path),
                    use_index_path,
                    input_device,
                    output_device,
                    int(block_size),
                    torch_device,
                    int(f0_up_key),
                    use_index_rate,
                    stop_event,
                ),
                daemon=True,
            )
            self._proc = proc
            proc.start()
            print(f"[session] audio worker pid={proc.pid}", flush=True)

            def watcher() -> None:
                proc.join()
                code = proc.exitcode
                with self._lock:
                    self.running = False
                    self._proc = None
                    # None/0 = clean; negative = killed by signal (e.g. segfault)
                    if code not in (0, None) and not self._error:
                        if code is not None and code < 0:
                            self._error = f"audio worker crashed (signal {-code})"
                        else:
                            self._error = f"audio worker exited ({code})"
                        print(f"[session] {self._error}", file=sys.stderr, flush=True)

            self._watcher = threading.Thread(target=watcher, name="rvc-watch", daemon=True)
            self._watcher.start()
        return self.status()

    def stop(self, timeout: float = 30.0) -> dict[str, Any]:
        with self._lock:
            stop_event = self._stop
            proc = self._proc
            if stop_event is not None:
                stop_event.set()
        if proc is not None and proc.is_alive():
            proc.join(timeout=timeout)
            if proc.is_alive():
                print("[session] stop timeout — terminating audio worker", flush=True)
                proc.terminate()
                proc.join(timeout=5.0)
        with self._lock:
            if proc is not None and proc.is_alive() and not self._error:
                self._error = "stop timed out — stream may still be closing"
            self.running = False
            self._proc = None
            self._stop = None
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
        default=DEFAULT_TORCH_DEVICE,
        help="Torch device: cpu | mps | cuda (default: cpu for realtime stability)",
    )
    p.add_argument("--f0-up-key", type=int, default=0, help="Pitch shift in semitones")
    p.add_argument(
        "--index-rate",
        type=float,
        default=DEFAULT_INDEX_RATE,
        help="FAISS blend rate 0..1 (default 0 — IVF search can segfault on macOS)",
    )
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
            index_path=args.index if args.index_rate > 0 else None,
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
