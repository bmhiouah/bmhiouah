#!/usr/bin/env python3
"""
Realtime mic → (RVC V2 hook) → speaker loop.

Runs without model weights: audio is passed through unchanged until
`infer_rvc_v2()` is replaced with a real RVC V2 pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "sounddevice is required. Install with: pip install -r requirements.txt"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.environ.get("RVC_MODEL_PATH", "models/YOUR_SPEAKER_v2.pth")
DEFAULT_SR = 48_000
DEFAULT_BLOCK = 1024


def resolve_model_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_rvc_v2_placeholder(model_path: Path) -> dict[str, Any]:
    """
    Placeholder model loader.

    TODO: load RVC V2 .pth (+ optional .index) and HuBERT / content encoder.
    Keep returning a simple dict so the audio loop stays runnable without weights.
    """
    exists = model_path.is_file()
    if exists:
        print(f"[model] Found file at {model_path} (not loaded yet — stub).")
    else:
        print(
            f"[model] No weights at {model_path} — running passthrough. "
            "Drop a ready-made V2 .pth there when ready (see README)."
        )
    return {
        "path": str(model_path),
        "loaded": False,
        "exists": exists,
        # TODO: "net_g": ..., "hubert": ..., "index": ...
    }


def infer_rvc_v2(
    audio_block: np.ndarray,
    model: dict[str, Any],
    sample_rate: int,
) -> np.ndarray:
    """
    Convert one float32 mono block. Shape: (frames,) or (frames, 1).

    CURRENT: identity (passthrough) so mic→speaker works without weights.

    TODO — plug real RVC V2 here:
      1. Resample / pad block to model sample rate if needed
      2. Extract content features (HuBERT / contentvec)
      3. Optional FAISS index retrieval
      4. Run RVC V2 generator (net_g) with pitch / f0
      5. Return float32 waveform same length as input (or document framing)
    """
    _ = model, sample_rate  # used once real infer lands
    # Ensure contiguous float32 mono
    x = np.asarray(audio_block, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, 0]
    return x


def list_devices() -> None:
    print(sd.query_devices())
    print(f"Default input/output: {sd.default.device}")


def run_stream(
    model: dict[str, Any],
    sample_rate: int,
    block_size: int,
    input_device: Optional[int],
    output_device: Optional[int],
) -> None:
    """Open a duplex stream: capture → infer_rvc_v2 → play."""

    def callback(indata, outdata, frames, time_info, status):  # noqa: ARG001
        if status:
            # Underrun/overrun hints — useful when tuning block_size
            print(f"[audio] {status}", file=sys.stderr)
        converted = infer_rvc_v2(indata, model, sample_rate)
        if converted.ndim == 1:
            outdata[:, 0] = converted
        else:
            outdata[:] = converted.reshape(-1, 1)

    print(
        f"[audio] duplex {sample_rate} Hz, block={block_size}, "
        f"in={input_device}, out={output_device} — Ctrl+C to stop"
    )
    with sd.Stream(
        device=(input_device, output_device),
        samplerate=sample_rate,
        blocksize=block_size,
        dtype="float32",
        channels=1,
        callback=callback,
    ):
        while True:
            time.sleep(0.5)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime RVC V2 voice changer (skeleton)")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to RVC V2 .pth (placeholder OK)",
    )
    p.add_argument("--sr", type=int, default=DEFAULT_SR, help="Sample rate")
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK, help="Frames per block")
    p.add_argument("--input-device", type=int, default=None, help="sounddevice input id")
    p.add_argument("--output-device", type=int, default=None, help="sounddevice output id")
    p.add_argument("--list-devices", action="store_true", help="Print audio devices and exit")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_devices:
        list_devices()
        return 0

    model_path = resolve_model_path(args.model)
    model = load_rvc_v2_placeholder(model_path)

    try:
        run_stream(
            model=model,
            sample_rate=args.sr,
            block_size=args.block_size,
            input_device=args.input_device,
            output_device=args.output_device,
        )
    except KeyboardInterrupt:
        print("\n[audio] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
