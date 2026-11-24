#!/usr/bin/env python3
"""
Local browser dashboard for realtime RVC V2.

Stdlib HTTP server (no FastAPI/Flask). Open http://127.0.0.1:8765 after start.
Does not speak until you press Start.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import traceback
import zipfile
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

# Limit OpenMP threads early — libomp SIGSEGVs were seen during RVC infer on this Mac.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .realtime_vc import (
    DEFAULT_BLOCK,
    DEFAULT_F0_UP_KEY,
    DEFAULT_INDEX_RATE,
    DEFAULT_RMS_GAIN,
    ROOT,
    RealtimeSession,
    abort_audio_streams,
    auto_select_devices,
    install_shutdown_hooks,
    kill_related_rvc_processes,
)
from .rvc_engine import DEFAULT_FRIDA_PTH, ensure_frida_pth

try:
    import sounddevice as sd
except (ImportError, OSError):  # pragma: no cover
    sd = None  # type: ignore[assignment]

HOST = "127.0.0.1"
PORT = 8765
MODELS_DIR = ROOT / "models"
TMP_DIR = MODELS_DIR / "_tmp"
# Cache heavy catalog endpoints so 2–4s UI polls stay cheap.
_CATALOG_TTL_S = 20.0
_voices_cache: dict[str, Any] = {"t": 0.0, "payload": None}
_devices_cache: dict[str, Any] = {"t": 0.0, "payload": None}
_cache_lock = threading.Lock()

_session = RealtimeSession()
_log: deque[str] = deque(maxlen=200)
_log_lock = threading.Lock()


def log(msg: str) -> None:
    line = msg.rstrip()
    with _log_lock:
        _log.append(line)
    print(line, flush=True)


def _safe_slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()
    return s or "model"


def _find_index_near(pth: Path) -> Optional[Path]:
    sibling = pth.with_suffix(".index")
    if sibling.is_file():
        return sibling
    matches = sorted(pth.parent.glob("*.index"))
    return matches[0] if matches else None


def discover_voices() -> list[dict[str, Any]]:
    """List voices from models/*.zip and extracted .pth under models/_tmp."""
    voices: list[dict[str, Any]] = []
    claimed_pth: set[str] = set()

    # Zip archives in models/ (primary catalog)
    if MODELS_DIR.is_dir():
        for zpath in sorted(MODELS_DIR.glob("*.zip")):
            size = zpath.stat().st_size
            usable = size >= 10_000
            stem = zpath.stem
            extracted: Optional[Path] = None
            if stem.lower() == "frida" and DEFAULT_FRIDA_PTH.is_file():
                extracted = DEFAULT_FRIDA_PTH
            else:
                cand_dir = TMP_DIR / _safe_slug(stem)
                if cand_dir.is_dir():
                    found = list(cand_dir.rglob("*.pth"))
                    if found:
                        extracted = found[0]
                # Also claim any matching stem under _tmp
                if extracted is None and TMP_DIR.is_dir():
                    for p in TMP_DIR.rglob("*.pth"):
                        if p.stem.lower() == stem.lower() or stem.lower() in p.stem.lower():
                            extracted = p
                            break

            if extracted and extracted.is_file():
                claimed_pth.add(str(extracted.resolve()))
                idx = _find_index_near(extracted)
                rel = extracted.relative_to(ROOT)
                voices.append(
                    {
                        "id": f"zip:{zpath.name}",
                        "name": stem,
                        "kind": "zip",
                        "path": rel.as_posix(),
                        "zip": zpath.name,
                        "index_path": idx.relative_to(ROOT).as_posix() if idx else None,
                        "ready": True,
                        "size_bytes": size,
                    }
                )
                continue

            voices.append(
                {
                    "id": f"zip:{zpath.name}",
                    "name": stem,
                    "kind": "zip",
                    "path": None,
                    "zip": zpath.name,
                    "index_path": None,
                    "ready": usable,
                    "size_bytes": size,
                    "note": None if usable else "Archive placeholder / not pulled (git lfs)",
                }
            )

    # Orphan extracted .pth (not tied to a zip above)
    if TMP_DIR.is_dir():
        for pth in sorted(TMP_DIR.rglob("*.pth")):
            if not pth.is_file() or pth.stat().st_size < 1000:
                continue
            key = str(pth.resolve())
            if key in claimed_pth:
                continue
            rel = pth.relative_to(ROOT)
            idx = _find_index_near(pth)
            voices.append(
                {
                    "id": f"pth:{rel.as_posix()}",
                    "name": pth.stem,
                    "kind": "extracted",
                    "path": rel.as_posix(),
                    "index_path": idx.relative_to(ROOT).as_posix() if idx else None,
                    "ready": True,
                    "size_bytes": pth.stat().st_size,
                }
            )

    voices.sort(
        key=lambda v: (
            0 if "frida" in str(v.get("name", "")).lower() else 1,
            str(v.get("name", "")).lower(),
        )
    )
    return voices


def resolve_voice(voice_id: str) -> tuple[Path, Optional[Path]]:
    """Return (.pth path, optional .index) for a voice id; extract zip if needed."""
    voices = {v["id"]: v for v in discover_voices()}
    if voice_id not in voices:
        # Allow raw relative path
        p = ROOT / voice_id if not Path(voice_id).is_absolute() else Path(voice_id)
        if p.is_file() and p.suffix.lower() == ".pth":
            return p, _find_index_near(p)
        raise FileNotFoundError(f"Unknown voice: {voice_id}")

    meta = voices[voice_id]
    if meta.get("path"):
        pth = ROOT / meta["path"]
        if pth.is_file():
            idx = ROOT / meta["index_path"] if meta.get("index_path") else _find_index_near(pth)
            return pth, idx if idx and idx.is_file() else None

    zip_name = meta.get("zip")
    if not zip_name:
        raise FileNotFoundError(f"Voice has no path or zip: {voice_id}")

    zpath = MODELS_DIR / zip_name
    if not zpath.is_file() or zpath.stat().st_size < 10_000:
        raise FileNotFoundError(
            f"Zip unusable: {zpath.name} (run git lfs pull if this is a pointer file)"
        )

    if zpath.stem.lower() == "frida":
        pth = ensure_frida_pth(DEFAULT_FRIDA_PTH)
        return pth, _find_index_near(pth)

    out_dir = TMP_DIR / _safe_slug(zpath.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.rglob("*.pth"))
    if not existing:
        log(f"[model] Extracting {zpath.name} → {out_dir}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(out_dir)
        existing = list(out_dir.rglob("*.pth"))
    if not existing:
        raise FileNotFoundError(f"No .pth inside {zpath.name}")
    pth = existing[0]
    return pth, _find_index_near(pth)


def list_audio_devices(*, force: bool = False) -> dict[str, Any]:
    import time as _time

    now = _time.monotonic()
    with _cache_lock:
        cached = _devices_cache.get("payload")
        if (
            not force
            and cached is not None
            and (now - float(_devices_cache.get("t") or 0.0)) < _CATALOG_TTL_S
        ):
            return cached

    if sd is None:
        return {
            "ok": False,
            "error": "sounddevice/PortAudio unavailable",
            "devices": [],
            "default_input": None,
            "default_output": None,
        }

    devices = []
    try:
        raw = sd.query_devices()
        default_in, default_out = sd.default.device
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "devices": [],
            "default_input": None,
            "default_output": None,
        }

    if not isinstance(default_in, int):
        default_in = None
    if not isinstance(default_out, int):
        default_out = None

    for i, dev in enumerate(raw):
        name = str(dev.get("name", f"Device {i}"))
        max_in = int(dev.get("max_input_channels", 0) or 0)
        max_out = int(dev.get("max_output_channels", 0) or 0)
        devices.append(
            {
                "id": i,
                "name": name,
                "max_input_channels": max_in,
                "max_output_channels": max_out,
                "is_input": max_in > 0,
                "is_output": max_out > 0,
            }
        )

    # quiet=True — dashboard polls this; printing every time flooded the log + UI.
    suggested_in, suggested_out = auto_select_devices(None, None, quiet=True)
    payload = {
        "ok": True,
        "devices": devices,
        "system_default_input": default_in,
        "system_default_output": default_out,
        "default_input": suggested_in,
        "default_output": suggested_out,
    }
    with _cache_lock:
        _devices_cache["t"] = now
        _devices_cache["payload"] = payload
    return payload


def list_voices_cached(*, force: bool = False) -> list[dict[str, Any]]:
    import time as _time

    now = _time.monotonic()
    with _cache_lock:
        cached = _voices_cache.get("payload")
        if (
            not force
            and cached is not None
            and (now - float(_voices_cache.get("t") or 0.0)) < _CATALOG_TTL_S
        ):
            return cached
    voices = discover_voices()
    with _cache_lock:
        _voices_cache["t"] = now
        _voices_cache["payload"] = voices
    return voices


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RVC Realtime Voice Changer</title>
  <style>
    :root {
      --bg0: #0f1419;
      --bg1: #1a2332;
      --ink: #e8eef6;
      --muted: #8b9bb0;
      --accent: #3d9cf0;
      --accent2: #2a7bc8;
      --danger: #e85d5d;
      --ok: #3ecf8e;
      --line: rgba(232,238,246,0.12);
      --font: "IBM Plex Sans", "Segoe UI", sans-serif;
      --mono: "IBM Plex Mono", ui-monospace, monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh;
      font-family: var(--font);
      color: var(--ink);
      background:
        radial-gradient(900px 480px at 12% -10%, #1e3a5f 0%, transparent 55%),
        radial-gradient(700px 400px at 100% 0%, #163028 0%, transparent 50%),
        linear-gradient(165deg, var(--bg0), var(--bg1));
    }
    main {
      max-width: 640px;
      margin: 0 auto;
      padding: 2.5rem 1.25rem 3rem;
    }
    h1 {
      font-size: 1.65rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      margin: 0 0 0.35rem;
    }
    .sub { color: var(--muted); margin: 0 0 1.75rem; font-size: 0.95rem; }
    label {
      display: block;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin: 0 0 0.4rem;
    }
    .field { margin-bottom: 1.15rem; }
    select {
      width: 100%;
      padding: 0.7rem 0.85rem;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: rgba(0,0,0,0.28);
      color: var(--ink);
      font: inherit;
      font-size: 0.95rem;
    }
    select:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
    .row {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      margin-top: 1.5rem;
    }
    button {
      flex: 1 1 140px;
      padding: 0.85rem 1.1rem;
      border: none;
      border-radius: 8px;
      font: inherit;
      font-weight: 600;
      font-size: 1rem;
      cursor: pointer;
      transition: transform 0.12s ease, filter 0.12s ease, opacity 0.12s;
    }
    button:active { transform: scale(0.98); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    #btn-start {
      background: linear-gradient(180deg, var(--accent), var(--accent2));
      color: #fff;
    }
    #btn-stop {
      background: rgba(232,93,93,0.18);
      color: #ffb4b4;
      border: 1px solid rgba(232,93,93,0.35);
    }
    .status {
      margin-top: 1.25rem;
      padding: 0.75rem 0.9rem;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: rgba(0,0,0,0.22);
      font-size: 0.9rem;
      display: flex;
      align-items: center;
      gap: 0.55rem;
    }
    .dot {
      width: 9px; height: 9px; border-radius: 50%;
      background: var(--muted);
      flex-shrink: 0;
    }
    .dot.on { background: var(--ok); box-shadow: 0 0 10px rgba(62,207,142,0.55); }
    .dot.err { background: var(--danger); }
    #log {
      margin-top: 1.25rem;
      height: 220px;
      overflow: auto;
      padding: 0.85rem;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: rgba(0,0,0,0.35);
      font-family: var(--mono);
      font-size: 0.75rem;
      line-height: 1.45;
      white-space: pre-wrap;
      color: #b8c5d6;
    }
    .hint { margin-top: 0.75rem; color: var(--muted); font-size: 0.8rem; }
  </style>
</head>
<body>
  <main>
    <h1>RVC Realtime Voice Changer</h1>
    <p class="sub">Local mic → RVC V2 → speakers. Nothing runs until you press Start.</p>

    <div class="field">
      <label for="voice">Voice</label>
      <select id="voice"></select>
    </div>
    <div class="field">
      <label for="input">Input device</label>
      <select id="input"></select>
    </div>
    <div class="field">
      <label for="output">Output device</label>
      <select id="output"></select>
    </div>

    <div class="row">
      <button type="button" id="btn-start">Start speaking</button>
      <button type="button" id="btn-stop" disabled>Stop</button>
    </div>

    <div class="status" id="status">
      <span class="dot" id="dot"></span>
      <span id="status-text">Idle — not speaking</span>
    </div>
    <div id="log" aria-live="polite"></div>
    <p class="hint">Server: http://127.0.0.1:8765 · Voice list from models/*.zip + models/_tmp · RVC V2 (not TTS)</p>
  </main>
  <script>
    const voiceEl = document.getElementById("voice");
    const inputEl = document.getElementById("input");
    const outputEl = document.getElementById("output");
    const btnStart = document.getElementById("btn-start");
    const btnStop = document.getElementById("btn-stop");
    const statusText = document.getElementById("status-text");
    const dot = document.getElementById("dot");
    const logEl = document.getElementById("log");

    // Lag fix: catalog (voices/devices) loaded once; light poll only status+logs.
    // Never rebuild <select>s on every tick (that was the main UI jank).
    const POLL_MS = 4000;
    let catalogReady = false;
    let lastLogText = "";
    let pollBusy = false;

    function setRunning(on, err) {
      btnStart.disabled = on;
      btnStop.disabled = !on;
      voiceEl.disabled = on;
      inputEl.disabled = on;
      outputEl.disabled = on;
      dot.className = "dot" + (err ? " err" : on ? " on" : "");
      statusText.textContent = err
        ? ("Error: " + err)
        : on
          ? "Speaking…"
          : "Idle — not speaking";
    }

    async function api(path, opts) {
      const res = await fetch(path, opts);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    }

    function fillSelect(el, items, valueKey, labelFn, selected, disabledFn) {
      const prev = el.value;
      el.innerHTML = "";
      for (const item of items) {
        const opt = document.createElement("option");
        opt.value = String(item[valueKey]);
        opt.textContent = labelFn(item);
        if (disabledFn && disabledFn(item)) opt.disabled = true;
        el.appendChild(opt);
      }
      const want = selected != null ? String(selected) : (prev || "");
      if (want && [...el.options].some((o) => o.value === want && !o.disabled)) {
        el.value = want;
      } else {
        const firstOk = [...el.options].find((o) => !o.disabled);
        if (firstOk) el.value = firstOk.value;
      }
    }

    function applyStatus(status) {
      setRunning(!!status.running, status.error || null);
    }

    function applyLogs(logs) {
      const text = (logs.lines || []).join("\\n");
      if (text === lastLogText) return;
      const nearBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;
      lastLogText = text;
      logEl.textContent = text;
      if (nearBottom) logEl.scrollTop = logEl.scrollHeight;
    }

    async function loadCatalog(statusHint) {
      const [voices, devices, status] = await Promise.all([
        api("/api/voices"),
        api("/api/devices"),
        statusHint ? Promise.resolve(statusHint) : api("/api/status"),
      ]);
      fillSelect(
        voiceEl,
        voices.voices || [],
        "id",
        (v) => {
          const tag = v.ready ? "" : " (unavailable — git lfs)";
          const kind = v.kind === "zip" ? "zip" : "pth";
          return `${v.name} [${kind}]${tag}`;
        },
        status.voice_id || null,
        (v) => !v.ready
      );
      const inputs = (devices.devices || []).filter((d) => d.is_input);
      const outputs = (devices.devices || []).filter((d) => d.is_output);
      fillSelect(
        inputEl,
        inputs,
        "id",
        (d) => `${d.id}: ${d.name}`,
        status.input_device != null ? status.input_device : devices.default_input
      );
      fillSelect(
        outputEl,
        outputs,
        "id",
        (d) => `${d.id}: ${d.name}`,
        status.output_device != null ? status.output_device : devices.default_output
      );
      applyStatus(status);
      catalogReady = true;
    }

    async function pollLight() {
      if (pollBusy) return;
      pollBusy = true;
      try {
        const [status, logs] = await Promise.all([
          api("/api/status"),
          api("/api/logs"),
        ]);
        applyStatus(status);
        applyLogs(logs);
      } catch (e) {
        /* keep last UI state if server briefly unreachable */
      } finally {
        pollBusy = false;
      }
    }

    btnStart.addEventListener("click", async () => {
      if (!voiceEl.value) {
        setRunning(false, "No voice selected");
        return;
      }
      setRunning(true, null);
      try {
        await api("/api/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            voice_id: voiceEl.value,
            input_device: Number(inputEl.value),
            output_device: Number(outputEl.value),
          }),
        });
      } catch (e) {
        setRunning(false, e.message);
      }
      await pollLight();
    });

    btnStop.addEventListener("click", async () => {
      try {
        await api("/api/stop", { method: "POST" });
      } catch (e) {
        statusText.textContent = "Stop failed: " + e.message;
      }
      await pollLight();
    });

    (async () => {
      try {
        await loadCatalog();
        const logs = await api("/api/logs");
        applyLogs(logs);
      } catch (e) {
        setRunning(false, e.message || String(e));
      }
      setInterval(pollLight, POLL_MS);
    })();
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep access noise out of the voice log panel
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self._send(code, data, "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/api/voices":
            self._json(200, {"voices": list_voices_cached()})
            return
        if path == "/api/devices":
            self._json(200, list_audio_devices())
            return
        if path == "/api/status":
            st = _session.status()
            self._json(200, st)
            return
        if path == "/api/logs":
            with _log_lock:
                lines = list(_log)
            self._json(200, {"lines": lines})
            return
        self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/start":
            body = self._read_json()
            voice_id = str(body.get("voice_id") or "")
            try:
                in_dev = body.get("input_device")
                out_dev = body.get("output_device")
                input_device = int(in_dev) if in_dev is not None and in_dev != "" else None
                output_device = int(out_dev) if out_dev is not None and out_dev != "" else None
                log(f"[ui] Start requested voice={voice_id} in={input_device} out={output_device}")
                pth, idx = resolve_voice(voice_id)
                log(f"[ui] Resolved model → {pth}")
                # Quality path: larger block, CPU, protect consonants, RMS match.
                # index_rate stays 0 — FAISS IVF search segfaults on this Mac stack.
                st = _session.start(
                    pth,
                    index_path=idx,
                    input_device=input_device,
                    output_device=output_device,
                    block_size=DEFAULT_BLOCK,
                    device="cpu",
                    f0_up_key=DEFAULT_F0_UP_KEY,
                    index_rate=DEFAULT_INDEX_RATE,
                    rms_gain=DEFAULT_RMS_GAIN,
                    protect=0.33,
                    voice_id=voice_id,
                )
                self._json(200, st)
            except Exception as exc:  # noqa: BLE001
                log(f"[ui] Start failed: {exc}")
                traceback.print_exc()
                self._json(400, {"error": str(exc), **_session.status()})
            return

        if path == "/api/stop":
            log("[ui] Stop requested")
            st = _session.stop()
            self._json(200, st)
            return

        self._json(404, {"error": "Not found"})


def main(argv: Optional[list[str]] = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    host = HOST
    port = PORT
    pid_path = Path("/tmp/rvc-web-dashboard.pid")
    stop_all = False
    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--daemon":
            # Kept as a no-op for old scripts. Orphan detach was removed: exit must
            # always stop capture+playback (no surviving session after close).
            log(
                "[dashboard] --daemon ignored: orphan detach disabled "
                "(use foreground; --stop-all to force-kill leftovers)"
            )
            i += 1
        elif args[i] == "--stop-all":
            stop_all = True
            i += 1
        elif args[i] == "--pid-file" and i + 1 < len(args):
            pid_path = Path(args[i + 1])
            i += 2
        elif args[i] == "--log-file" and i + 1 < len(args):
            # Accepted for backwards compat; logging stays on stdout/stderr.
            i += 2
        else:
            print(f"Unknown arg: {args[i]}", file=sys.stderr)
            return 2

    if stop_all:
        records = kill_related_rvc_processes()
        if not records:
            print("[shutdown] no RVC processes found", flush=True)
        else:
            for rec in records:
                print(
                    f"[shutdown] pid={rec['pid']} killed={rec['killed']} "
                    f"cmd={rec['cmd'][:160]}",
                    flush=True,
                )
        return 0

    # Warm device defaults into the log (no audio stream yet)
    log("[dashboard] Idle — waiting for Start in the browser")
    try:
        info = list_audio_devices()
        if info.get("ok"):
            log(
                f"[audio] Suggested input={info.get('default_input')} "
                f"output={info.get('default_output')}"
            )
        else:
            log(f"[audio] Device list unavailable: {info.get('error')}")
    except Exception as exc:  # noqa: BLE001
        log(f"[audio] {exc}")

    # Touch voice list so Frida shows as ready when extracted
    n = len(discover_voices())
    log(f"[dashboard] {n} voice(s) discovered under models/")

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    # Allow fast rebinds after crash / force-kill.
    server.allow_reuse_address = True
    url = f"http://{host}:{port}/"
    log(f"[dashboard] Serving {url} pid={os.getpid()} detached=false")
    log("[dashboard] Press Ctrl+C to quit (always stops capture+playback)")

    shutting_down = {"done": False}

    def _shutdown() -> None:
        if shutting_down["done"]:
            return
        shutting_down["done"] = True
        log("[dashboard] shutting down — stopping capture+playback")
        try:
            _session.stop(timeout=1.5)
        except Exception as exc:  # noqa: BLE001
            log(f"[dashboard] session stop error: {exc}")
        abort_audio_streams()
        try:
            server.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            server.server_close()
        except Exception:  # noqa: BLE001
            pass
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass

    install_shutdown_hooks(_shutdown)
    try:
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    except OSError as exc:
        log(f"[dashboard] could not write pid file {pid_path}: {exc}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
