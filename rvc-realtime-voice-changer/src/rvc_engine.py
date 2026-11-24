"""
Minimal RVC V2 inference helper (vendored synthesizer + ContentVec/HuBERT).

Pipeline:
  16 kHz mono float32 → ContentVec (layer 12, 768-D) → optional FAISS index
  → F0 (parselmouth) → SynthesizerTrnMs768NSFsid → waveform at model sample rate.

HuBERT / ContentVec weights (lj1995 hub contentvec, shipped as hubert_base.pt):
  https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt

Place at ``models/hubert/hubert_base.pt``. On first load we convert it to a
local Transformers checkpoint under ``models/hubert/contentvec_hf/`` (no fairseq).
"""

from __future__ import annotations

import sys
import types
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRIDA_ZIP = ROOT / "models" / "Frida.zip"
DEFAULT_FRIDA_PTH = ROOT / "models" / "_tmp" / "frida" / "frida.pth"
DEFAULT_FRIDA_INDEX = (
    ROOT / "models" / "_tmp" / "frida" / "added_IVF780_Flat_nprobe_1_frida_v2.index"
)
HUBERT_PT = ROOT / "models" / "hubert" / "hubert_base.pt"
HUBERT_HF_DIR = ROOT / "models" / "hubert" / "contentvec_hf"
HUBERT_URL = (
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
)

HUBERT_SR = 16_000
WINDOW = 160  # samples per F0 / content frame at 16 kHz


def pick_torch_device(preferred: Optional[str] = None) -> torch.device:
    """CPU, or MPS on Apple Silicon when available. CUDA if explicitly preferred/available."""
    if preferred:
        pref = preferred.lower().strip()
        if pref == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            print("[device] MPS requested but unavailable — falling back to CPU")
            return torch.device("cpu")
        if pref.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(pref if ":" in pref else "cuda")
            print("[device] CUDA requested but unavailable — falling back to CPU")
            return torch.device("cpu")
        return torch.device("cpu")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_frida_pth(model_path: Optional[Path] = None) -> Path:
    """Resolve Frida .pth; unzip ``models/Frida.zip`` into ``models/_tmp/frida/`` if needed."""
    target = Path(model_path) if model_path else DEFAULT_FRIDA_PTH
    if not target.is_absolute():
        target = ROOT / target
    if target.is_file():
        return target

    zip_path = DEFAULT_FRIDA_ZIP
    if not zip_path.is_file() or zip_path.stat().st_size < 10_000:
        raise FileNotFoundError(
            f"Frida weights missing at {target} and zip unusable at {zip_path}. "
            "Run: git lfs pull && unzip models/Frida.zip -d models/_tmp/frida/"
        )

    out_dir = target.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[model] Extracting {zip_path.name} → {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # Zip may dump files at out_dir or in a nested folder
    if target.is_file():
        return target
    found = list(out_dir.rglob("frida.pth"))
    if not found:
        raise FileNotFoundError(f"frida.pth not found after unzipping {zip_path}")
    if found[0] != target:
        # Keep canonical path used by config
        if not target.exists():
            target.write_bytes(found[0].read_bytes())
    return target


def ensure_hubert_weights(*, download: bool = True) -> Path:
    """
    Ensure Transformers ContentVec dir exists.

    Preference order:
      1. ``models/hubert/contentvec_hf/`` (already converted)
      2. Convert from ``models/hubert/hubert_base.pt`` (lj1995)
      3. Download hubert_base.pt then convert
    """
    cfg = HUBERT_HF_DIR / "config.json"
    if cfg.is_file():
        return HUBERT_HF_DIR

    if not HUBERT_PT.is_file():
        if not download:
            raise FileNotFoundError(
                f"HuBERT/ContentVec missing. Download:\n  {HUBERT_URL}\n"
                f"to {HUBERT_PT} (see README)."
            )
        HUBERT_PT.parent.mkdir(parents=True, exist_ok=True)
        print(f"[hubert] Downloading ContentVec → {HUBERT_PT}")
        urllib.request.urlretrieve(HUBERT_URL, HUBERT_PT)

    print(f"[hubert] Converting fairseq ContentVec → {HUBERT_HF_DIR}")
    _convert_contentvec_pt_to_hf(HUBERT_PT, HUBERT_HF_DIR)
    return HUBERT_HF_DIR


def _install_fairseq_dictionary_stub() -> None:
    """Allow torch.load of lj1995 hubert_base.pt without installing fairseq."""
    if "fairseq.data.dictionary" in sys.modules:
        return
    fairseq = types.ModuleType("fairseq")
    fairseq_data = types.ModuleType("fairseq.data")
    fairseq_data_dictionary = types.ModuleType("fairseq.data.dictionary")

    class Dictionary:  # noqa: D401 — pickle target only
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    fairseq_data_dictionary.Dictionary = Dictionary  # type: ignore[attr-defined]
    fairseq.data = fairseq_data  # type: ignore[attr-defined]
    fairseq_data.dictionary = fairseq_data_dictionary  # type: ignore[attr-defined]
    sys.modules["fairseq"] = fairseq
    sys.modules["fairseq.data"] = fairseq_data
    sys.modules["fairseq.data.dictionary"] = fairseq_data_dictionary


def _convert_contentvec_pt_to_hf(pt_path: Path, out_dir: Path) -> None:
    """Map fairseq ContentVec state_dict → transformers HubertModel (+ final_proj)."""
    from torch import nn
    from transformers import HubertConfig, HubertModel

    _install_fairseq_dictionary_stub()
    cpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    fair_sd = cpt["model"]

    class HubertModelWithFinalProj(HubertModel):
        def __init__(self, config: HubertConfig) -> None:
            super().__init__(config)
            self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    hubert = HubertModelWithFinalProj(HubertConfig())
    mapping: dict[str, str] = {
        "masked_spec_embed": "mask_emb",
        "encoder.layer_norm.bias": "encoder.layer_norm.bias",
        "encoder.layer_norm.weight": "encoder.layer_norm.weight",
        "encoder.pos_conv_embed.conv.bias": "encoder.pos_conv.0.bias",
        "encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv.0.weight_g",
        "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv.0.weight_v",
        "feature_projection.layer_norm.bias": "layer_norm.bias",
        "feature_projection.layer_norm.weight": "layer_norm.weight",
        "feature_projection.projection.bias": "post_extract_proj.bias",
        "feature_projection.projection.weight": "post_extract_proj.weight",
        "final_proj.bias": "final_proj.bias",
        "final_proj.weight": "final_proj.weight",
    }
    for layer in range(12):
        for j in ("q", "k", "v"):
            mapping[f"encoder.layers.{layer}.attention.{j}_proj.weight"] = (
                f"encoder.layers.{layer}.self_attn.{j}_proj.weight"
            )
            mapping[f"encoder.layers.{layer}.attention.{j}_proj.bias"] = (
                f"encoder.layers.{layer}.self_attn.{j}_proj.bias"
            )
        mapping[f"encoder.layers.{layer}.final_layer_norm.bias"] = (
            f"encoder.layers.{layer}.final_layer_norm.bias"
        )
        mapping[f"encoder.layers.{layer}.final_layer_norm.weight"] = (
            f"encoder.layers.{layer}.final_layer_norm.weight"
        )
        mapping[f"encoder.layers.{layer}.layer_norm.bias"] = (
            f"encoder.layers.{layer}.self_attn_layer_norm.bias"
        )
        mapping[f"encoder.layers.{layer}.layer_norm.weight"] = (
            f"encoder.layers.{layer}.self_attn_layer_norm.weight"
        )
        mapping[f"encoder.layers.{layer}.attention.out_proj.bias"] = (
            f"encoder.layers.{layer}.self_attn.out_proj.bias"
        )
        mapping[f"encoder.layers.{layer}.attention.out_proj.weight"] = (
            f"encoder.layers.{layer}.self_attn.out_proj.weight"
        )
        mapping[f"encoder.layers.{layer}.feed_forward.intermediate_dense.bias"] = (
            f"encoder.layers.{layer}.fc1.bias"
        )
        mapping[f"encoder.layers.{layer}.feed_forward.intermediate_dense.weight"] = (
            f"encoder.layers.{layer}.fc1.weight"
        )
        mapping[f"encoder.layers.{layer}.feed_forward.output_dense.bias"] = (
            f"encoder.layers.{layer}.fc2.bias"
        )
        mapping[f"encoder.layers.{layer}.feed_forward.output_dense.weight"] = (
            f"encoder.layers.{layer}.fc2.weight"
        )
    for layer in range(7):
        mapping[f"feature_extractor.conv_layers.{layer}.conv.weight"] = (
            f"feature_extractor.conv_layers.{layer}.0.weight"
        )
        if layer == 0:
            mapping[f"feature_extractor.conv_layers.{layer}.layer_norm.weight"] = (
                f"feature_extractor.conv_layers.{layer}.2.weight"
            )
            mapping[f"feature_extractor.conv_layers.{layer}.layer_norm.bias"] = (
                f"feature_extractor.conv_layers.{layer}.2.bias"
            )

    new_sd = {hk: fair_sd[fk] for hk, fk in mapping.items() if fk in fair_sd}
    hubert.load_state_dict(new_sd, strict=False)
    hubert.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    hubert.save_pretrained(out_dir)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    import librosa

    return librosa.resample(
        np.asarray(audio, dtype=np.float32),
        orig_sr=orig_sr,
        target_sr=target_sr,
    ).astype(np.float32)


def _f0_pm(audio_16k: np.ndarray, p_len: int, f0_up_key: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Parselmouth pitch → coarse + continuous F0 (RVC convention)."""
    import parselmouth

    time_step = WINDOW / HUBERT_SR
    f0_min, f0_max = 50.0, 1100.0
    f0 = (
        parselmouth.Sound(audio_16k, HUBERT_SR)
        .to_pitch_ac(
            time_step=time_step,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        .selected_array["frequency"]
    )
    pad_size = (p_len - len(f0) + 1) // 2
    if pad_size > 0 or p_len - len(f0) - pad_size > 0:
        f0 = np.pad(f0, [[pad_size, max(0, p_len - len(f0) - pad_size)]], mode="constant")
    f0 = f0[:p_len]
    f0 = f0 * (2.0 ** (f0_up_key / 12.0))
    f0bak = f0.copy()
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int32)
    return f0_coarse, f0bak.astype(np.float32)


class RVCModel:
    """Loaded RVC V2 generator + ContentVec encoder."""

    def __init__(
        self,
        model_path: Path,
        *,
        index_path: Optional[Path] = None,
        device: Optional[str] = None,
        index_rate: float = 0.5,
        f0_up_key: int = 0,
        protect: float = 0.33,
        download_hubert: bool = True,
    ) -> None:
        self.device = pick_torch_device(device)
        self.index_rate = float(index_rate)
        self.f0_up_key = int(f0_up_key)
        self.protect = float(protect)
        self.model_path = Path(model_path)
        self.loaded = False
        self.version = "v2"
        self.if_f0 = 1
        self.tgt_sr = 40_000
        self.sid = 0
        self.net_g: Any = None
        self.hubert: Any = None
        self.index = None
        self.big_npy = None

        hubert_dir = ensure_hubert_weights(download=download_hubert)
        from torch import nn
        from transformers import HubertConfig, HubertModel

        class HubertModelWithFinalProj(HubertModel):
            def __init__(self, config: HubertConfig) -> None:
                super().__init__(config)
                self.final_proj = nn.Linear(
                    config.hidden_size, config.classifier_proj_size
                )

        print(f"[hubert] Loading ContentVec from {hubert_dir} on {self.device}")
        self.hubert = (
            HubertModelWithFinalProj.from_pretrained(str(hubert_dir))
            .to(self.device)
            .eval()
        )

        from .rvc_pack.models import (
            SynthesizerTrnMs256NSFsid,
            SynthesizerTrnMs256NSFsid_nono,
            SynthesizerTrnMs768NSFsid,
            SynthesizerTrnMs768NSFsid_nono,
        )

        print(f"[model] Loading RVC checkpoint {self.model_path}")
        cpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.tgt_sr = int(cpt["config"][-1])
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.if_f0 = int(cpt.get("f0", 1))
        self.version = str(cpt.get("version", "v1"))
        n_spk = cpt["weight"]["emb_g.weight"].shape[0]

        if self.version == "v1":
            net_g = (
                SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=False)
                if self.if_f0
                else SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            )
        else:
            net_g = (
                SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
                if self.if_f0
                else SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            )
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g = net_g.float()
        net_g.remove_weight_norm()
        del net_g.enc_q  # training-only posterior; not used in infer()
        self.net_g = net_g.eval().to(self.device)
        self.sid = 0 if n_spk < 1 else 0
        print(
            f"[model] Ready version={self.version} f0={self.if_f0} "
            f"sr={self.tgt_sr} n_spk={n_spk} device={self.device}"
        )

        idx = index_path
        if idx is None:
            cand = self.model_path.with_suffix(".index")
            sibling = DEFAULT_FRIDA_INDEX
            if cand.is_file():
                idx = cand
            elif sibling.is_file():
                idx = sibling
            else:
                matches = list(self.model_path.parent.glob("*.index"))
                idx = matches[0] if matches else None
        if idx and Path(idx).is_file() and self.index_rate > 0:
            try:
                import faiss

                self.index = faiss.read_index(str(idx))
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                print(f"[index] Loaded {idx} (rate={self.index_rate})")
            except Exception as exc:  # noqa: BLE001
                print(f"[index] Skipping ({exc})")
                self.index = None
                self.big_npy = None

        self.loaded = True

    @torch.inference_mode()
    def convert(
        self,
        audio: np.ndarray,
        *,
        input_sr: int = HUBERT_SR,
    ) -> tuple[np.ndarray, int]:
        """
        Convert mono float audio.

        Parameters
        ----------
        audio:
            Mono waveform. Prefer 16 kHz; other rates are resampled to 16 kHz first.
        input_sr:
            Sample rate of ``audio``.

        Returns
        -------
        (waveform, tgt_sr) at the RVC model sample rate.
        """
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x, self.tgt_sr
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1.0:
            x = x / peak

        audio_16k = _resample(x, input_sr, HUBERT_SR)
        # light DC / rumble cut like upstream
        from scipy import signal as sp_signal

        bh, ah = sp_signal.butter(N=5, Wn=48, btype="high", fs=HUBERT_SR)
        audio_16k = sp_signal.filtfilt(bh, ah, audio_16k).astype(np.float32)

        feats = torch.from_numpy(audio_16k).float().view(1, -1).to(self.device)
        out = self.hubert(input_values=feats, output_hidden_states=True)
        # RVC v2 uses encoder layer 12; transformers hidden_states[12] matches.
        layer = 12 if self.version == "v2" else 9
        phone = out.hidden_states[layer]
        if self.version == "v1" and hasattr(self.hubert, "final_proj"):
            phone = self.hubert.final_proj(phone)

        if self.index is not None and self.big_npy is not None and self.index_rate != 0:
            # faiss-cpu IVF search has segfaulted on some macOS/numpy stacks —
            # keep index optional; callers can set index_rate=0 to skip entirely.
            npy = np.ascontiguousarray(phone[0].float().detach().cpu().numpy(), dtype=np.float32)
            try:
                if hasattr(self.index, "nprobe"):
                    self.index.nprobe = max(int(getattr(self.index, "nprobe", 1) or 1), 1)
                score, ix = self.index.search(npy, k=8)
            except Exception as exc:  # noqa: BLE001
                print(f"[index] search failed, disabling ({exc})", flush=True)
                self.index = None
                self.big_npy = None
            else:
                weight = np.square(1.0 / (score + 1e-8))
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                phone = (
                    torch.from_numpy(npy).unsqueeze(0).to(self.device) * self.index_rate
                    + (1.0 - self.index_rate) * phone
                )

        # ~50 Hz → ~100 Hz content rate
        phone = F.interpolate(phone.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        p_len = audio_16k.shape[0] // WINDOW
        if phone.shape[1] < p_len:
            p_len = phone.shape[1]
        phone = phone[:, :p_len, :]
        phone_lengths = torch.tensor([p_len], device=self.device).long()
        sid = torch.tensor([self.sid], device=self.device).long()

        if self.if_f0 == 1:
            pitch, pitchf = _f0_pm(audio_16k, p_len, self.f0_up_key)
            pitch_t = torch.from_numpy(pitch[:p_len]).unsqueeze(0).long().to(self.device)
            pitchf_t = (
                torch.from_numpy(pitchf[:p_len]).unsqueeze(0).float().to(self.device)
            )
            audio_out = (
                self.net_g.infer(phone, phone_lengths, pitch_t, pitchf_t, sid)[0][0, 0]
                .float()
                .cpu()
                .numpy()
            )
        else:
            audio_out = (
                self.net_g.infer(phone, phone_lengths, sid)[0][0, 0]
                .float()
                .cpu()
                .numpy()
            )

        audio_out = np.asarray(audio_out, dtype=np.float32)
        peak_o = float(np.max(np.abs(audio_out))) if audio_out.size else 0.0
        if peak_o > 1.0:
            audio_out = audio_out / peak_o * 0.99
        return audio_out, self.tgt_sr


def load_rvc_v2(
    model_path: str | Path,
    *,
    index_path: Optional[str | Path] = None,
    device: Optional[str] = None,
    download_hubert: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load Frida (or any V2 .pth) + ContentVec; return a dict for ``infer_rvc_v2``."""
    path = ensure_frida_pth(Path(model_path))
    idx = Path(index_path) if index_path else None
    engine = RVCModel(
        path,
        index_path=idx,
        device=device,
        download_hubert=download_hubert,
        **kwargs,
    )
    return {
        "path": str(path),
        "loaded": True,
        "exists": True,
        "engine": engine,
        "tgt_sr": engine.tgt_sr,
        "device": str(engine.device),
        "version": engine.version,
    }


def infer_audio(
    audio_block: np.ndarray,
    model: dict[str, Any],
    sample_rate: int,
) -> np.ndarray:
    """Run conversion; return float32 mono at ``model['tgt_sr']`` (may differ in length)."""
    engine: RVCModel = model["engine"]
    out, _sr = engine.convert(audio_block, input_sr=sample_rate)
    return out
