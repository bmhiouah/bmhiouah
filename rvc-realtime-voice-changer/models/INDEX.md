# RVC V2 English feminine models

Free community weights downloaded 2026-11-24 (Africa/Casablanca).  
Each archive was integrity-checked (`unzip -t`) and confirmed to contain at least one `.pth` (and usually a `.index`). **No inference was run.**

| Display name | Archive (exact filename) | Size (bytes) | Size (human) | Contained weights | Source |
| --- | --- | --- | --- | --- | --- |
| Girl Voice (RMVPE) | `GirlVoice.zip` | 345640428 | 330M | `GirlVoice/Quumsy_e300_s38400.pth`, `GirlVoice/added_IVF3823_Flat_nprobe_1_Quumsi_v2.index` | [HF Quumsy/VoiceModels](https://huggingface.co/Quumsy/VoiceModels/resolve/main/GirlVoice.zip) |
| E-Girl | `EGirl.zip` | 221389729 | 212M | `egirl.pth`, `added_IVF2182_Flat_nprobe_1_egirl.index` | [HF pendmg/Models](https://huggingface.co/pendmg/Models/resolve/main/egirl.zip) |
| Female Pop Star | `FemalePopStar.zip` | 78647527 | 76M | `model.pth`, `model.index` | [HF MysticWoIf/Voice-Model-archive](https://huggingface.co/MysticWoIf/Voice-Model-archive/resolve/main/Female%20Power%20Pop%20Vocals%20-%20Weights%20Model.zip) |
| Camila | `Camila.zip` | 70731497 | 68M | `camila2/camila2_200e_1400s.pth`, `camila2/camila2.index` | [HF iznalec/rvc](https://huggingface.co/iznalec/rvc/resolve/main/camila2.zip) |
| Furina English | `FurinaEN.zip` | 82566118 | 79M | `furina-EN_v2_48k_e60/Furina_EN_e60_v2_48k.pth`, `furina-EN_v2_48k_e60/Furina_EN_v2.index` | [HF biggyg/Furina-English-RVC-v2-48k](https://huggingface.co/biggyg/Furina-English-RVC-v2-48k/resolve/main/furina-EN_v2_48k_e60.zip) |
| Phrolova EN | `PhrolovaEN.zip` | 104664031 | 100M | `Phrolova.pth`, `Phrolova.index` | [HF dmKaltsit/Phrolova](https://huggingface.co/dmKaltsit/Phrolova/resolve/main/PhrolovaEN.zip) |
| Frida (Hilda) EN | `Frida.zip` | 151379263 | 145M | `frida.pth`, `added_IVF780_Flat_nprobe_1_frida_v2.index` | [HF SirWildernessForestLiver/Frida_Hilda](https://huggingface.co/SirWildernessForestLiver/Frida_Hilda/resolve/main/frida.zip) |

**Total archive bytes:** 1,055,118,593 (~1006M).

## Usage

```bash
# Example: extract one model
unzip models/FurinaEN.zip -d models/_tmp/
export RVC_MODEL_PATH=models/_tmp/furina-EN_v2_48k_e60/Furina_EN_e60_v2_48k.pth
```

Large zips are stored with **Git LFS**. After clone: `git lfs pull`.

Character / celebrity likeness may still have IP limits — see each publisher card.
