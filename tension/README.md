# Tension

PC-first local **two-player** social deduction prototype. Loop adapted from *Death Note: Killer Within*: **Investigate → Deduce → Eliminate**.

Skill and observation over luck: no random role assignment — Kira **chooses** a secret cover identity; clues are always true traits of that cover.

No networking. Placeholder UI. Structured so game logic (`scripts/core/`) stays separate from scenes (`scripts/ui/`) for a later phone port.

## Requirements

- **Godot 4.2+** (tested against 4.2 / 4.3 project features)
- Desktop / PC

## How to launch

1. Install [Godot 4](https://godotengine.org/download) (Standard / GDScript build).
2. Open Godot → **Import** → select this folder’s `project.godot`  
   - Primary path on this machine: `~/Desktop/tension/project.godot`  
   - Repo mirror (for PR / version control): `tension/project.godot` at the repo root
3. Open the project, then press **F5** (or **Play**).  
   Main scene: `scenes/main_menu.tscn` (set in `project.godot`).

## How to play (pass-and-play)

1. **Main menu** → New Match.
2. **Role select**: L looks away. Kira secretly picks one of four covers (Aya / Ren / Kai / Mio), then hands off.
3. Each round:
   - **Investigate (Kira)**: leave one true trait trace (advances New World bar).
   - **Investigate (L)**: search Lab or Hall (advances Investigation bar; true zone hits harder).
   - **Deduce**: shared evidence board — narrow who still matches all traces.
   - **Eliminate**: L arrests a cover. Correct = L wins. Wrong = New World spike + next day.
4. Win by arrest, or by filling Investigation (L) / New World (Kira) to 100%. Max 3 rounds.

## Killer Within → prototype map

| Killer Within | Prototype |
| --- | --- |
| Investigation day (tasks, Note, ID theft) | `scenes/investigate.tscn` + `scripts/ui/investigate.gd` → `GameState.kira_leave_trace` / `l_search_zone` |
| Discussion / progress report | `scenes/deduce.tscn` + `scripts/ui/deduce.gd` |
| Vote / arrest | `scenes/eliminate.tscn` + `scripts/ui/eliminate.gd` → `GameState.arrest` |
| Team L / Team Kira win | `scenes/result.tscn` + bars in `scripts/core/rules.gd` |
| Role assignment | `scenes/role_select.tscn` — **player-chosen** secret cover (not RNG) |

## Layout (mobile-ready later)

```
tension/
  project.godot
  scenes/           # UI only
  scripts/
    core/           # GameState autoload, PhaseMachine, Rules, Suspects — no UI
    ui/             # Scene controllers
  assets/
```

Do **not** put presentation logic in `scripts/core/`. A future mobile shell can swap scenes and keep the same autoload API.
