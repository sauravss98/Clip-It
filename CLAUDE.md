# CLAUDE.md — game-clipper
 
## What this project is
 
`game-clipper` is an open-source Python CLI that automatically finds action moments in
**no-commentary gaming videos** and exports them as clips, ready to be edited into YouTube
Shorts.
 
It exists because most "AI clippers" rely on speech (transcripts + LLMs) and fail on
gameplay with no voice. This tool works from **game audio alone**: fights, kills,
gunshots, explosions and crashes are louder and more percussive than looting, walking or
menus.
 
The owner runs a faceless YouTube gaming channel and will publish this repo publicly on
GitHub. Treat it as a real open-source project: clean code, tests, CI, a good README.
 
## Owner's environment
 
- Windows 11 laptop, NVIDIA RTX 3050 Ti (NVENC available), plus a PS5.
- Comfortable with Python and TypeScript. Prefer Python for this project.
- ffmpeg may need installing: `winget install Gyan.FFmpeg`.
- Paths often contain spaces (e.g. `C:\Users\<name>\Videos\...`). Always handle them.
## Supported games and footage
 
Games (each gets a **preset**): `generic` (default), `fortnite`, `league`, `r6siege`,
`gta5_online`, `gta6`. Story/open-world games such as RDR2, Witcher 3 and Ghost of Yōtei
should work with `generic` or a story preset later.
 
Footage sources:
- OBS recordings (MP4 or MKV), 1080p60 or 1440p60, 1–3 hours long.
- Downloaded YouTube stream VODs.
- PS5 captures or stream downloads (GTA 6): may be 4K, HDR or variable frame rate.
There is **no commentary**. Only game audio (and sometimes in-game music or radio).
 
## How detection works (v1: audio only)
 
Never load the full video or audio into memory. Stream ffmpeg output and parse it.
 
1. **Extract loudness timeline** with ffmpeg: the `ebur128` short-term loudness, or RMS
   per 0.5 s window via `astats`/`astats=metadata=1:reset=…` + `ametadata=print`.
2. **Baseline**: rolling median or percentile per video, so detection adapts to each
   recording's overall volume.
3. **Two scores per window**, combined with per-preset weights:
   - **Sustained score**: loudness above baseline (good for Fortnite and League fights).
   - **Transient/onset score**: sudden jumps vs the previous few seconds (essential for
     R6 Siege: quiet tactical play, then sharp gunfire or breaches).
4. **Music/radio suppression** (GTA 5 Online, GTA 6): car radio, music and cutscenes are
   loud but are not highlights.
   - Penalise events whose loudness is steady and long.
   - Optional lightweight spectral feature (spectral flatness or crest factor via numpy on
     downsampled audio) to separate percussive bursts from continuous music.
5. **Events**: merge flagged windows that are close together, add padding (default 6 s
   before, 4 s after), enforce min/max length (default 8 s / 60 s; story presets allow
   90 s).
6. **Rank**: score each event (peak, sustained intensity, transient strength, duration)
   and keep the top N (default 20), then output in chronological order.
### Detector architecture
 
Detectors are **pluggable** behind one interface (e.g. a `Detector` protocol returning
`list[Event]` with start, end, score and reasons). v1 implements only `LoudnessDetector`.
Keep the interface ready for future detectors (see Roadmap). No game-specific code paths
in v1: presets only change parameters and weights.
 
## CLI
 
```
game-clipper detect VIDEO [options]   # detect events, print a table, write events.json
game-clipper clip   VIDEO [options]   # detect + export clips
game-clipper tune   VIDEO [options]   # timeline summary (+ optional PNG plot) for tuning
game-clipper train  [--preset --out]  # train the user's model from 👍/👎 ratings
game-clipper ui     [--port --out]    # local drag-and-drop web UI (stdlib http.server)
```
 
Common options:
`--out DIR`, `--preset {generic,fortnite,league,r6siege,gta5_online,gta6}`,
`--config FILE`, `--top N`, `--threshold`, `--pad-before`, `--pad-after`,
`--min-len`, `--max-len`, `--vertical`, `--reencode` / `--copy`, `--audio-track N`,
`--vision` / `--no-vision`, `--text` / `--no-text`, `--player NAME`,
`--dry-run`, `--verbose`.
 
- Presets live in a config file (TOML) shipped with the package. Users can override them
  with `--config`. CLI flags override both.
- `tune` prints loudness, transient and score over time, explains why moments were or
  weren't picked, and optionally saves a PNG plot (matplotlib as an **optional extra**:
  `pip install game-clipper[plot]`).
- `ui` (added at the owner's request): local web page on 127.0.0.1 to upload a video or
  paste a local path, pick preset/format, and preview/download clips. No extra
  dependencies; jobs run one at a time on a worker thread. The CLI and the UI share
  `pipeline.py` (probe -> analyse -> export -> reports); keep logic there, not in either UI.
- Vision (added at the owner's request; **local only, no paid/online AI APIs**): optional
  `[vision]` extra (torch + open_clip). Audio proposes 3x top-N candidates, `vision/rerank.py`
  samples frames, CLIP zero-shot drops menus/lobbies/results screens, and a numpy logistic
  regression trained on the owner's 👍/👎 (`labels.json` per output folder, features cached
  in `vision.npz`) re-ranks. Models live in `%LOCALAPPDATA%/game-clipper/models/<preset>.npz`
  (override `GAME_CLIPPER_HOME`). Evaluation is leave-one-video-out vs the audio baseline.
  Tests use a fake embedder; CI never installs torch. The trained model's weight scales
  with `HighlightModel.trust` (held-out AUC gain over audio, >=2 recordings).
- On-screen text (owner's request, after audio missed a triple kill/victory in a stream
  VOD): `detectors/text.py` `TextDetector` (optional `[ocr]` extra, RapidOCR on CPU)
  reads per-preset `text_regions` every `text_interval` s, fuzzy-matches `text_events`
  phrases, groups re-reads of the same line, and emits events with a `boost` score floor.
  `player_name` gives the owner's own kills a bonus. Text events are merged with audio
  candidates in `pipeline.analyse` and never rejected by the vision menu filter.
## Export
 
- **Default `--reencode`**: frame-accurate cuts with `h264_nvenc`. Fall back to
  `libx264` automatically if NVENC isn't available. Keep the original resolution and fps.
- **`--copy`**: fast stream-copy cutting, accurate only to keyframes. Document the
  trade-off.
- **`--vertical`**: 1080×1920 output for Shorts. A blurred, zoomed copy of the video
  fills the background, with the original gameplay scaled to fit and centred on top
  (single ffmpeg `filter_complex`).
- HDR sources: tone-map to SDR when re-encoding (document it as best-effort).
- Clip names: `0012_01h23m45s_score87.mp4` (index, source timestamp, score).
- Output folder also contains:
  - `events.json`: every event with times, scores, and reasons.
  - `index.html`: a simple self-contained review page listing each clip with a
    `<video>` preview, timestamp and score, so the owner can review in a browser.
## Project layout
 
```
game-clipper/
├─ pyproject.toml
├─ README.md
├─ LICENSE                  # MIT
├─ CLAUDE.md
├─ .gitignore               # exclude videos, output/, *.mp4, *.mkv, .venv
├─ src/game_clipper/
│  ├─ __init__.py
│  ├─ cli.py                # typer app: detect / clip / tune / ui
│  ├─ pipeline.py           # analyse -> export -> reports, shared by CLI and UI
│  ├─ config.py             # preset + config loading and merging
│  ├─ presets.toml
│  ├─ ffmpeg.py             # ffmpeg/ffprobe checks, probing, streaming helpers
│  ├─ audio.py              # loudness/transient/spectral feature extraction
│  ├─ detectors/
│  │  ├─ base.py            # Detector protocol, Event dataclass
│  │  ├─ loudness.py
│  │  └─ text.py            # OCR of kill feed / banners
│  ├─ events.py             # merge, pad, clamp, score, rank (pure functions)
│  ├─ export.py             # cutting, NVENC fallback, vertical filter
│  ├─ report.py             # events.json + index.html
│  ├─ training.py           # ratings -> dataset -> model
│  ├─ vision/               # frames, embedder (CLIP), scenes (zero-shot), rerank, model
│  └─ web/                  # server.py (stdlib HTTP + job queue), static/index.html
└─ tests/
   ├─ conftest.py           # synthetic video/audio generation with ffmpeg
   ├─ test_events.py
   ├─ test_detector.py
   ├─ test_export.py
   ├─ test_config.py
   ├─ test_cli.py
   ├─ test_vision.py
   ├─ test_training.py
   ├─ test_text.py
   └─ test_web.py
```
 
## Engineering rules
 
- Python 3.11+. Package with `pyproject.toml` and expose the `game-clipper` command.
  Install for development with `pip install -e ".[dev]"`.
- Minimal dependencies: `typer`, `rich`, `numpy`. `matplotlib` only as an optional extra.
  Use stdlib `tomllib` for config.
- On startup, check that `ffmpeg` and `ffprobe` are on PATH and give a clear error with
  install hints for Windows, macOS and Linux.
- Run ffmpeg via `subprocess` with argument lists (never shell strings). This keeps paths
  with spaces safe.
- Keep the merge, pad, score and rank logic as **pure functions** so it's easy to test.
- Show progress with `rich` for long videos (ffmpeg `-progress pipe:1`).
- Type hints everywhere. Docstrings on public functions.
- Lint and format with **ruff**. Tests with **pytest**.
## Tests
 
Generate short synthetic media with ffmpeg in fixtures (no binary files committed):
- (a) Quiet tone with a few short loud bursts at known timestamps (R6-like). The detector
  must find all of them within tolerance.
- (b) A long steady loud section (music-like) plus short bursts. The bursts must rank
  **above** the steady section.
- (c) Mixed content. Sanity-check counts and ordering.
- Unit tests for merge, padding, clamping, min/max length and top-N.
- A small export test: cut a clip and verify its duration with ffprobe; verify the
  vertical output is 1080×1920.
## CI
 
GitHub Actions workflow: ruff + pytest on `ubuntu-latest` and `windows-latest`, Python
3.11 and 3.12, installing ffmpeg in each job. The NVENC path must be skipped or fall back
cleanly in CI (no GPU there).
 
## README must include
 
- What it does and why (built for no-commentary / faceless gaming channels).
- A GIF or screenshot placeholder of the `index.html` review page.
- Install (Windows first, then macOS/Linux), including ffmpeg.
- Usage examples for each command and preset.
- How detection works (sustained + transient + music suppression), in plain language.
- Tuning tips per game, including: **turn off in-game radio/music in GTA when
  recording**, since car radio is the biggest source of false clips.
- Roadmap (below). MIT license.
## Roadmap (do not build in v1 unless asked)
 
- **League of Legends detector** using the official Live Client Data API
  (`https://127.0.0.1:2999/liveclientdata/eventdata`): real `ChampionKill`, `Multikill`
  and `Ace` events. Possibly a live mode that triggers OBS Replay Buffer saves via
  OBS WebSocket.
- Kill-feed OCR / vision detectors (Fortnite "Eliminated", R6 kill feed).
- Auto-captions / text hooks burned into vertical clips.
- Packaged desktop app (the local web UI exists; see `ui`).
## Workflow for Claude
 
1. Before building, briefly propose the structure and detection algorithm, then proceed.
2. Build in small steps with meaningful git commits (conventional commit style).
3. Run ruff and pytest before saying a step is done.
4. After the tests pass, ask the owner for the path to a real recording and run
   `game-clipper detect` / `tune` on it to tune the default thresholds for that game's
   preset. Never commit the owner's videos or output clips.
5. When v1 works, help create a public GitHub repo with `gh` and push.
6. Keep this file up to date when architecture or commands change.