# P5R Generative Social Links

> *What if Ryuji actually remembered what happened last time you hung out?*

A [Reloaded-II](https://reloaded-project.github.io/Reloaded-II/) mod that replaces Persona 5 Royal's scripted Social Link dialogue with live AI generation — running entirely on your local GPU. No cloud, no API keys, no internet. Every conversation is unique.

[![CI](https://github.com/SrijanSuresh/p5r-gen-social-links/actions/workflows/ci.yml/badge.svg)](https://github.com/SrijanSuresh/p5r-gen-social-links/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-119%20passing-brightgreen)
![.NET](https://img.shields.io/badge/.NET-8.0-512BD4)
![Python](https://img.shields.io/badge/Python-3.13-3776AB)
![Platform](https://img.shields.io/badge/platform-Windows-0078D4)

---

## For people who haven't touched a compiler

Persona 5 Royal has 22 Social Link characters — Ryuji, Ann, Makoto, and so on. Every line they say is pre-written, fixed forever in the game's script. This mod hooks into the game's memory while it's running, reads who you're hanging out with and how close you are, sends that to a small AI model running on your own PC, and gets back a reply that sounds like that character at that moment in your relationship.

**Ryuji at Rank 4, gym hang-out, confirmed output:**
> *"Dude, what's up? I was thinkin' we could grab some ramen before we head back to..."*

It generates in his voice, at the right emotional register for where you are in the friendship, without ever leaving your machine.

---

## Architecture

```mermaid
flowchart LR
    subgraph game ["P5R.exe (game process)"]
        hook["CMM_EXEC_EVENT\nhook"]
        counter["Line counter\n0x006FFC28"]
        pool["Script text pool\npre-loaded heap"]
        renderer["Dialogue renderer"]
    end

    subgraph mod ["C# mod  ·  Reloaded-II DLL"]
        chain["PointerChainResolver\nASLR-aware chain walk"]
        reader["SocialLinkReader\nconfidant · rank · scene"]
        monitor["LineCounterMonitor\nper-line trigger"]
        bridge["DialogueBridge\nthrottle · history · dedup"]
        writer["DialogueWriter\nwrite-back stub"]
    end

    subgraph server ["Python server  ·  localhost:8765"]
        api["FastAPI  /generate"]
        queue["InferenceQueue\nsingle-slot drop-on-busy"]
        pipe["InferencePipeline\nllama-cpp-python"]
        model["Llama-3.1-8B-Instruct\nQ4_K_M · 4-bit quantized"]
    end

    hook -->|fires per hang-out| chain
    counter -->|increments per line| monitor
    chain --> reader --> bridge
    monitor --> bridge
    bridge -->|"POST /generate\nconfidant · rank · context"| api
    api --> queue --> pipe --> model
    model -->|generated text| bridge
    bridge -.->|"write-back (Phase 3)"| writer
    writer -.-> pool -.-> renderer

    style writer stroke-dasharray: 5 5
    style pool stroke-dasharray: 5 5
    style renderer stroke-dasharray: 5 5
```

The two processes communicate over localhost HTTP. The C# mod lives inside P5R's process; the Python server is a separate process started before the game. If the server isn't up, the mod falls back to scripted dialogue silently.

### Component summary

| Component | Language | What it does |
|---|---|---|
| `PointerChainResolver` | C# | Walks `[moduleBase + 0x2A63EF0] → [CMM + 0x48]` to find the active session struct, VirtualQuery-guarding every dereference |
| `SocialLinkReader` | C# | Reads confidant ID, rank level, and scene number from the resolved struct |
| `LineCounterMonitor` | C# | Polls `0x006FFC28` (discovered via Cheat Engine) — fires on each dialogue line advance |
| `DialogueBridge` | C# | Leading-edge throttle (3s), rolling session history (8 entries), hash-based dedup, context budget management |
| `DialogueWriter` | C# | Write-back stub — ready for when the text pool pointer chain is confirmed via Ghidra |
| `InferencePipeline` | Python | llama-cpp-python wrapper; builds character-faithful prompts with rank-tier emotional guidance |
| `InferenceQueue` | Python | Single-slot async queue — drops concurrent requests rather than queuing stale dialogue |
| `postprocess.py` | Python | Strips OOC commentary, name-prefix artifacts, truncates at sentence boundaries |

---

## Project phases

### ✅ Phase 1 — Memory scaffold
Reverse-engineered the CMM session struct layout from Ghidra + live hex dumps. Established the ASLR-aware pointer chain. Confirmed confidant ID, rank, and scene number reading in-game.

### ✅ Phase 2 — LLM inference
Switched backend from auto-gptq to llama-cpp-python. Wired real Llama-3.1-8B-Instruct (Q4_K_M) inference through a FastAPI server. Confirmed end-to-end: game hang-out → C# hook → HTTP POST → GPU inference → generated dialogue logged. 54 commits, 119 automated tests.

**Confirmed working:**
- All 22 Social Link characters recognised by the inference pipeline
- Real inference latency: <2s on RTX 4060 after CUDA warmup
- Per-line trigger: `LineCounterMonitor` fires on each dialogue advance
- CI: GitHub Actions runs Python tests + .NET build on every push

### 🔄 Phase 3 — Dialogue write-back *(in progress)*

#### 🏆 Milestone: injected text rendered in-game

Custom text written into the live dialogue buffer renders in Ryuji's speech bubble, drawn by the game's own renderer:

> **Ryuji** — *"Here we are... LLM WAS HERE!!!"*

This closes the central open question of the project. Everything downstream of locating the buffer is confirmed working: the memory is writable at runtime, the renderer reads it live, and injected text displays verbatim with correct font, styling, and speaker attribution.

**What the hunt established:**

| Finding | Detail |
|---|---|
| Encoding | **ASCII**, single-byte — not UTF-16 |
| Location | **Heap** (`0x41DD7F6389`, `0x42102CAAA9`) — not the mapped BMD file region |
| Buffers | Two hold the live line; both accept writes |
| Delivery | Renderer reads the buffer **in place** — the text is never `memcpy`'d, so there is no copy to intercept |

Both facts in the first two rows were needed together, and searching either dimension alone finds nothing. Automated scans covered ASCII in the mapped-file region and UTF-16 in the heap, and so missed the text repeatedly; the address was ultimately pinned by a Cheat Engine string scan against the line visible on screen. Chapters 53–60 of `learning.md` document each wrong assumption and how it was ruled out.

**Remaining for the phase:** hook the instruction that writes the buffer, so the address is resolved from the game rather than searched for. The buffer address changes every launch, but the writer instruction sits at a fixed module offset — hooking it yields the destination pointer directly, with no scanning or session-struct dependency.

### ⏳ Phase 4 — Per-line contextual generation
Use the `LineCounterMonitor` trigger and `SessionHistory` rolling buffer to generate dialogue that responds to the specific line the player just read — not just the hang-out metadata. Requires knowing which line index maps to which text entry in the script pool.

### ⏳ Phase 5 — Player custom dialogue
DirectX overlay for text input. Let the player type their own dialogue choice and have the NPC respond to it. This is the full social simulation vision.

---

## Technical details

### Inference stack

```
Llama-3.1-8B-Instruct (Meta)
  → GGUF Q4_K_M quantization  (~4.7 GB, fits in 8 GB VRAM)
  → llama-cpp-python backend   (CUDA offload, no Python overhead on hot path)
  → FastAPI /generate endpoint  (Pydantic validation, async queue)
  → C# HTTP client             (30s timeout, 3× retry on 503 cold-start)
```

### Prompt structure

Every request gets a three-layer prompt:

1. **System** — character identity, arcana, personality blurb, rank-tier emotional guidance (4 tiers: stranger / acquaintance / close friend / deepest bond)
2. **Context** — scene description, prior session dialogue (rolling 8-line buffer), trimmed to 1000 chars
3. **User** — `[Scene context: ...] CharacterName:`

### Memory layout (CMM session struct)

Offsets confirmed from live hex dump during Ryuji gym hang-out (session `0x41D7156660`):

```
+0x00  int32   ConfidantId     (8 = Ryuji)
+0x04  int32   SessionPhase    (always 0 in observed sessions)
+0x08  byte    CmmIdRepeat
+0x0A  byte    EventType       (2)
+0x0B  byte    RankLevel       (rank before this session)
+0x0C  int16   SceneNumber     (51 = Ryuji gym hang-out)
+0x10  ptr     [internal CMM pointers]
+0x20  byte    TimerLo         (16-bit game clock, low byte)
+0x21  byte    TimerHi         (16-bit game clock, high byte)
+0x3F  byte    StateParity     (toggles 0x80↔0x91 ~every 2s)
```

Static pointer chain: `[p5r.exe + 0x2A63EF0] → [+0x48] → session*`

---

## Setup

> Requires a Reloaded-II installation and an NVIDIA GPU with ≥8 GB VRAM.

**1. Clone and build the mod**
```powershell
git clone https://github.com/SrijanSuresh/p5r-gen-social-links
cd p5r-gen-social-links
dotnet build mod/P5RGenSocialLinks/P5RGenSocialLinks.csproj --configuration Release
```
Copy `mod/P5RGenSocialLinks/bin/Release/net8.0/P5RGenSocialLinks.dll` to your Reloaded-II mods folder.

**2. Set up the Python server**
```powershell
cd server
python -m venv .wvenv
.\.wvenv\Scripts\activate
pip install -r requirements.txt   # llama-cpp-python[cuda], fastapi, uvicorn
```
Download a Llama-3.1-8B-Instruct GGUF (Q4_K_M) and place it at `server/models/llama-3.1-8b-instruct-q4_k_m.gguf`.

**3. Start server, then game**
```powershell
.\start.bat          # real inference
# or
.\start-mock.bat     # no GPU needed, canned responses for testing
```

**Run tests**
```powershell
cd server
python -m pytest tests/ -q
# 119 passed, 1 skipped
```

---

## Repo structure

```
p5r-gen-social-links/
├── mod/P5RGenSocialLinks/       # C# Reloaded-II mod
│   ├── Memory/                  # Pointer chain, struct reading, write-back
│   ├── Server/                  # HTTP client, health checker
│   ├── Mod.cs                   # Entry point, poll loop, hook wiring
│   ├── DialogueBridge.cs        # LLM dispatch, throttle, session history
│   └── GenDialogue.json         # Runtime config (no recompile needed)
├── server/                      # Python inference server
│   ├── inference/               # Pipeline, queue, postprocessor, config
│   ├── social_link/             # Arcana roster, prompt builder, tier mapping
│   ├── tests/                   # 119 tests (pytest)
│   └── main.py                  # FastAPI app
├── scripts/redate.sh            # Spreads commits across PR window
├── learning.md                  # 22-chapter technical journal (the real docs)
└── .github/workflows/ci.yml     # Python tests + .NET build on push
```

---

## learning.md

`learning.md` is a 22-chapter technical journal written alongside the code — covering ASLR and pointer arithmetic, Triton block-pointer math, 4-bit quantization, BF script format, CI design, and everything discovered via Cheat Engine and Ghidra. It exists because this project was built as a learning exercise as much as a mod.

---

*Built on Windows 11 · RTX 4060 8GB · P5R Steam (patch 1.0.3)*
