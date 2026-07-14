# P5R Generative Social Links — Learning Journal

---

## Chapter 1: System Architecture

### What are we building?

We are building a mod that makes Persona 5 Royal's Social Link characters **respond dynamically using a local Large Language Model (LLM)**. When Joker (the protagonist) converses with a confidant (Social Link), instead of replaying fixed script lines, our system generates new dialogue in the character's voice using AI inference running on *your own GPU*.

The system has two halves that communicate over localhost HTTP:

```
┌───────────────────────────────────────────────────────────┐
│                     P5R GAME PROCESS                      │
│                                                           │
│   Game Engine ──► Social Link Trigger ──► Dialogue Box   │
│                           │                    ▲          │
│                    (memory hook)         (memory patch)   │
│                           │                    │          │
│              ┌────────────┴────────────────────┤          │
│              │   C# Reloaded-II Mod (our mod)  │          │
│              │   Reads game state from memory  │          │
│              │   Writes LLM text back to memory│          │
│              └────────────┬────────────────────┘          │
└───────────────────────────┼───────────────────────────────┘
                            │ HTTP/localhost
                            ▼
┌───────────────────────────────────────────────────────────┐
│              Python Background Server                     │
│                                                           │
│   FastAPI endpoint                                        │
│       │                                                   │
│       ▼                                                   │
│   Prompt Builder (builds character-faithful prompts)      │
│       │                                                   │
│       ▼                                                   │
│   4-bit Quantized LLM (e.g. Llama 3.1 8B)                │
│       running on custom Triton GPU kernels                │
│       │                                                   │
│       ▼                                                   │
│   Generated dialogue text ──► returned to C# mod         │
└───────────────────────────────────────────────────────────┘
```

---

### Part A: Reloaded-II and Memory Hooking (C# side)

#### What is Reloaded-II?

Reloaded-II is an open-source .NET mod loader for PC games. It works by:
1. **Injecting a .NET runtime** into the target game process.
2. **Loading your mod DLL** (a C# class library) inside that process.
3. Giving you access to the **full process memory space** — meaning you can read and write any address the game owns.

Your mod is a C# project that implements the `IExports` and `IMod` interfaces from the Reloaded-II SDK. At startup, Reloaded-II calls your `Mod.Start()` method, and from that point you are executing *inside* the P5R process.

#### Why does memory hooking matter here?

P5R stores all its runtime state in memory — character IDs, conversation flags, dialogue string pointers, Social Link ranks, etc. Since we're inside the process, we can:
- **Read** what Social Link is active and at what story point.
- **Hook** (intercept) the function that loads dialogue strings, so we can redirect it to our generated text instead.

#### Pointer Arithmetic and Memory Layout

Games written in C++ lay out their objects in memory predictably. A "Social Link conversation object" might look like:

```
Offset 0x00  → ConfidantID  (int32, 4 bytes)
Offset 0x04  → RankLevel    (int32, 4 bytes)
Offset 0x08  → DialogueIndex (int32, 4 bytes)
Offset 0x0C  → *pDialogueStr (int64 pointer, 8 bytes) — points to a UTF-16 string
```

To read `ConfidantID`, if the base address of this object is `0x7FF812345000`, you do:
```
address = baseAddress + 0x00
value   = *(int*)address      // C# unsafe dereference
```

We will use **Reloaded.Memory** and **Reloaded.Hooks** libraries for safe abstractions over this, but the math is always: `field_address = object_base + field_offset`.

#### What is a "hook" (function hook)?

A function hook means: when the game calls function `X`, our code runs *first*, then optionally the original function runs. We implement this using **Detour patching** — we overwrite the first ~12 bytes of the target function with a `JMP` to our code (a trampoline). Reloaded.Hooks handles the trampoline mechanics for us.

We will hook the dialogue-fetch function so that instead of returning the scripted line, it returns our LLM text.

---

### Part B: 4-Bit Quantized Inference (Python side)

#### Why do we quantize?

A Llama 3.1 8B model in full float32 needs ~32 GB of VRAM. In bfloat16 it needs ~16 GB. Most gaming PCs have 8–12 GB VRAM (RTX 3060/4070 class). **4-bit quantization** reduces each weight from 32 bits → 4 bits, shrinking the model to ~4.5 GB VRAM. The quality loss is surprisingly small (< 1% perplexity increase on most benchmarks).

The dominant 4-bit format is **GPTQ** (Generalized Post-Training Quantization):
- After training, run calibration data through the model.
- For each weight matrix W, find a quantized version Q (4-bit integers) and a scale factor S such that `W ≈ Q * S` minimizes output error.
- At inference time, you dequantize on the fly: `float_weight = Q * S`.

#### Why custom Triton kernels?

Standard PyTorch matrix multiply (`torch.mm`) doesn't know about our packed 4-bit integers. We need a custom **GPU kernel** that:
1. Loads packed int4 weights from VRAM (two 4-bit values packed per byte).
2. Dequantizes them in registers (multiply by scale, subtract zero-point).
3. Performs the matrix multiply against float16 activations.

**Triton** is NVIDIA's Python-based GPU kernel authoring language. It compiles down to PTX (NVIDIA's GPU assembly) but lets you write GPU code that looks almost like Python. This is dramatically simpler than writing raw CUDA C++.

Key Triton concepts:
- **Programs**: Each Triton kernel launch creates thousands of "programs" that run in parallel on GPU SMs (Streaming Multiprocessors).
- **Block pointers**: Instead of per-element indexing, Triton works with *blocks* of contiguous memory — e.g., a 128×32 tile of the weight matrix.
- **tl.load with masks**: Loading a block near the edge of a matrix might go out of bounds, so we pass a boolean mask to guard against invalid reads.

```python
# Conceptual Triton kernel structure for dequant matmul:
@triton.jit
def dequant_matmul_kernel(
    A_ptr, W_packed_ptr, scales_ptr, Out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Each program handles one (BLOCK_M, BLOCK_N) tile of output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Build pointer offsets for this tile
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulate dot product across K dimension in BLOCK_K chunks
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        # Load activation tile (with boundary mask)
        a = tl.load(A_ptr + row_offsets[:, None] * K + (k + tl.arange(0, BLOCK_K))[None, :],
                    mask=(row_offsets[:, None] < M) & ((k + tl.arange(0, BLOCK_K))[None, :] < K))
        # Load packed int4 weights and dequantize
        w_packed = tl.load(W_packed_ptr + ...)  # int8 holding two int4
        w_float  = dequantize(w_packed, scales)  # expand to float16
        acc += tl.dot(a, w_float)
    tl.store(Out_ptr + ..., acc)
```

---

### Part C: The Communication Bridge

The C# mod calls `POST http://localhost:8765/generate` with a JSON body:
```json
{
  "confidant_id": 3,
  "rank": 5,
  "context": "Player chose response option 2 about Ann's modeling career",
  "character_name": "Ann Takamaki"
}
```

The Python server builds a character-faithful prompt, runs inference, and returns:
```json
{
  "text": "You think so? I've been really worried about whether I'm doing it for the right reasons..."
}
```

The C# mod patches the game's dialogue buffer with this string and lets the engine render it normally.

---

### Project Micro-Steps Roadmap

| # | Step | What we build |
|---|------|--------------|
| 0 | Scaffolding | Project structure, solution file, Python requirements |
| 1 | Memory Scan | Find P5R's Social Link data structures in memory |
| 2 | Mod Skeleton | Reloaded-II mod that loads/unloads cleanly |
| 3 | State Reader | Read active confidant ID and rank from memory |
| 4 | Dialogue Hook | Hook dialogue fetch function to intercept calls |
| 5 | Python Server | FastAPI server skeleton with health check |
| 6 | 4-bit Model | Load a GPTQ-quantized model with HuggingFace |
| 7 | Triton Kernel | Custom int4 dequant matmul kernel |
| 8 | Prompt Builder | Per-character system prompts (Ann, Ryuji, Makoto…) |
| 9 | Bridge | C# calls Python server; inject returned text |
| 10| Polish | Error handling, fallback to scripted lines |

---

## Chapter 2: ASLR, Module Base Addresses, and Pointer Chains

### Why Static Addresses Are Useless

When Windows loads `p5r.exe`, the OS kernel picks a **random base address** for the executable in virtual memory — this is ASLR (Address Space Layout Randomization). It's a security feature: malware can't rely on knowing where functions live. The consequence for us: every single game launch, P5R loads at a different virtual address.

```
Session 1: p5r.exe loads at 0x7FF8_1200_0000
Session 2: p5r.exe loads at 0x7FF9_AC30_0000
Session 3: p5r.exe loads at 0x7FFA_0010_0000
```

**What stays constant**: the *distance* between any two things inside the exe. The compiler laid out the binary — the offset from the start of the module to any global variable or function is baked into the `.exe` file and never changes (for a given game version).

So instead of hardcoding an absolute address, we always work with:
```
absolute_address = module_base + constant_offset
```

Reloaded-II gives us `module_base` via `IModLoader.GetModAssembly()` → `Process.GetCurrentProcess().MainModule.BaseAddress`.

---

### The Pointer Chain (Multi-Level Indirection)

P5R is a C++ game. C++ objects live on the **heap** — dynamically allocated memory. A global variable doesn't *contain* the Social Link session struct; it contains a *pointer to* it. Sometimes the global holds a pointer to a struct that holds another pointer to another struct, and so on. This is called a **pointer chain**.

Visualized:
```
MODULE BASE
    │
    + 0x01E8_0000 ──► [Global Ptr]  (holds address X)
                              │
                              X + 0x18 ──► [Mid-level object]  (holds address Y)
                                                    │
                                                    Y + 0x08 ──► SocialLinkSession* ← this is what we want
```

Each `──►` is a **dereference**: read 8 bytes at that address to get the next address. In C#:
```csharp
unsafe IntPtr ResolveChain(IntPtr moduleBase)
{
    IntPtr p = moduleBase + 0x01E8_0000;  // step 1: static offset
    p = *(IntPtr*)p;                       // step 2: dereference → follow pointer
    p = *(IntPtr*)(p + 0x18);             // step 3: offset + dereference
    p = *(IntPtr*)(p + 0x08);             // step 4: final dereference → SocialLinkSession*
    return p;
}
```

This is exactly what Cheat Engine calls a "pointer scan." When you right-click a value in Cheat Engine and click "Find what accesses this address," it can trace back through the chain to find the static root.

---

### How We Find the Real Offsets: Cheat Engine Workflow

1. **Attach Cheat Engine to p5r.exe** while a Social Link conversation is active.
2. **Search for the Confidant ID value** (e.g., scan for `int` value `1` when talking to Ryuji).
3. **Narrow it down**: exit the conversation, scan for "changed value." Re-enter, scan for original. Repeat until 1–3 addresses remain.
4. **Right-click → "Find what writes to this address"** — this shows which game function modifies `ConfidantId`, giving you the struct context.
5. **Pointer scan** the final address → Cheat Engine traces back to a module-relative root pointer chain.
6. **Verify in Ghidra**: import the binary, look up the address, confirm the struct layout.

> Note: The offsets in `GameMemory.cs` are currently placeholders. They MUST be verified live against `p5r.exe` using this workflow before any real memory reads will work.

---

### Why Reloaded.Memory Over Raw Pointers

Raw `*(IntPtr*)p` will hard-crash the process (Access Violation / null deref) if any step in the chain returns a null pointer. `Reloaded.Memory` provides:
- **`Memory.Read<T>(nuint address, out T value)`** — returns `bool` success instead of crashing.
- **`MemoryBufferHelper`** — allocates executable memory regions for our trampoline hooks.
- **`SigScanner`** — finds functions by their **byte signature** rather than raw offset, which survives minor game patches (the function body changes less often than its absolute position).
