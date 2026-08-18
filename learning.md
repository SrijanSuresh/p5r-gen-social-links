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

---

## Chapter 3: The Reloaded-II Mod Lifecycle & Polling

### The IMod Interface

Reloaded-II defines a contract every mod must implement:

```
IMod
 ├── Start(IModLoaderV1 loader)   ← called once when mod is injected
 ├── Suspend()                    ← called when user disables mod at runtime
 ├── Resume()                     ← called when user re-enables mod
 ├── Unload()                     ← called when mod is fully removed
 ├── CanSuspend() → bool          ← tell Reloaded if you support suspend/resume
 └── CanUnload()  → bool          ← tell Reloaded if you support hot-unload
```

`Start()` is your constructor. Everything — hook setup, reader initialization, background threads — launches from here. The `IModLoaderV1` parameter is your gateway to:
- The game's `Process` object (for module base address)
- The `IReloadedHooks` service (for function detours)
- The logger (writes to Reloaded-II's console)

### Getting the Module Base Address

```csharp
void Start(IModLoaderV1 loader)
{
    // Reloaded gives us the current process; MainModule is the .exe itself
    nuint moduleBase = (nuint)Process.GetCurrentProcess()
                                     .MainModule!
                                     .BaseAddress;
    // Now moduleBase + any constant offset = a stable address regardless of ASLR
}
```

`MainModule.BaseAddress` is where Windows loaded `p5r.exe` this session. This value changes every launch (ASLR) but is always correct at runtime.

### Polling vs. Hooking: Two Ways to Watch Game State

**Polling** (what we implement now): a background thread wakes up every N milliseconds and calls `TryReadSnapshot()`. Simple, but wastes CPU when nothing is happening.

**Hooking** (Micro-step 4): detour the exact function that *sets* `ConfidantId` when a conversation begins. Zero wasted cycles — we're called only when the game itself triggers the event.

We start with polling because it lets us verify our memory reads without needing to identify the exact game function yet. Once reads are confirmed correct, we swap to a hook.

### System.Threading.PeriodicTimer (C# 6+)

Instead of `Thread.Sleep()` in a loop (wastes a thread), we use `PeriodicTimer`:

```csharp
// Fires every 250ms on the thread pool; no dedicated thread held between ticks.
_timer = new PeriodicTimer(TimeSpan.FromMilliseconds(250));
_pollTask = Task.Run(async () =>
{
    while (await _timer.WaitForNextTickAsync(_cts.Token))
    {
        var snap = _reader.TryReadSnapshot();
        if (snap is not null)
            OnConversationActive(snap);
    }
});
```

`CancellationTokenSource` (`_cts`) lets us cleanly stop the loop in `Unload()` without leaving orphan threads in the game process.

### Why Clean Unload Matters

Our mod runs *inside* P5R's process. If we leave a thread running after `Unload()`, it keeps reading memory addresses that may get freed as the game progresses. That's a use-after-free — eventual crash. Always cancel and await your background tasks in `Unload()`.

---

## Chapter 4: Function Hooking with Reloaded.Hooks

### The Problem with Polling

Our 250ms poll loop works but wastes CPU: it calls `TryReadSnapshot()` 4× per second even when the player is in a dungeon, menu, or cutscene with no Social Link active. Over a 40-hour playthrough, that's ~576,000 pointless reads. The solution is to **eliminate polling entirely** and instead run our code *only* when P5R itself decides a conversation starts.

### What is a Function Hook (Detour)?

A **detour** (also called a hook) is a technique where you overwrite the first ~14 bytes of a target function with a `JMP` instruction pointing to your code:

```
Before hook:
p5r.exe + 0xABC123:  push rbp          ; original function start
                      mov rbp, rsp
                      ...

After hook:
p5r.exe + 0xABC123:  jmp 0x7FF9_0001_0000  ; → our trampoline stub
                      nop nop nop           ; (original bytes saved by Reloaded.Hooks)
```

Reloaded.Hooks:
1. **Saves** the overwritten bytes into a "trampoline" — a small stub that executes the original bytes then jumps back into the original function *after* the patch.
2. **Writes** the `JMP` to our code at the function's entry point.
3. **Gives us a delegate** (`IHook<TDelegate>`) that we can call to invoke the original function, optionally skipping or modifying its arguments/return value.

### Hook Types: Pre-hook vs. Mid-function

- **Pre-hook** (what we use): runs before the original function. We read or modify state, then optionally call the original.
- **Post-hook**: some frameworks let you also intercept the return. With Reloaded.Hooks we achieve this by calling the original ourselves and inspecting its return value.

### Defining the Hook Signature

The hook delegate must exactly match the C++ calling convention and signature of the target function. P5R is an x64 Windows binary, which uses the **Microsoft x64 calling convention**:
- First 4 integer args → `rcx`, `rdx`, `r8`, `r9`
- Remaining args → stack
- Return value → `rax`

For a hypothetical conversation-init function:
```csharp
// Matches void __fastcall SocialLink_BeginConversation(SocialLinkSession* session)
[Function(CallingConventions.Microsoft)]
delegate void BeginConversationDelegate(nuint sessionPtr);
```

### Finding the Target Function

To hook the right function, we need its address. Two methods:

**Method 1 — Static offset** (fragile across patches):
```csharp
nuint funcAddr = moduleBase + 0x00_ABC1_23;
```

**Method 2 — Byte signature scan** (robust):
```csharp
// Signature: first N bytes of the function that are unlikely to change
// ?? = wildcard byte
string sig = "48 89 5C 24 ?? 48 89 6C 24 ?? 48 89 74 24 ?? 57 41 56 41 57";
var scanner = new Scanner((byte*)moduleBase, moduleSize);
var match = scanner.CompiledFindPattern(sig);
nuint funcAddr = moduleBase + (nuint)match.Offset;
```

A byte signature targets the *body* of the function (specific instruction sequences) which changes far less often than a function's absolute position in memory when the binary is rebuilt.

### How Reloaded.Hooks Wires It Up

```csharp
// 1. Get the hooks service from the mod loader
var hooks = _modLoader.GetController<IReloadedHooks>()!.GetWrapper();

// 2. Create the hook (Reloaded.Hooks writes the JMP, saves the trampoline)
_conversationHook = hooks.CreateHook<BeginConversationDelegate>(
    OnBeginConversation, (long)funcAddr).Activate();

// 3. Our handler — called instead of the original function
private void OnBeginConversation(nuint sessionPtr)
{
    // Read state from sessionPtr directly (it's the session being initialized)
    // ... fire LLM request ...

    // Always call the original so P5R continues normally
    _conversationHook.OriginalFunction(sessionPtr);
}
```

The original function still runs — we just get to execute code *before* (or after) it every time the game triggers a conversation.


---

## Chapter 5: 4-Bit Quantization — The Mathematics

### Why Not Just Use float16?

A float16 weight occupies 16 bits. With 7 billion parameters (Llama-7B):
  7,000,000,000 × 2 bytes = **14 GB** — too large for an RTX 3060 (12 GB) or 4070 (12 GB).

With 4-bit quantization:
  7,000,000,000 × 0.5 bytes = **3.5 GB** — fits with room for activations and KV cache.

### GPTQ: Post-Training Quantization

GPTQ (Frantar et al., 2022) quantizes a trained model without retraining. For each weight matrix W:

1. Run calibration data (a few hundred text samples) through the model.
2. For each row of W, solve the optimization:
   ```
   minimize ‖W·X - Q·S·X‖²   subject to Q ∈ {0..15}ⁿ
   ```
   where X is the layer''s input activations on the calibration data.
3. Q is a matrix of 4-bit unsigned integers (values 0–15).
4. S is a float16 scale factor (one per group of K consecutive weights).
5. Z is a zero-point (shifts Q so the quantized range is symmetric).

The dequantization formula at inference time:
```
W_float ≈ (Q - Z) × S
```

### Groups: Why Not One Scale Per Layer?

A single scale per layer means every weight must fit in the range
`[scale × 0, scale × 15]`. Low-magnitude weights get crushed to 0;
high-magnitude weights clip. **Group quantization** assigns one scale
per G consecutive weights (G = 32, 64, or 128). Smaller G = more
scales stored = better quality, but more memory for scales.

### What Changes at Inference Time

At runtime, for each matmul:
1. Load packed int4 weights (2 per byte) from VRAM → GPU registers.
2. Unpack to int8 via bit masking.
3. Subtract zero-point and multiply by scale → float16.
4. Perform standard float16 multiply-accumulate with the activation.

Steps 1–3 are what our Triton kernel implements.
---

## Chapter 6: Triton Block Pointers and the dequant_matmul Kernel

### What is a Triton "Program"?

When you launch `dequant_matmul_kernel[grid](...)`, Triton spawns
`grid[0] × grid[1]` parallel programs on the GPU''s Streaming Multiprocessors.
Each program handles one tile of the output matrix, completely independently.
This is called **data parallelism at the tile level**.

`tl.program_id(0)` returns which row-tile this program owns.
`tl.program_id(1)` returns which col-tile this program owns.

### Block Pointer Arithmetic (the hardest part)

For a 2D matrix A with shape [M, K] stored row-major:
```
element A[row, col] lives at address:  A_ptr + row * K + col
```

For a **tile** of shape [BLOCK_M, BLOCK_K] starting at (pid_m * BLOCK_M, k_start):
```
row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # shape [BLOCK_M]
k_offs   = k_start          + tl.arange(0, BLOCK_K)   # shape [BLOCK_K]

# Broadcasting creates a [BLOCK_M, BLOCK_K] address matrix:
ptrs = A_ptr + row_offs[:, None] * K + k_offs[None, :]
```

`[:, None]` promotes the row vector to shape [BLOCK_M, 1].
`[None, :]` promotes the col vector to shape [1, BLOCK_K].
NumPy-style broadcasting produces a [BLOCK_M, BLOCK_K] matrix of addresses.

### The Boundary Mask

The matrix edge tiles (last row-tile or col-tile) often go out of bounds.
Loading from an invalid GPU address causes undefined behavior (silent wrong data
or a kernel crash). We guard every load:

```python
mask = (row_offs[:, None] < M) & (k_offs[None, :] < K)
tile  = tl.load(ptrs, mask=mask, other=0.0)
```

`other=0.0` fills out-of-bounds lanes with zero, which contributes nothing to
the dot product — mathematically correct, not just safe.

### tl.dot vs. element-wise multiply

`tl.dot(A, B)` is Triton''s tensor-core-accelerated matmul. It emits an mma
(matrix multiply-accumulate) instruction that runs on the GPU''s Tensor Cores
(WMMA / MMA units), achieving 8–16× throughput vs. a naive loop. This is the
core reason writing a custom kernel beats a for-loop dequant on the CPU.

### BLOCK_M / BLOCK_N / BLOCK_K Tuning

These three constants control how much work each program does:
- Too small → too many programs → overhead from launch/sync dominates.
- Too large → registers spill to local memory (slow) or the tile doesn''t fit.
- `triton.autotune` tries all combinations and picks the fastest for your GPU.
  We use fixed values (16, 64, 32) as a starting point; autotune will be
  wired in Micro-step 7.
---

## Chapter 6: Triton Block Pointers and the dequant_matmul Kernel

### What is a Triton "Program"?

When you launch `dequant_matmul_kernel[grid](...)`, Triton spawns
`grid[0] × grid[1]` parallel programs on the GPU's Streaming Multiprocessors.
Each program handles one tile of the output matrix, completely independently.

`tl.program_id(0)` returns which row-tile this program owns.
`tl.program_id(1)` returns which col-tile this program owns.

### Block Pointer Arithmetic

For a 2D matrix A with shape [M, K] stored row-major:
```
A[row, col] lives at:  A_ptr + row * K + col
```

For a tile [BLOCK_M, BLOCK_K] starting at (pid_m * BLOCK_M, k_start):
```python
row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
k_offs   = k_start          + tl.arange(0, BLOCK_K)   # [BLOCK_K]
# Broadcasting → [BLOCK_M, BLOCK_K] address matrix
ptrs = A_ptr + row_offs[:, None] * K + k_offs[None, :]
```

`[:, None]` promotes to shape [BLOCK_M, 1]; `[None, :]` to [1, BLOCK_K].
Broadcasting produces the full [BLOCK_M, BLOCK_K] address block.

### The Boundary Mask

The matrix edge tiles go out of bounds. Loading from an invalid GPU address
causes silent wrong data or a kernel crash. We guard every load:

```python
mask = (row_offs[:, None] < M) & (k_offs[None, :] < K)
tile  = tl.load(ptrs, mask=mask, other=0.0)
```

`other=0.0` fills out-of-bounds lanes with zero — mathematically correct
(contributes nothing to the dot product), not just safe.

### tl.dot vs. element-wise multiply

`tl.dot(A, B)` emits a Tensor Core mma (matrix-multiply-accumulate) instruction,
achieving 8–16× throughput over a naive multiply-accumulate loop. This is the
core reason the custom kernel outperforms a CPU dequant loop.

### BLOCK_M / BLOCK_N / BLOCK_K Tuning

- Too small → too many programs → launch overhead dominates.
- Too large → register spill to local memory (slow).
- `triton.autotune` searches all combinations; we use (16, 64, 32) as a
  starting point and wire autotune in Micro-step 7.

---

## Chapter 8: ContextBuilder — Reading Live Dialogue from Game Memory

### Why We Need Context At All

The LLM prompt currently receives `"Dialogue line 3"` as context. That's nearly
useless — the model doesn't know who's speaking, what the scene is, or what the
character was about to say. The fix is to read the ACTUAL scripted dialogue line
from the game's own memory buffer before we overwrite it.

### Timing: The Read-Before-Write Window

Our hook calls `OriginalFunction(sessionPtr)` FIRST. That call runs the real P5R
conversation-init code, which among other things copies the scripted NPC line into
the dialogue buffer at `sessionPtr + 0x10`. By the time `OriginalFunction` returns,
the buffer already contains the game's intended dialogue.

Timeline (all on the game thread):
```
OnConversationInit called
  → OriginalFunction(sessionPtr)   ← game writes scripted line to buffer[0x10]
  ← OriginalFunction returns
  → ContextBuilder.Build(snap)     ← WE READ the scripted line (synchronous)
  → _bridge.DispatchAsync(...)     ← async task fires; game thread returns immediately
     (later, on thread pool)
     → HTTP → LLM → write new text → OVERWRITE buffer[0x10]
```

The read is synchronous and cheap (a handful of cache hits). The overwrite happens
later on the thread pool. There is no race on the read because the game thread won't
modify the buffer again until the player advances the dialogue.

### UTF-16LE In-Memory Strings

Windows and all games using Win32 text APIs store strings as UTF-16 Little Endian.
Each character is 2 bytes. In C# `char` is also 2 bytes (UTF-16), so `char*` maps
directly to the game's wide-character pointer with zero conversion.

Memory layout of the string "Hi!" in UTF-16LE:
```
Address:  [+0x00] [+0x01] [+0x02] [+0x03] [+0x04] [+0x05] [+0x06] [+0x07]
Bytes:      48 00   69 00   21 00   00 00
Chars:        H       i       !      \0
```

Reading it in C#:
```csharp
unsafe string ReadWideString(nuint addr, int maxChars = 512)
{
    char* ptr = (char*)addr;

    // Walk until null terminator or max — guards against unterminated buffers
    int len = 0;
    while (len < maxChars && ptr[len] != '\0')
        len++;

    // new string(char*, start, length) copies exactly `len` chars from unmanaged memory
    return len == 0 ? string.Empty : new string(ptr, 0, len);
}
```

`new string(char*, 0, len)` is an unsafe constructor that copies from unmanaged
memory into a managed C# string. The GC then owns the copy — the original buffer
can be overwritten without affecting the string we just made.

### Separation: Pure Logic vs. Unsafe Read

We split ContextBuilder into two layers:
1. `ReadCurrentDialogue(snap)` — unsafe, reads from game memory, returns raw string
2. `Build(dialogueIndex, rawDialogue)` — pure C#, formats the context string

The pure `Build` method can be called from a unit test without a live game. The
unsafe read is inherently unverifiable in tests — we accept it and rely on the
"null guard + max length" to make it fail-safe (returns empty string if the buffer
looks wrong).

---

## Chapter 7: Wiring the LLM Bridge + GPU Autotune + Finding Real Memory Addresses

### Part A: Wiring DialogueBridge into the Hook

The `DialogueBridge` class already exists with a complete `DispatchAsync` method.
The missing piece was connecting it to `OnConversationInit` in `Mod.cs`.

Two sub-problems:

**1. The ILogger adapter pattern**

`DialogueBridge` defines its own `internal interface ILogger { void WriteLine(string); }`.
`Reloaded.Mod.Interfaces.Internal.ILoggerV2` also has `void WriteLine(string)`.

The signatures are identical, but C# interfaces are *nominal* — you cannot pass an
`ILoggerV2` where a `DialogueBridge.ILogger` is expected even if both have the same
methods. This is by design: a class opts in to an interface explicitly with `: IFoo`.

Solution: a tiny private adapter class inside `Mod.cs`:
```csharp
private sealed class LoggerAdapter : DialogueBridge.ILogger
{
    private readonly ILoggerV2 _inner;
    internal LoggerAdapter(ILoggerV2 inner) => _inner = inner;
    public void WriteLine(string msg) => _inner.WriteLine(msg);
}
```
One liner per member, zero overhead. This pattern appears constantly in systems code
whenever two interfaces define the same contract independently.

**2. The context string**

`DispatchAsync(snap, contextText)` — what do we pass for `contextText`?

We don't yet parse the full NPC dialogue text from VRAM (that's a future step once
we have the real offsets from Cheat Engine). For now we send the dialogue line index:
`$"Dialogue line {snap.DialogueIndex}"`. The LLM server's `build_prompt` will embed
this into the user turn so Ryuji / Makoto know roughly where in the conversation they
are. Good enough for smoke testing.

### Part B: triton.autotune

`@triton.autotune` is a decorator that benchmarks a list of `triton.Config` objects
the first time the kernel is called with a given `key` combination, then caches the
winner.

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N":  64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N":  64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N":  64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],       # different shapes → different best config
)
@triton.jit
def dequant_matmul_kernel(...):
    ...
```

**num_warps**: a warp is 32 GPU threads that execute in lockstep. More warps = more
occupancy (GPU can hide memory latency by swapping warps), but also more register
pressure per SM. Typical values: 4 (conservative) to 8 (aggressive for large tiles).

**num_stages**: how many "software pipeline" stages Triton uses to overlap memory
loads with compute. Higher = more registers used but latency better hidden. 2–4 is
the useful range.

**Lambda grid**: because BLOCK_M and BLOCK_N are now chosen at runtime by autotune
(not hardcoded in the wrapper), the grid calculation must be a lambda that reads from
the `meta` dict that autotune injects:
```python
grid = lambda meta: (
    triton.cdiv(M, meta["BLOCK_M"]),
    triton.cdiv(N, meta["BLOCK_N"]),
)
```
This is the single biggest API difference from a non-autotuned kernel.

**Cache behaviour**: autotune benchmarks once per unique `(M, N, K)` tuple, writes
the result to a `.triton_cache` directory, and reuses it on subsequent process starts.
First inference call is slow (~seconds); all following calls use the cached winner.

### Part C: Finding Real P5R Memory Addresses with Cheat Engine

THIS IS YOUR JOB — the game must be running. Here is the exact step-by-step.

**Goal**: find the real byte address of the function that initializes a Social Link
conversation, so we can replace the placeholder in `Signatures.cs`.

**Tools needed**:
- Cheat Engine 7.5+ (free, cheatengine.org)
- P5R running via Steam (windowed mode is easier)
- Optionally: x64dbg or Ghidra for the byte extraction step

---

**Step 1 — Attach Cheat Engine to P5R**

1. Launch P5R. Get to a point where you can START but haven't started a Social Link
   conversation (e.g., talk to Ryuji in Shujin after school).
2. Open Cheat Engine → click the glowing PC icon (top-left) → select `p5r.exe`.
3. Change value type to `4 Bytes`, scan type to `Exact Value`.

**Step 2 — Find the ConfidantId address**

1. In-game: begin a Social Link conversation with Ryuji (Chariot = ID 1 in our table).
2. In Cheat Engine: scan for `1`. You'll get thousands of results — that's fine.
3. End the conversation. Scan again for `0` (or start a different confidant, scan for
   their ID). The address that changed is the live ConfidantId field.
4. Repeat 1–2 more times; you should narrow to 1–5 addresses. Add the survivor to
   the address list (double-click it).

**Step 3 — Find what WRITES to that address**

1. Right-click the ConfidantId address in the list → "Find out what writes to this
   address" → "Find out what writes to this address".
2. Cheat Engine installs a hardware watchpoint. Click "Yes" to any warning.
3. In-game: trigger a NEW Social Link conversation. Cheat Engine's instruction list
   will populate with the assembly instruction that just wrote to that address.
4. The top entry is the instruction inside `BeginConversation`. Note the **instruction
   address** (left column) and the **instruction bytes** (right column).

**Step 4 — Get the function start and its bytes**

The instruction Cheat Engine shows is inside the function, not at its prologue.
We need the function's STARTING bytes for our signature.

1. In Cheat Engine: click the instruction → "Show disassembler" (or press Ctrl+D).
   This opens the memory view at that instruction.
2. Scroll UP until you see the function prologue — usually starts with:
   ```
   40 55           PUSH RBP
   48 89 5C 24 ??  MOV [RSP+??], RBX
   ```
   or similar register-save sequence. Function starts = the first instruction after
   the `CALL` target (or after `INT3` padding bytes `CC CC CC`).
3. Note down ~20 bytes from the start. Replace stack-relative offsets (`??`) with
   wildcards in `Signatures.cs`.

**Step 5 — Update Signatures.cs**

Open `mod/P5RGenSocialLinks/Memory/Signatures.cs` and replace:
```csharp
internal const string BeginConversation =
    "48 89 5C 24 ?? 48 89 6C 24 ?? 48 89 74 24 ?? 57 41 56 41 57";
```
with the bytes you found. Run `dotnet build` and then `TryActivateHook()` will find
the real function on the next P5R launch.

---

## Chapter 9 — Pointer Indirection: Chasing the Dialogue Buffer

### The Problem: Inline String vs. Pointer-to-String

A C struct field `char* dialogueBuffer` and a C struct field `char dialogueText[256]`
look identical when you only know the field's **offset** — both appear at `+0x10` in
the struct. But they are fundamentally different in memory:

```
Inline (char[256]):
  sessionBase + 0x10 → A B C D E F ...   ← the text itself starts here

Pointer-to-string (char*):
  sessionBase + 0x10 → 60 66 F2 41 00 00 00 00   ← an 8-byte pointer value
                             ↓
             0x41F2E46660 → 43 00 79 00 6F 00 ...  ← "C\0y\0o\0..." (UTF-16LE)
```

The P5R session struct stores a **pointer** at `+0x10`, not inline text. Reading bytes
directly from `sessionBase + 0x10` gives you the raw address bits of the dialogue
string, which looks like garbage when decoded as characters.

### Why P5R Uses Pointer Indirection

P5R's dialogue strings live in a separate heap allocation managed by the CMM/flowscript
system. The session struct only holds a reference (pointer) to that allocation because:

1. **The text varies in length** — a fixed inline buffer wastes space for short lines
   and can't hold long ones.
2. **Strings are shared** — the same scripted line text may be referenced from multiple
   events. A pointer lets both events point to the same allocation.
3. **Hot-swap without struct resize** — P5R's streaming system can replace a string
   in-place by updating the pointer, keeping the session struct's size constant.

### The Double-Dereference Pattern

Reading a `char*` field from a game struct always requires two unsafe reads:

```csharp
// Step 1: read the pointer stored at struct+offset → nuint
nuint ptrFieldAddr = sessionBase + P5ROffsets.DIALOGUE_BUFFER;  // where the ptr lives
if (!MemoryGuard.IsReadable(ptrFieldAddr, sizeof(nuint))) return null;
nuint strAddr = *(nuint*)ptrFieldAddr;                           // the pointer value

// Step 2: validate the pointed-to address, then read the string
if (!MemoryGuard.IsReadable(strAddr, 2)) return null;
char* chars = (char*)strAddr;
// walk null-terminated UTF-16LE ...
```

Skipping the first `IsReadable` check crashes with `AccessViolationException` if the
session struct is partially initialized. Skipping the second crashes if the string
pointer points at unmapped memory (e.g. freed heap block).

### UTF-16LE: Two Bytes Per Character

P5R stores all in-game text as **UTF-16 Little Endian**. In this encoding every
ASCII character occupies two bytes with a zero byte appended:

```
'C' = 0x43 0x00
'y' = 0x79 0x00
'o' = 0x6F 0x00
'\0' (null terminator) = 0x00 0x00
```

So `ptr[len] != '\0'` (where `char* ptr`) checks the first of the two null bytes.
C# `char` is already 2 bytes (UTF-16), so iterating `char*` naturally steps by 2
bytes and aligns with the game's encoding.

### Discovery Loop: Chasing the Pointer Live

Until we confirm `+0x10` is the right field, the poll loop logs the raw 8 bytes at
`+0x10` as a hex address, then attempts to follow it. In the Reloaded console you
should see:

```
[P5RGenSocialLinks] Poll: session=0x41F2E4F090
[P5RGenSocialLinks] +0x10 ptr=0x41F2E46660
[P5RGenSocialLinks] +0x10 content (UTF-16): "Yo, what's up man..."
```

If `+0x10 ptr` is `0x0000000000000000` or unreadable, `+0x10` is not the dialogue
pointer and we need to scan further offsets with Cheat Engine or Ghidra.

---

## Chapter 10 — Intra-Session Change Detection and Dialogue Pointer Scanning

### Why Session-Change Gating Is Wrong Here

The original poll loop logged only when `session != lastSession` — i.e., when a
completely new conversation started. That fires once (at session creation) and then
goes silent for the entire conversation.

The problem: the dialogue buffer pointer and the dialogue-index field both CHANGE
**within** the same session, every time the player advances to a new line. If we only
log on session change, we will always see the cold, partially-initialized struct and
miss every mid-conversation state.

**Fix**: track three separate "last seen" values:

```
lastSession      → triggers full struct hex-dump (expensive, only on new conversation)
lastDialoguePtr  → triggers follow + decode (fires per-line during conversation)
lastDialogueIdx  → fires on every line advance even if ptr hasn't changed
```

### Scanning Multiple Ptr Candidates

When `+0x10` is null at session init, the dialogue ptr might live at a later offset.
The hex dump revealed `+0x18 = 0x70BAA0B8` (non-null) and everything else null/data.
A systematic scan deferences every 8-byte-aligned word in the first 0x80 bytes of the
struct and asks: "is this address readable and does it contain any non-zero chars?"

```csharp
for (nuint off = 0x10; off <= 0x70; off += 8)
{
    nuint candidate = *(nuint*)(sessionBase + off);
    if (candidate == 0) continue;
    if (!MemoryGuard.IsReadable(candidate, 2)) continue;
    char* chars = (char*)candidate;
    if (chars[0] != '\0')
        _logger.WriteLine($"  ptr at +0x{off:X2} → 0x{candidate:X} = '{chars[0]}{chars[1]}...'");
}
```

Running this every time `dialogueIdx` ticks up shows which offset "wakes up" as
dialogue begins — that is the real dialogue buffer pointer.

### Why DIALOGUE_INDEX Is Useful Even Without the Buffer

Even if we never find the text buffer, `dialogueIndex` (int32 at `+0x04`) increments
every time the player taps the text-advance button. Combined with the known
`confidantId` and `rankLevel`, it uniquely identifies which scripted line is playing.

P5R's flowscripts (`.flow` files) contain every dialogue line indexed by scene number
and line offset. ShrineFox's Atlus Script Tools can decompile them. Once extracted,
we can build a lookup table:

```python
{ (confidant_id, rank, dialogue_index): "scripted line text" }
```

The mod sends `(confidantId, rank, dialogueIndex)` to the server. The server looks up
the scripted line to use as **context** for the LLM, then generates an alternative.
This approach completely bypasses the dialogue buffer problem — we never need to read
live game memory for the text at all.

The tradeoff: the lookup database requires a one-time offline extraction step, and it
only works for scenes that are in the flowscripts (not runtime-generated text).

---

## Chapter 11 — Pivoting from Live-Text Reading to Metadata-Driven Context

### Why Pointer Scanning Failed

The ptr scan at session init revealed no pointer to readable text:

```
+0x18 → 0x70D15418  [0008 0402]  ← binary data (backspace + device-control bytes)
+0x28 → 0x420A560000 [000B 000A]  ← LF + VT control bytes (likely a script binary header)
+0x30 → 0x420A56F7D8 [DE40 7D5E]  ← first char is a lone low-surrogate (invalid UTF-16)
+0x48 → 0x420BAC39F0 [0000 0000]  ← self-referential pointer, all zeros
```

This is expected. The CMM session struct is a **controller object**: it tracks *which*
conversation is happening, not *what text* is being displayed. The dialogue text is
managed by a completely separate subsystem — P5R's flowscript/VMD engine — with its
own allocator and display pipeline. There is no direct pointer from the session struct
to the text buffer.

Finding that text pointer would require either:
1. Hooking the text-render function (requires more Ghidra analysis)
2. Scanning process memory for the live text string during display

Both are feasible but complex. We don't need to do either.

### The Three Fields We Already Have Are Enough

```
ConfidantId   (int32 +0x00) = 8          → "Ryuji Sakamoto"
RankLevel     (byte  +0x0B) = 4          → rank 4 Social Link
SceneNumber   (int16 +0x0C) = 0x33 = 51  → specific hang-out event script
DialogueIndex (int32 +0x04) = 0, 1, 2…  → line within that scene
```

These four values **uniquely identify every scripted line** in P5R. The game's
flowscript database (`.flow` files extracted with ShrineFox's Atlus Script Tools)
is indexed by exactly this tuple. We can:

- Phase 1 (current): Send the tuple to the LLM with no text context — the LLM
  knows Ryuji's character and generates believable rank-4 dialogue from that alone.

- Phase 2 (later): Build a `(confidantId, sceneNumber, lineIndex) → text` lookup
  from extracted scripts, pass the scripted line as user-prompt context.

Phase 1 gives us a working E2E pipeline immediately. Phase 2 improves prompt quality
without changing the architecture.

### Why the LLM Can Generate Without the Scripted Text

The system prompt already injects full character context:
```
"You are Ryuji Sakamoto from Persona 5 Royal. Your arcana is Chariot.
 Character notes: Loud, loyal, hot-headed best friend; talks in street slang.
 Match the emotional tone appropriate for Social Link rank 4/10."
```

P5R Social Link conversations at rank 4 follow a known emotional arc (Ryuji is working
through his track team trauma). An LLM trained on internet text has extensive P5R
fan-fiction, wiki content, and dialogue transcripts in its training data. The rank +
character identity is sufficient to generate plausible, in-character lines.

### JSON Serialization: PascalCase vs snake_case

C# `System.Text.Json` serializes property names **as-is** (PascalCase by default):
```json
{"ConfidantId": 8, "Rank": 4, "Context": "...", "CharacterName": "Ryuji"}
```

Python Pydantic (v2) expects field names as declared — snake_case:
```python
confidant_id: int   # Pydantic sees the JSON key "ConfidantId" and finds no match
```

This causes a **422 Unprocessable Entity** silently: Pydantic rejects the request and
the mod logs "LLM error: 422". Fix on the C# side using `JsonNamingPolicy.SnakeCaseLower`
(available in .NET 8):

```csharp
private static readonly JsonSerializerOptions _jsonOpts = new()
{
    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
};
string json = JsonSerializer.Serialize(request, _jsonOpts);
```

This converts `ConfidantId → confidant_id`, `CharacterName → character_name` at the
call site, without changing the C# class definition.

### Triton on Windows

`triton` (the NVIDIA Triton GPU compiler) only supports **Linux**. On Windows it either
fails to install or crashes at import time. The auto-gptq `use_triton=True` flag
switches the matmul backend to Triton — on Windows this must be `False` so auto-gptq
falls back to its built-in CUDA or ExLlama kernels.

Our custom Triton kernel (`dequant_matmul.py`) is still valid and runs in WSL2 or a
Linux deployment. For the Windows dev loop, we use `use_triton=False`.

### Mock Mode for Dev/E2E Testing

Loading a 4-bit Llama-7B model takes ~30 seconds and requires 4 GB VRAM. During
development we want instant E2E tests. A `MOCK_LLM=1` env-var makes the server skip
model loading and return a canned Ryuji response:

```python
if os.getenv("MOCK_LLM"):
    _pipeline = _MOCK_SENTINEL   # truthy sentinel
```

```python
@app.post("/generate")
async def generate(req: GenerateRequest) -> GenerateResponse:
    if _pipeline is _MOCK_SENTINEL:
        return GenerateResponse(text=f"[MOCK] Yo, Confidant #{req.confidant_id} rank {req.rank}. Let's roll!")
```

This lets us run `MOCK_LLM=1 python main.py` and verify the full HTTP round-trip
(C# POST → Pydantic parse → JSON response → C# log) without any GPU or model download.

The `context` field sent to the server can be:
```
"[Scene 51] Social Link hang-out — Confidant #8, rank 4"
```

This tells the LLM: a scene-51 hang-out, early in Ryuji's friendship arc.
That's enough to produce good output.

---

## Chapter 12 — First End-to-End Round-Trip (Milestone)

**Date**: 2026-08-08

The full pipeline executed successfully for the first time:

```
P5R game (Ryuji gym hang-out, rank 4)
  ↓  CMM session detected by poll loop
C# mod → SocialLinkSnapshot(ConfidantId=8, RankLevel=4, SceneNumber=51)
  ↓  ContextBuilder.Build()
  ↓  "[Scene 51] Social Link hang-out — Confidant #8, rank 4"
  ↓  DialogueBridge.DispatchAsync()
  ↓  LLMClient.GenerateAsync() → POST http://localhost:8765/generate
  ↓  {"confidant_id":8,"rank":4,"context":"[Scene 51]...","character_name":"Ryuji Sakamoto"}
Python FastAPI server (MOCK_LLM=1)
  ↓  200 OK
  ↓  {"text":"[MOCK] Ryuji Sakamoto (rank 4): Yo, let's do this! ..."}
C# mod
  ↓  [P5RGenSocialLinks] LLM: "[MOCK] Ryuji Sakamoto (rank 4): Yo, let's do this!..."
```

### What Was Confirmed

- `JsonNamingPolicy.SnakeCaseLower` correctly converted `ConfidantId → confidant_id`
  so Pydantic could parse the request without a 422 error.
- WSL2 `localhost` routes to the Windows host — the C# mod on the Windows P5R process
  can reach a FastAPI server running inside WSL2 on the same port.
- The background `Task.Run` in `DialogueBridge.DispatchAsync` returns from the hook
  immediately, then completes the HTTP call without blocking the game thread.
- One LLM dispatch fires per new Social Link hang-out (session pointer change), which
  is the correct Phase 1 granularity.

### What Remains

1. **Dialogue write-back**: The generated text is logged but not injected into the game.
   The dialogue text buffer is managed by P5R's flowscript engine, not the CMM session
   struct. Finding it requires hooking the text-render function or a separate CE scan
   of live text addresses during active dialogue display.

2. **Real model**: Switched from auto-gptq to llama-cpp-python — see Chapter 13.

---

## Chapter 13 — llama-cpp-python vs auto-gptq: Backend Choice and GGUF Format

### Why We're Switching from auto-gptq to llama-cpp-python

`auto-gptq` was the original plan because it supports our custom Triton dequant kernel.
But three Windows-specific problems made it the wrong choice for dev:

| Problem | auto-gptq | llama-cpp-python |
|---|---|---|
| Triton backend | Linux-only — had to disable | Not needed — uses llama.cpp's own CUDA kernels |
| Windows CUDA wheels | Fragile — often breaks on new CUDA/PyTorch versions | Pre-built wheels for every CUDA version |
| Model format | GPTQ (multi-file: safetensors + config) | GGUF (single binary file, self-describing) |
| Dependencies | torch + triton + transformers + accelerate | Just llama-cpp-python |

For production Linux deployment the Triton kernel is still valuable. For the Windows
gaming loop (P5R running alongside the server), llama-cpp-python is the right tool.

### What GGUF Is

GGUF (Generic GPU Unified Format) is the model file format used by llama.cpp.
A single `.gguf` file contains:
- Model weights (quantized)
- Tokenizer vocabulary and merge rules
- Architecture hyperparameters (n_layers, n_heads, etc.)
- Quantization metadata (scale factors, zero points)

This self-description means you can load any GGUF model with one call:
```python
from llama_cpp import Llama
model = Llama(model_path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
              n_gpu_layers=-1,   # offload all layers to GPU
              n_ctx=2048)        # context window
```

No tokenizer download, no config.json, no `trust_remote_code`.

### Q4_K_M: The Quantization Name Decoded

`Q4_K_M` means:
- `Q4` — 4-bit integers for weights (same bit-width as GPTQ)
- `K`  — k-quants: groups of weights share a scale factor, reducing error vs flat Q4
- `M`  — "medium" variant: attention layers use Q5 (one step higher), FFN uses Q4

On an RTX 4060 8GB:
- Model VRAM: ~4.9 GB
- P5R VRAM: ~1.5 GB
- OS + driver overhead: ~0.5 GB
- Total: ~7 GB → 1 GB headroom ✓

Generation speed for a 1-3 sentence response (~50 tokens): **~1.5–2 seconds**.
Well inside our 8-second `DialogueBridge` timeout.

### llama-cpp-python's Chat Completion API

Instead of manually constructing `<s>[INST]` prompt strings (the Llama-2 format),
llama-cpp-python exposes an OpenAI-compatible chat API:

```python
response = model.create_chat_completion(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ],
    max_tokens=80,
    temperature=0.8,
    top_p=0.9,
    repeat_penalty=1.1,
)
text = response["choices"][0]["message"]["content"].strip()
```

The library handles tokenization and special token injection for each supported
model family (Llama-3 uses `<|begin_of_text|>`, Mistral uses `[INST]`, etc.)
automatically from the GGUF metadata.

### n_gpu_layers=-1: Full GPU Offload

`n_gpu_layers=-1` tells llama.cpp to offload **all** transformer layers to VRAM.
With all layers on the GPU, token generation is pure CUDA matmul — no PCIe transfers
during the generation loop, giving maximum throughput.

If VRAM fills up (both P5R and the LLM running simultaneously), set
`n_gpu_layers=28` to offload 28 of 32 layers, keeping the last 4 on CPU. This
reduces peak VRAM by ~0.6 GB with a small speed cost (~+200ms per response).

---

## Chapter 18 — Finding the Per-Line Trigger: StructDiff and Hook Diagnostics

### What We Know So Far

Phase 1 dispatches one LLM call per hang-out session (triggered by session pointer
change). Phase 2 needs one call per NPC speech bubble. We have two unknowns:

1. **Is CMM_EXEC_EVENT firing at all?** The startup log now prints
   `hook:ON` or `hook:OFF`. If it says ON but we never see `CmmExecEvent #N:`
   in the console during a hang-out, the function exists but never gets called
   for gym sessions. If it says OFF, either `IReloadedHooks` is null or the
   signature scan failed.

2. **What struct field changes per dialogue line?** The `StructDiffScanner`
   in the poll loop diffs the first 64 bytes of the session struct every 500ms.
   Any `+0xNN:prev→cur` log line during a hang-out reveals a live field.

### Reading the StructDiff Output

Example log output during a hang-out (hypothetical):
```
[StructDiff] +0x04:00→01
[StructDiff] +0x04:01→02
[StructDiff] +0x04:02→03
```

If +0x04 increments each time the player presses confirm to advance dialogue,
that's our per-line counter! We already know +0x04 was always 0 in Phase 1
testing (we renamed it SESSION_PHASE). But that was during a gym hang-out
with a specific Ryuji scene. Other scenes might behave differently.

Fields to watch for:
- **Counter fields**: unsigned int that increments, resets on new scene
- **Pointer fields**: change when a new dialogue string is pointed to
- **Bitfield flags**: single-bit changes that toggle on each line

### The StructDiff Scanner Design

StructDiffScanner captures 64 bytes at session start (`_hasPrevious = false`
→ first call stores baseline). Every subsequent poll tick: byte-by-byte
comparison, log any that changed, update `_previous`. It auto-resets when
`sessionPtr` changes (new hang-out).

This is passive — we make no writes to memory, only reads. The 500ms poll
interval means we catch changes that last at least 500ms (too fast = aliasing).
For a player that advances dialogue every ~1-2s, 500ms should catch every line.

### If StructDiff Finds Nothing

If NO byte changes during a full gym hang-out with 15 dialogue lines:
- The session struct is read-once (set at hang-out start, not updated per line)
- The per-line state lives in a *different* object (a dialogue VM, not CMM)
- We need to expand the scan: look at [CMM+0x48] child objects, not just
  the session struct itself

Next step would be to use `HexDump(session + some_offset, 64)` at each
offset to find a changing sub-object — or use Cheat Engine's "what accesses
this address" feature to find what the game reads when rendering each line.

---

## Chapter 16 — Connection Resilience: Retries, Health Checks, and Cold Starts

### The Cold-Start Problem

When P5R launches with the mod active, the mod's `Start()` runs immediately.
But the Python server may not have finished loading the 4.9 GB model yet.
First POST to `/generate` → 503 Service Unavailable.

Without retry logic, the first hang-out of every game session gets no LLM
response. With retry logic (3 retries × 8s delay), the C# side waits up to
24s for the model to finish loading — more than enough for the ~20s load time.

### Why Not Retry Indefinitely?

Two reasons:
1. The `CancellationTokenSource` timeout in `DialogueBridge` (default 30s) cancels
   the whole operation regardless. Infinite retries wouldn't exceed 30s anyway.
2. A 503 that persists after 3 retries means the server is broken (missing model
   file, CUDA OOM, etc.) — retrying forever masks the real error.

### The Health-Check Approach (Complementary)

`ServerHealthChecker.CheckAsync()` runs 2s after startup, reads `/health`, and
logs the server's state. This is not a retry loop — it's observability: it tells
the developer what the server thinks of itself without blocking the game.

If `/health → model_not_loaded`, you know to wait. If `/health → ready`, the
mod is ready to dispatch. The 2s delay is a best-effort guess at when the health
endpoint will reflect the final state.

### HTTP Timeout Layering

Three timeout layers protect the system:

```
CancellationTokenSource (30s)  ←  DialogueBridge: outer hard deadline
    │
    └─► HttpClient.Timeout (60s)    ←  LLMClient: TCP-level connection timeout
            │
            └─► Retry loop (3 × 8s = 24s)  ←  503 wait: model loading
```

The HttpClient timeout (60s) is set longer than the CTS timeout (30s) so the
CTS always fires first — we never want HttpClient to terminate a connection that
the business logic still considers in-flight.

### The InferenceInFlightException Fast Path

429 is never retried. The InferenceQueue on the server drops concurrent requests
rather than queueing them — a stale response queued for 10 seconds would appear
after the conversation has moved on. Immediate exception → immediate fallback to
scripted dialogue is the correct behavior.

---

## Chapter 15 — Context Engineering: Making the LLM Sound Like P5R

### Why Prompt Design Matters More Than Model Size

Llama-3.1-8B is a mid-size model. By default it generates plausible English, but
not specifically Ryuji Sakamoto slang. The gap between "competent English" and
"convincingly in-character P5R dialogue" is almost entirely closed through
**prompt engineering** — structuring what the model knows before it generates.

### The Three-Layer Prompt Structure

```
┌──────────────────────────────────┐
│ SYSTEM PROMPT                    │
│  Who the character IS            │ ← identity, arcana, personality
│  What the relationship IS        │ ← rank-tier emotional guidance
│  Hard rules (no meta-commentary) │ ← guardrails
├──────────────────────────────────┤
│ USER PROMPT                      │
│  [Scene context: ...]            │ ← where in the story we are
│  CharacterName:                  │ ← leading-edge: model completes the line
└──────────────────────────────────┘
```

The system prompt is static per hang-out (same character, same rank). The user
prompt is the only dynamic part — it includes the scene metadata string built
by `ContextBuilder` in the C# mod.

### Rank Tiers: Emotional Distance as a Design Parameter

P5R Social Links are a 10-rank progression from strangers to deep bonds. The LLM
has no concept of in-game rank unless we tell it explicitly. The `_tier_note()`
function maps rank to emotional vocabulary:

| Rank | Tier note |
|------|-----------|
| 1-2  | "just met, polite but reserved" |
| 3-5  | "warming up, casual, shared history implied" |
| 6-8  | "close friends, comfortable banter" |
| 9-10 | "deepest bond, trust and vulnerability" |

A rank-4 Ryuji should be friendly but not yet at "best friend" intimacy. A
rank-9 Ryuji should speak more openly about personal struggles. The tier note
shifts the model's word-choice, not just tone.

### Post-processing: Why the Model Ignores Its Own Rules

Even with a rule "do not start your response with the character's name", Llama
sometimes generates:
```
Ryuji Sakamoto: Yo, let's hit the gym!
```

This is because the system prompt is just context — the model assigns it
probability weight, not absolute constraint. Our post-processor strips the
name-prefix pattern (`_NAME_PREFIX` regex) as a hard guarantee.

Similarly, sentence-boundary truncation (`_truncate_at_sentence`) ensures a
clean cut at `max_chars`. Without it:
```
"Dude, what's up? I was thinkin' we could grab some ramen before we head back to"
```
The sentence is incomplete because the character buffer cuts at 80 chars. With
sentence-boundary truncation, the model's natural sentence ending at "." is used.

### What Context Does NOT Yet Include (Phase 2 Target)

The current context string is only:
```
"[Scene 51] Hang-out with Ryuji Sakamoto (rank 4/10). This is a Social Link
 conversation where Ryuji Sakamoto is spending time with the protagonist."
```

It says nothing about:
- What the previous dialogue line was
- What topic Ryuji is discussing
- Whether the player made a choice just before this line

Phase 2 improves this by either:
a) Including the struct-diff field that changes per line as a "line number"
b) Injecting recent event history from a rolling buffer in the C# mod

---

## Chapter 14 — Phase 2: Per-Line NPC Generation Architecture

### The Problem: One LLM Call Per Hang-Out Is Not Enough

Phase 1 gave us one LLM response per session (triggered when the session pointer
changes). A real P5R gym hang-out has ~15-30 dialogue lines and 3-4 player choices.
Phase 2 makes the LLM respond to **each NPC speech bubble** independently.

### Two Strategies for Per-Line Triggering

**Strategy A — Hook-driven**: Intercept the function the game engine calls to
display each new NPC speech bubble. That function is called exactly once per line
render, giving us a natural "new line" signal with no polling.

**Strategy B — Struct-diff polling**: Find a field inside the session struct that
increments (or changes) every time the dialogue advances. Poll it at 100ms; when
the value changes, dispatch a new LLM call.

We are investigating both simultaneously. The CMM_EXEC_EVENT hook (already wired)
is a candidate for Strategy A. For Strategy B, we need to scan more of the struct.

### Why CMM_EXEC_EVENT Might Not Be Per-Line

CMM_EXEC_EVENT (`CmmExecEvent` in Ghidra) fires when the Crimson Mask Manager
executes a social-link **event** — but an "event" in P5R is the entire hang-out
session, not a single dialogue line. That's why:

- We hooked it → it fires once at session start (or not at all for some session types)
- It never fires again until the next session

The per-dialogue-line hook we actually want is deeper in the VMD (Virtual Machine
Dialogue) stack — the function that writes each text string into the dialogue box
render buffer. Finding that function requires Ghidra analysis of the call chain from
CmmExecEvent down to the string-copy site.

### The struct-diff approach: Passive Discovery

While we work on Ghidra analysis, we can instrument the poll loop to **diff the
entire first 64 bytes** of the session struct on every tick. Any byte that changes
between ticks is a candidate for a per-line counter.

```csharp
// Capture baseline snapshot at session start
byte[] _baseline = new byte[64];
Buffer.MemoryCopy((void*)session, Unsafe.AsPointer(ref _baseline[0]), 64, 64);

// Each tick: compare, log changed offsets
for (int i = 0; i < 64; i++)
{
    if (current[i] != _baseline[i])
        _logger.WriteLine($"  +0x{i:X2}: {_baseline[i]:X2} → {current[i]:X2}");
}
_baseline = current;
```

This passive scan requires zero Ghidra work — we let the game tell us which offsets
are "live" during dialogue advancement.

### Rate-Limiting LLM Dispatch Per Line

If we find a per-line trigger, we can't just call `DispatchAsync` on every line:
- Fast dialogue tap (player spamming confirm) could queue 10 calls
- InferenceQueue already drops concurrent calls (429), so only the first survives
- But the user sees long delays as each queued call completes in sequence

The correct design is **leading-edge throttle with dead-time**:

```
line 1 fires at t=0    → dispatch (starts LLM, ~2s)
line 2 fires at t=0.3  → skip (still within dead-time)
line 3 fires at t=0.6  → skip
...
line 8 fires at t=3.1  → dispatch (dead-time expired, new LLM call)
```

Implementation: store `_lastDispatch = DateTimeOffset.UtcNow` on each dispatch.
New dispatch only if `(UtcNow - _lastDispatch) > MinDispatchInterval` (default 3s).

### Hook Diagnostic Checklist

Before building more Phase 2 infrastructure, we must know why CmmExecEvent hook
shows no log output. Possible causes:

1. `IReloadedHooks` is null — shared lib not installed → hook creation skipped entirely
2. Signature scan fails → `InvalidOperationException` → falls back to poll loop
3. Hook IS active but the function is never called during gym hang-outs
4. Hook fires but throws before reaching the log line

To distinguish these: add explicit startup status logging that prints the hook
state after `TryActivateHook()`, and add a hit counter inside `OnCmmExecEvent`
that logs on first fire.

### Architecture After Phase 2

```
                     P5R GAME ENGINE
                           │
          ┌────────────────┴────────────────┐
          │ CmmExecEvent (session start)    │
          │ TextDisplay hook (per line) ←── │ ← Phase 2 target
          └────────────────┬────────────────┘
                           │
              ┌────────────▼───────────────┐
              │    Mod.cs dispatcher       │
              │  leading-edge throttle     │
              │  3s dead-time per session  │
              └────────────┬───────────────┘
                           │
              ┌────────────▼───────────────┐
              │  DialogueBridge            │
              │  POST /generate            │
              └────────────┬───────────────┘
                           │
              ┌────────────▼───────────────┐
              │  Llama-3.1-8B (GGUF)       │
              │  ~2s per response          │
              └────────────────────────────┘
```

---

## Chapter 17 — Session State Management: History, Deduplication, and Context Budget

### Why Session State Is Necessary

Phase 1 dispatches once per hang-out: the LLM has no memory of previous responses
within the same conversation. In Phase 2 (per-line), the LLM might generate the
same line twice if the per-line trigger fires on the same dialogue beat.

Three problems to solve:
1. **Continuity**: Each generated line should build on the previous ones
2. **Deduplication**: The same response should never appear twice
3. **Context budget**: Prior dialogue text grows with each line; Pydantic's
   max_length=1024 on the context field is a hard ceiling

### SessionHistory: Rolling Buffer + Hash Deduplication

SessionHistory stores the last 8 LLM responses as a List<string>. On each new
dispatch, the prior lines are joined with ' | ' and prepended to the context
string as "Prior dialogue: [line1] | [line2]".

For deduplication, a HashSet<int> stores the OrdinalIgnoreCase GetHashCode of
each recorded response. RecordResponse() returns false on a hash collision —
DialogueBridge then suppresses the duplicate and skips the log line.

Hash collisions (two different strings with the same hash) are possible but
extremely rare for dialogue lines. A full string equality check on every entry
would be O(n×m) per insertion; the hash approach is O(1) with acceptable FP rate.

### The Context Budget Problem

With 8 entries of ~120 chars each, prior dialogue can be ~960 chars. Add the
base context string (~150 chars) and you get ~1110 chars — over Pydantic's
max_length=1024 field constraint. The server would return 422 Unprocessable Entity.

Fix: hard-trim the combined context to 1000 chars before building the request.
The 24-char safety margin accounts for ' | ' separators. The trim is a dumb
character count (may split mid-word) but that's fine — the LLM handles partial
sentences gracefully, and the important content (character name, rank, setting)
is always at the start of the context string.

### Context String Priority Order

The context string is built as:
```
"Hang-out with {name} (rank N/10) at {scene}. This is a Social Link conversation...
 Prior dialogue: [line1] | [line2] | ..."
```

Most important info is at the START because if we do trim, we lose the END first.
The scene hints and character info come before the prior dialogue — we would rather
lose "Prior dialogue: line7 | line8" than lose "Ryuji Sakamoto at gym".

---

## Chapter 19 — Pointer Chain Verbose Diagnostics

### Why Verbose Mode Matters

When the game boots and our mod's `TryResolve()` returns `false`, the log says nothing
about *where* in the chain it failed. Was the static pointer zero? Did the first heap
offset land in unreadable memory? Did the second dereference return null?

Without verbose diagnostics, you open Cheat Engine and manually walk the chain to find
the broken link. With verbose mode, the Reloaded-II console tells you exactly which
step failed and what address it tried to read — saving 10–30 minutes per debugging session.

### The Three Failure Modes

A multi-level pointer chain can fail at three distinct points:

```
[moduleBase + SL_STATIC_PTR]  ← Step 0: static address in .data
        |
        v (dereference)
  heap_object_A               ← could be 0 if CMM not yet initialised
        |
        + CMM_SESSION_OFFSET   ← Step 1: field inside heap object A
        |
        v (dereference)
  CmmSession*                 ← could be 0 if no hang-out is active
```

**Failure 0 — unreadable static**: VirtualQuery says the static address is not
readable. Happens if the sig scan located the wrong address or if the game uses
a DRM loader that maps the .data section late.

**Failure 1 — null root pointer**: The static address is readable but contains 0.
Normal before the CMM subsystem initialises (first few frames after game boot).

**Failure 2 — null session pointer**: The root object exists but the session field
is 0 because no Social Link hang-out is currently active. This is the most common
case — the poll loop should silently ignore it rather than logging on every tick.

### Verbose Mode Design

We add a `bool VerboseChain` flag to `GenConfig`. When true, each `TryResolve()` call
logs the step it reached before returning false. When false (the default), silence
is preserved for the null-session case which fires hundreds of times per minute.

The log format uses step numbers so you can correlate with the chain array:

```
[P5RGenSocialLinks] ChainStep 0: addr=0x7FF612345678 → value=0x1A2B3C4D5E6F
[P5RGenSocialLinks] ChainStep 1: addr=0x1A2B3C4D5EBF → value=0x0000000000000000 (null — no active session)
```

The `(null — no active session)` suffix is only appended on the LAST step's zero result
because that's the one that's expected to be zero during idle. Earlier zero results
indicate a real initialisation problem and get a `(UNEXPECTED NULL)` suffix instead.

### Integration with GenConfig

`GenConfig` already supports loading from `GenDialogue.json` next to the DLL. Adding
`VerboseChain` follows the same pattern: a `[JsonPropertyName("verbose_chain")]` property
with a `false` default so existing config files are unaffected.

In `Mod.cs`, the `PointerChainResolver` is constructed once and holds a reference to
the config. This avoids re-reading the JSON on every tick and lets the user enable
verbose mode by editing the JSON (plus a mod reload) without recompiling.

### Practical Workflow

1. First boot: leave `verbose_chain: false` — the poll loop runs silently.
2. If hook or poll loop consistently says "unresolved", set `verbose_chain: true` in
   `GenDialogue.json`, reload the mod, and reproduce.
3. The first logged `ChainStep N` with a null or unreadable address is your broken link.
4. Open Ghidra, navigate to that offset, and update `P5ROffsets.cs` with the correct value.
5. Set `verbose_chain: false` again once resolved.

This is the "printf debugging for memory" approach — cheap, zero-overhead when off,
and completely visible in the Reloaded-II console without attaching a debugger.

---

## Chapter 20 — Text Buffer Write-Back: Injecting Generated Dialogue

### The Gap Between Logging and Injection

Right now `DialogueBridge.DispatchAsync()` calls the LLM, gets a response, and *logs*
it. The actual game text is untouched — Ryuji still delivers his scripted line.
Write-back means overwriting the game's in-memory dialogue string before the renderer
reads it, so the LLM response appears on screen.

### Two Approaches

**Approach A — Direct Buffer Overwrite**
P5R stores dialogue text as a UTF-8 (or Shift-JIS) C-string in the social link session
struct or a nearby heap allocation. If we find the exact byte offset, we can
`Marshal.Copy()` our generated text into the buffer before the renderer reads it.

Risk: The buffer has a fixed capacity (usually 256-512 bytes). If our LLM text is
longer, we overflow adjacent fields and corrupt the struct. Solution: `clean_response()`
already enforces a 200-char maximum and sentence-boundary truncation, so we stay well
under any sane buffer size.

**Approach B — Pointer Swap**
Some engines store a `wchar_t*` pointer field that points to the display string.
If we allocate our own heap string (`Marshal.AllocHGlobal`), write the LLM text
into it, and then overwrite the pointer field, the renderer follows the new pointer
and shows our text. The allocation must stay live for the frame duration, so we
keep it in a field and free the *previous* allocation before overwriting.

P5R most likely uses Approach A (fixed buffer) based on how other Atlus engine games
store dialogue, but Ghidra analysis of the text-display function is required to confirm.

### The Offset Discovery Plan

1. Run `struct_diff_enabled: true` during a hang-out.
2. Look for a `[StructDiff]` line where the offset changes *precisely when dialogue
   text changes on screen* (not when you press a button, but when new text appears).
3. Dump 64 bytes around that offset with `SocialLinkReader.HexDump()`.
4. Match the hex bytes to ASCII/UTF-8 text from the current dialogue line.
5. That confirmed offset is `DIALOGUE_TEXT_OFFSET` in `P5ROffsets.cs`.

### Write-Back Safety

Before writing:
1. `MemoryGuard.IsReadable(ptr + offset, expectedLength)` — confirm the buffer exists.
2. Write exactly `Math.Min(text.Length, maxBufferSize - 1)` bytes.
3. Null-terminate at position `bytesWritten`.
4. Restore the original text on session end in case the game re-reads from the buffer.

The write happens on the thread pool (inside `Task.Run`), which means it races with
the game's main thread. This is inherently racy — the game could read the buffer
*during* our write. A future improvement is to use a semaphore tied to the render
frame boundary, but for a non-commercial mod the race window is small (~microseconds).

### Current Status

`DialogueBridge.DispatchAsync()` has a `// TODO: dialogue write-back` comment where
the write will go. Once `DIALOGUE_TEXT_OFFSET` is confirmed via StructDiff,
the implementation is:

```csharp
unsafe void WriteDialogue(nuint sessionPtr, string text)
{
    byte[] bytes = System.Text.Encoding.UTF8.GetBytes(text + "\0");
    nuint target = sessionPtr + P5ROffsets.DIALOGUE_TEXT_OFFSET;
    if (!MemoryGuard.IsReadable(target, bytes.Length)) return;
    fixed (byte* src = bytes)
        Buffer.MemoryCopy(src, (void*)target, bytes.Length, bytes.Length);
}
```

This is intentionally left as a placeholder until offset discovery is complete.

---

## Chapter 21 — CI Pipeline: Keeping Both Halves Honest

### Why Two-Language Projects Need CI

Our project is split: a C# Reloaded-II mod (Windows only) and a Python FastAPI server
(cross-platform). A bug in either half can break the whole system invisibly. Manual
testing requires launching P5R + the server, which takes minutes and can't run
unattended. CI closes this gap: every push to the branch triggers automated checks
that surface regressions in seconds.

### Our GitHub Actions Setup

The `.github/workflows/ci.yml` workflow has two parallel jobs:

```
┌─────────────────┐     ┌──────────────────┐
│  python-tests   │     │  dotnet-build    │
│  ubuntu-latest  │     │  ubuntu-latest   │
│                 │     │                  │
│  pip install    │     │  dotnet restore  │
│  pytest --all   │     │  dotnet build    │
└─────────────────┘     └──────────────────┘
         ↑ parallel — neither waits for the other
```

Both jobs run on `ubuntu-latest` (the cheapest free runner). The .NET build job
doesn't run the mod in-game — it just confirms the C# compiles cleanly against
the Reloaded-II NuGet packages.

### Why pytest Instead of In-Game Tests

The Python tests use `MOCK_LLM=1` (via the `client_with_mock` fixture) so they
never attempt to load a real model. This makes them:
- **Fast**: ~1s total, no GPU, no model download
- **Deterministic**: mock responses are hardcoded
- **Free**: run on GitHub's free tier Ubuntu runners

Real inference tests (requiring a GPU + 4GB model file) are excluded from CI and
run manually in the development environment before merging.

### Test Coverage Pyramid

```
          [manual in-game]
         real P5R + real GPU
        ─────────────────────
       [local GPU, no P5R needed]
      real model, FastAPI endpoints
     ─────────────────────────────
    [CI — mock server, 86+ tests]
   unit + integration + endpoint tests
  ──────────────────────────────────────
```

The CI tier covers 86 tests across:
- `test_queue.py`: InferenceQueue drop policy, stats, clear
- `test_server.py`: health/ready/stats/model-info/generate endpoints
- `test_generate_endpoint.py`: validation, session_id, 503 handling
- `test_integration.py`: full mock round-trips for all 22 confidants
- `test_prompt_context.py`: system prompt template correctness
- `test_postprocess.py`: OOC removal, emoji, Japanese, truncation
- `test_config.py`: ModelConfig validation bounds
- `test_arcana.py`: roster completeness (22 confidants)
- `test_mock_responses.py`: per-character canned lines
- `test_prompt_builder.py`: build_prompt() fields and format
- `test_tier.py`: rank-to-tier mapping, parametric all 10 ranks

### The dotnet build Gate

The C# build runs `dotnet build` without running any tests — we have no C# unit
test project because the mod's core logic is tested indirectly through in-game
observation. The build gate exists to catch:
- Syntax errors from refactors
- Missing using directives (exactly what we hit with `ILoggerV2` in `ModLogger.cs`)
- NuGet package version mismatches after dependency updates

Future improvement: a C# unit test project using `xUnit` that mocks `ILoggerV2` and
`IReloadedHooks` to test `ModLogger`, `GenConfig`, and `DialogueBridge` in isolation.

---

## Chapter 22 — Mock Architecture: Testing Without a GPU

### The Problem with Real Inference in Tests

Real Llama-3.1-8B inference requires:
- ~4 GB GPU VRAM or CPU RAM
- ~20s cold-start CUDA JIT compile
- A model file that's 4.7 GB on disk
- CUDA toolkit or CPU inference fallback

None of these are available on GitHub Actions free runners. Every CI run would timeout
or fail with OOM. The solution: a mock layer that completely bypasses inference.

### The MOCK_LLM Environment Variable

When the server starts with `MOCK_LLM=1`, the lifespan function sets `_pipeline = _MOCK`
instead of calling `load_model()`. The `_MOCK` object is a plain Python sentinel
(`object()`) — it has no methods, no attributes, no GPU interaction.

In the `/generate` route, we check `if _pipeline is _MOCK:` *before* calling any
inference code. If true, we call `get_mock_response(confidant_id, rank)` and return
immediately. Total latency: <1ms.

This design has a key property: **the entire HTTP path is exercised**, including
Pydantic validation, request parsing, and response serialisation. We're only
substituting the inference step itself.

### The conftest.py Fixture Stack

```python
@pytest.fixture
def mock_pipeline() -> MagicMock:        # ← MagicMock, not _MOCK sentinel
    mock = MagicMock()
    mock.generate.return_value = "Yo, let's crush it today!"
    return mock

@pytest.fixture
async def client_with_mock(mock_pipeline):
    srv._pipeline = mock_pipeline         # ← replaces the real pipeline
    srv._queue = InferenceQueue()         # ← fresh queue per test
    ...
```

Note: `client_with_mock` uses a `MagicMock`, not the `_MOCK` sentinel. This means
`model-info` returns `"real"` mode (the sentinel check fails), but inference actually
calls `mock_pipeline.generate()` → the mocked return value. This is intentional:
it exercises the *real* inference code path (including the queue) while keeping
responses deterministic.

For tests that need `/health → {"status": "mock"}`, use `srv._pipeline = srv._MOCK`
directly (as in `test_model_info_mock_mode`).

### Per-Character Canned Lines

`mock_responses.py` stores 22 character-specific one-liners indexed by confidant ID.
Each one is hand-written to sound like the character:

- Ryuji (8): "Yo, you ready? We're not leavin' till we crush every set!"
- Futaba (10): "SYSTEM ALERT: Fun level exceeding critical threshold!"
- Kasumi (22): "Please, let us focus. Every second here is a second we could be training!"

The `get_mock_response(confidant_id, rank)` function returns:
```
"[MOCK rank N] {canned_line}"
```

The `[MOCK rank N]` prefix makes mock responses instantly identifiable in logs,
so you can tell at a glance whether a log line came from real inference or a test.
The prefix also makes the rank number machine-readable for log parsers.

### What the Mock Tests Prove

With 116 tests (all passing), we have confidence that:
1. **All 22 confidants** can make a successful round-trip through `/generate`
2. **Pydantic validation** correctly rejects out-of-range ranks, long contexts, and negative IDs
3. **Stats counters** increment, accumulate, and reset correctly
4. **Postprocessor** handles edge cases (emoji, Japanese, OOC removal) without crashes
5. **Prompt templates** contain all required fields for every confidant
6. **Tier notes** cover all 10 rank levels without gaps

What the tests do NOT prove: whether real Llama-3.1-8B inference returns valid,
in-character dialogue. That requires manual playtesting, which we've done: the
confirmed LLM response from the previous session was:

> "Dude, what's up? I was thinkin' we could grab some ramen before we head back to..."

In character, rank-appropriate, correct length — the real inference works.

---

## Chapter 23 — Cheat Engine Recon: Script Pool, Line Counter, and the Pre-Load Problem

### What We Found

A live CE session against P5R during Ryuji's Scene 51 (gym hang-out) revealed:

**1. The 16-bit game timer at +0x20/+0x21**
The StructDiff output showed `+0x20` changing every 500ms poll tick by ~60 units,
with `+0x21` catching the carry overflow. Together they form a continuously running
16-bit little-endian clock — NOT a dialogue-line counter. The CMM session struct's
first 64 bytes contain timing state, not line progression.

**2. The dialogue line counter at 0x006FFC28**
CE's "increased by 1" scan + "Find out what writes to this address" revealed:
- Address `0x006FFC28` holds a byte that increments once per dialogue advance
- The write instruction: `mov [rcx+18], eax` at `0x7FFA995C2928` (a system/middleware DLL)
- This means the counter sits at offset `+0x18` inside a struct based at `0x006FFC10`
- Writes stop when the CMM session ends — confirming it's Social Link specific

**3. The pre-loaded script text pool**
A CE string scan for on-screen dialogue found the text at `~0x41DE9104BA` — about
1.6MB past the session struct. Crucially, scrolling around that address revealed
multiple dialogue lines stored **contiguously**. The text doesn't change in-place
when you advance a line; instead, the game moves a read-pointer forward through the pool.

### The Pre-Load Problem

This is the central challenge for write-back. In Unreal or Unity games, a "current
dialogue" string variable gets overwritten on each line. In P5R's BF script system:

```
[Script Text Pool — loaded at hang-out start]
  offset 0x000: "Yo, you ready? We're not leavin'...\0"
  offset 0x030: "Man, I've been thinking about...\0"
  offset 0x060: "Hey, you think we can actually...\0"
  ...
```

A read-pointer (somewhere near `0x006FFC10`) advances through this pool. The game
never "writes" a new string — it just reads further ahead.

**Consequence for write-back**: we cannot replace text in real-time as each line
appears. We must overwrite entries in the pool *before* the player reaches them —
at hang-out start. This means:
1. Detect hang-out (we already do this)
2. Immediately generate 2-3 LLM lines for the opening exchange
3. Find the text pool address and overwrite specific offsets
4. Player sees our text when they advance

### Why the Pool Address Isn't Fixed

`0x41DE9104BA` is a heap allocation — it changes every game restart. To find it
reliably we need a pointer chain from a stable root (a static global or the session
struct). That pointer chain requires Ghidra: we trace from `0x006FFC10` (the counter
struct) back through whatever holds a reference to it, until we hit a module-relative
static offset we can hardcode.

### Session Timing Constraint

The CMM session struct deallocates *before* the hang-out scene fully ends — confirmed
when the mod logged `Hang-out ended` while gym dialogue was still playing on screen.
This means our write-back window is the **opening phase** of each hang-out, not the
closing transition. All LLM writes must complete before the first rank-up animation.

### What This Unlocks Right Now

Even without the text pool pointer, we gained a per-line trigger: when `0x006FFC28`
increments, a new dialogue line just appeared. Wiring this into the poll loop replaces
our once-per-session dispatch with a true per-line dispatch — closer to the full vision.


---

## Ch25 — README as Architecture: How to Document a Multi-Process Systems Project

Every README is simultaneously an introduction, a specification, and a social contract
with future collaborators (including yourself three months from now). A project like
this one has unusual documentation requirements because it spans two programming
languages, two OS processes, a proprietary binary format (BF scripts), live memory
addresses that change on restart, and a GPU-side inference pipeline — none of which
fits into a standard "clone, npm install, run" template.

### Two Audiences, One Document

The first design decision is audience stratification. This project has two distinct
reader types who need completely different information:

1. **The technically curious non-programmer** — a P5R fan who understands what Social
   Links are and why generative dialogue would be interesting, but has never compiled
   C# or traced a pointer chain. They need a 3-sentence "what it does" that anchors
   in something concrete (the actual Ryuji output), not a component diagram.

2. **The systems engineer** — someone who wants to clone and run it, or who's looking
   to contribute write-back support. They need: exact pointer chains, struct offsets
   with provenance, what the two processes are and how they communicate, and what's
   known vs. what's still speculation.

The mistake most technical READMEs make is writing for only the second audience and
burying the lede with component names that mean nothing before the "why" is established.
The structure we chose: hook → what it does (plain English) → architecture → phases →
technical details → setup. This lets both audiences exit at their depth of interest.

### The Mermaid Diagram as a Strict Specification

GitHub renders Mermaid natively in markdown, which means a `flowchart LR` block is
the right choice for architecture documentation in 2024: no external CDN, no image
to regenerate, and the source is version-controlled alongside the code it documents.

The key discipline when drawing the diagram is **encoding what's unknown**:

```
bridge -.->|"write-back (Phase 3)"| writer
writer -.-> pool -.-> renderer
```

The dashed edge (`-.->`) is Mermaid's syntax for a "dotted" connection, which we use
to distinguish "not yet implemented" paths from confirmed data flows. This isn't a
purely aesthetic choice — it forces the diagram to reflect the actual system state
rather than an idealized future version. Any reader who tries to wire up write-back
can immediately see where the gap is, rather than discovering it after cloning.

The corresponding `style` declarations reinforce it:

```
style writer stroke-dasharray: 5 5
style pool stroke-dasharray: 5 5
```

Using CSS inline styles on individual nodes lets us visually differentiate confirmed
components (solid border) from in-progress ones (dashed border) without adding a
separate legend.

### Phase Narrative vs. Feature Inventory

Most project READMEs list features as a flat inventory ("✓ supports 22 confidants").
We chose a phase narrative instead, because this project is fundamentally about a
research process — each phase required new reverse-engineering techniques:

- Phase 1 needed Ghidra + hex dump to confirm struct offsets
- Phase 2 needed benchmarking to discover that auto-gptq was the wrong backend
- Phase 3 needs Cheat Engine + Ghidra to trace the text pool pointer chain

The phase structure communicates *how the project was built*, not just *what it does*.
For a learning-focused project this is the primary form of documentation: it encodes
the intellectual journey, which is at least as valuable as the final artifact.

### Memory Layout Tables and Provenance

Every hex offset in the README carries its source of truth:

```
+0x0B  byte    RankLevel       (rank before this session; Ryuji went 4→5) CONFIRMED
```

The `CONFIRMED` tag signals that this offset was validated against a live game session,
not inferred from static analysis. Offsets without it should be tagged `INFERRED` or
`UNVERIFIED`. This habit prevents the classic reverse-engineering documentation failure
where a hex dump from a specific version is silently assumed to be universal — the
offset may change between P5R patches.

The pointer chain line:

```
Static pointer chain: [p5r.exe + 0x2A63EF0] → [+0x48] → session*
```

follows a convention: `module + static_offset → [dereference chain]`. The `[]` syntax
for a dereference is borrowed from C — `[addr]` means "read the pointer stored at this
address." This notation is compact enough to include inline and unambiguous to anyone
who has done Windows RE work.

### What Makes a README Honest

The hardest discipline is accurately representing the project's current state, not the
planned state. In this README:

- Phase 3 says "write-back stub — needs `bufferPtr` from the pointer chain" — not
  "write-back supported"
- The Mermaid diagram has dashed edges to the renderer
- The setup says "NVIDIA GPU with ≥8 GB VRAM" as a hard requirement, not an asterisk

A README that overpromises is a form of technical debt: it creates a gap between
expectation and reality that future contributors have to debug before they can start
actual work. The setup instructions should describe exactly the state of the code at
the time of writing — if write-back isn't done, the instructions don't include it.

### The `learning.md` Reference

Linking to `learning.md` from the README serves one specific purpose: it signals to
readers who want to understand *why* decisions were made (why llama-cpp-python instead
of auto-gptq? why a single-slot queue? what is the BF script format?) that there is a
document for them. It keeps the README concise by not trying to explain everything, and
it keeps `learning.md` motivated by giving it an audience.

This is the standard pattern for research-oriented projects: the README is the contract,
the journal is the reasoning.

---

## Ch26 — In-Process Heap Scanning and Dialogue Write-Back Architecture

### Why We Don't Need Cheat Engine

CE is a GUI front-end over the same Win32 API calls any process can make. Its "find
what writes" is a VEH (vectored exception handler) memory-access watchpoint; its
string scan is a walk over `VirtualQuery`-enumerated committed pages. Since our C#
mod runs *inside* P5R's process (injected by Reloaded-II), we have identical access
to that address space with no extra privileges required.

The C# equivalents:
- CE "scan all memory for string" → `VirtualQuery` loop + `memcmp` / string heuristic
- CE "find what accesses address" → hardware breakpoint via `SetThreadContext` (not needed here)
- CE "pointer scan" → walk a region for 8-byte values in the heap VA range

### VirtualQuery and the Virtual Address Space Layout

`VirtualQuery(address)` returns a `MEMORY_BASIC_INFORMATION` block describing the
*region* containing `address` — its base, size, state, and protection flags.

The three state values we care about:

| State      | Meaning                              |
|------------|--------------------------------------|
| MEM_COMMIT | Pages are backed by RAM or page file |
| MEM_RESERVE| Reserved, not accessible             |
| MEM_FREE   | Not reserved — safe to skip          |

Heap allocations are always `MEM_COMMIT` + `MEM_PRIVATE` (not mapped from a file).
Code sections are `MEM_COMMIT` + `MEM_IMAGE`. This lets us filter heap regions from
everything else.

To walk the entire address space:
```csharp
nuint addr = 0;
while (VirtualQuery(addr, out MBI mbi, ...) != 0)
{
    if (mbi.State == MEM_COMMIT && mbi.Type == MEM_PRIVATE && readable)
        Probe(mbi.BaseAddress, mbi.RegionSize);
    addr = mbi.BaseAddress + mbi.RegionSize; // advance to next region
}
```

### The Counter Struct Pivot

Brute-force scanning the entire heap is slow and noisy. We have a better anchor:
`0x006FFC10` — the base of the counter struct whose `+0x18` field is the line counter
at `0x006FFC28`.

The CE write instruction was `mov [rcx+18],eax`, so `rcx` held a pointer to the
counter struct. That struct almost certainly also holds a pointer *to the text pool* —
it's the object responsible for tracking position within the pool.

Strategy: read 256 bytes from `0x006FFC10` and treat every aligned 8-byte word as a
candidate pointer. Filter to the heap VA range (roughly `0x1_0000_0000` to
`0x7FF_FFFF_0000` on Windows x64). For each candidate, probe the pointed-to address
for the text pool pattern. This is typically 1–4 probes rather than hundreds.

### Text Pool Heuristic

A BF script text pool has a distinctive fingerprint:
- Multiple consecutive null-terminated strings (5+ in a row)
- Each string is printable ASCII, 10–300 chars long (dialogue, not binary data)
- No binary junk between strings — just `\0` separators

Distinguishing it from other string pools (debug log buffers, localization tables,
path strings) relies on quantity: a dialogue scene has 20–50 lines, so a genuine pool
has more consecutive valid strings than any incidental string region.

### Write-Back Timing

P5R pre-loads the entire scene's text into the pool at hang-out start. The pool
persists until the session ends. Our write window is:

```
[Session struct appears] ←── our window ──→ [Player advances line 0]
         ↑                                              ↑
    CMM hook fires,                          Too late for line 0
    pool scan runs,                          (already rendered)
    LLM request fires
```

LLM response comes back in <2s. If the player takes >2s to press confirm on the
first line (they always do — cutscene animations, reading time), we win the race.

For subsequent lines: `LineCounterMonitor` fires when line N advances. We dispatch
LLM immediately, response arrives in <2s, we write to line N+1. The player has to
read line N+1 and press confirm, which is always more than 2s.

The key insight: **write ahead, not to the current line**. By the time the LLM
responds, line N is already displayed. We write to N+1 (the line the player hasn't
seen yet).

### Write Truncation

Strings in the text pool are packed contiguously:
```
[str0]\0[str1]\0[str2]\0...
```

If we write more bytes than `strlen(strN)`, we corrupt `str(N+1)`. The rule: always
`min(llm_text_length, original_string_length)`. Read the original length first by
scanning for the next `\0` from the target offset, then truncate the LLM text to
`original_length - 1` (leaving one byte for the null terminator).

### Encoding

P5R PC (English, Steam) stores dialogue as **UTF-8** (effectively ASCII for English
text — all chars are < 0x80). The existing `DialogueWriter` stub assumed UTF-16LE
(wide chars), which is wrong for this pool. The fix is straightforward:

```csharp
byte[] encoded = Encoding.UTF8.GetBytes(text);
// Write bytes, not chars
```

For future Japanese locale support: Shift-JIS encoding, not UTF-16. But for now
`UTF8.GetBytes` on pure English text gives identical output to ASCII.

### The IsWritable Guard

Write-back needs a writable page. `MemoryGuard.IsReadable` only checks readable
protection flags. We need `IsWritable`, which additionally requires:
- `PAGE_READWRITE` (0x04) — normal heap pages
- `PAGE_EXECUTE_READWRITE` (0x40) — JIT-compiled or self-modifying code (rare here)
- `PAGE_WRITECOPY` (0x08) — copy-on-write mapped sections

Heap memory is always `PAGE_READWRITE`, so in practice this is just checking that
flag. But we guard it properly to avoid an access violation if the protection ever
changes (e.g., after a `VirtualProtect` call by the game's anti-tamper layer).

---

## Chapter 27: Following Pointer Chains to Nested Sub-Objects

### Why the Counter Isn't at the Top Level

We scanned 256 bytes of the session struct while the player fast-forwarded through
dialogue with Shift+F — dozens of line-advance presses in a few seconds. Only the
game clock at +0x20/+0x21 changed. This rules out any per-line counter in the first
256 bytes of the session struct.

So where is it? Game engines decompose large systems into nested objects. P5R's
CommunityManager session is not a flat blob; it's a root object that holds **pointers
to sub-objects** for each subsystem: dialogue display, voice playback, camera control,
etc. The line counter almost certainly lives inside one of those sub-objects.

### What We Saw at Hang-Out Start

The StructDiff log showed bytes changing at session struct offsets **+0xE0, +0xE8,
+0xF0** at the moment the hang-out began — then staying constant for the rest of the
session. That pattern is diagnostic: it's a set of pointers being written into the
struct at initialization. Before the hang-out, those slots held 0 (or a prior
hang-out's stale pointer). When CMM_EXEC_EVENT fires, the engine allocates sub-objects
for this session and stores their addresses in those slots.

### Pointer Following as a Scanning Strategy

Instead of guessing the counter's offset in the session struct (where it doesn't
exist), we follow the known pointers:

```
session_struct + 0xE0  →  dialogue_manager_A  (8-byte pointer)
session_struct + 0xE8  →  dialogue_manager_B  (8-byte pointer)
session_struct + 0xF0  →  dialogue_manager_C  (8-byte pointer)
```

For each of those addresses, we run the same StructDiff scan we ran on the session
struct — polling every 500 ms, recording which bytes change. A byte that increments
monotonically with each line advance is the counter.

### Reading a Raw Pointer in Unsafe C#

```csharp
nuint ptrAddr = sessionPtr + 0xE0;           // address of the slot
nuint target  = *(nuint*)ptrAddr;            // dereference: read the 8-byte pointer
byte* subObj  = (byte*)target;               // cast to byte* to scan sub-object
```

This is identical to the two-level dereference our PointerChainResolver already does:
`[module + SL_STATIC_PTR]` → `[result + CMM_SESSION_OFFSET]`. We're just adding a
third level, but because the first two levels are stable (module-relative), and we
call `MemoryGuard.IsReadable` before every dereference, it's safe.

### The PointerFollowScanner Design

`PointerFollowScanner` captures the three pointer values once at hang-out start
(right after StructDiff.Reset), then on every poll tick it diffs the 256 bytes each
pointer points to. Output is labelled with the source offset and target address so
we can immediately identify which sub-object contains the counter:

```
[PtrFollow +0xE0 → 0x41DE91A000] +0x18:00→01  ← this is our counter
```

Once we see monotonically incrementing bytes, we have the offset. We then hard-code
that as a new field in `P5ROffsets` — exactly like we did for `CMM_SESSION_OFFSET`
after the Ghidra analysis.

---

## Chapter 28 — HeapScan Artefacts: 0xFF Fill Patterns and Mid-Session Comparison

### What we observed

After a full Takemi hang-out the HeapScan reported 30 candidates — all of them
`0 → 255 (+255)`. The LineCounterMonitor fired with values 48, 144, 0, 112 — jumping
around non-monotonically. Neither result is the dialogue line counter.

### Why all deltas were exactly +255

Windows game heaps commonly fill freed blocks with `0xDD` (MSVC debug heap) or
`0xFE`/`0xFF` (release allocators and custom game allocators for use-after-free
detection). At the moment we called `FindIncreased`, the game scene had already begun
or completed teardown. Objects allocated during the scene were freed; their backing
pages were flood-filled with `0xFF`. Our snapshot captured `0x00` (pages that existed
but held unused/zeroed bytes at session start). By comparison time those same bytes
held `0xFF`. Delta = 255, exact maximum — for every one of them.

The real dialogue counter, if it incremented from 0 to N (where N = lines pressed ≈
20–30), would show a delta of +20 to +30. But 30 slots × +255 entries dominated the
sorted output and our 30-result cap hid anything smaller.

### Why the LineCounterMonitor values made no sense

`0x6FFC28` is ASLR-relocated heap memory that belongs to whichever object the
allocator placed there in *this boot*. The CE session that identified this address ran
in a different launch. The allocator placed the counter object at that address then;
in subsequent boots a different object lives there. The values 48 (`0x30`), 144
(`0x90`), 0, 112 (`0x70`) are bytes written by *that other object* — likely an audio
channel state or animation frame byte — not by the dialogue counter. `HasAdvanced()`
fires on any change; the byte at `0x6FFC28` changes because that other object updates,
not because a dialogue line advanced.

### The two-pronged fix

**Fix 1 — Cap maxDelta at 50.** A dialogue scene with 20–30 button presses produces a
counter increment of 20–30. No one presses 255 lines in a single hang-out. By passing
`maxDelta: 50` we exclude every 0xFF teardown byte and surface only genuine small-step
increments. If the counter *is* in our scan range (0x10000–0x20000000), it will appear
in the filtered results after this change.

**Fix 2 — Mid-session comparison at tick 20 (≈10 s).** The snapshot is taken at
session start. If we compare too early nothing has changed yet; if we compare at
session *end* the teardown noise dominates. The sweet spot is mid-scene: the player
has been pressing dialogue buttons for ~10 seconds but the game has not started
freeing scene objects. At tick 20 of the 500 ms poll loop we call `FindIncreased`
against the original snapshot and log results. Teardown bytes are still 0x00 at this
point; the counter has incremented by however many lines the player pressed. This
gives us a clean signal window.

### The tick counter design

`_sessionTick` is an `int` field reset to 0 each time a new session is detected. It
increments once per poll tick (500 ms). At tick 20 (≈10 s) the mid-session comparison
runs automatically. This mirrors the "two-pass snapshot" mental model from Chapter 28:

```
t=0   : TakeSnapshot()         — baseline, all bytes at rest
t=10s : FindIncreased(max=50)  — lines pressed, no teardown noise
t=end : FindIncreased(max=50)  — may still be useful if session didn't clean up
```

If the counter appears in the t=10s window we have our address. If not, it's outside
0x10000–0x20000000 and we need to extend the scan ceiling.

---

## Chapter 29 — High-Heap Scanning and Cumulative Baseline Diffs

### Why all mid-session results were exactly +50

The mid-session scan fired at tick 20 — exactly 10 real seconds (20 × 500 ms). The
results all showed `+50`. The coincidence is deliberate: these bytes count elapsed game
time in 200 ms units. 10 seconds ÷ 200 ms = 50 ticks. Every byte controlled by this
timer incremented by exactly 50 between snapshot and comparison.

The dialogue counter (one increment per F-press) would show a much smaller delta —
typically +5 to +20 for a normal scene. But since ALL 50 result slots were occupied by
the +50 timer bytes, the actual counter (if present in the low heap) was cut off.

### Why the low-heap scan had no counter at all

The session struct is at `0x424BD57FC0` — roughly 265 GB into the process address
space. The PtrFollow sub-object target is at `0x41E67C9240` — about 174 MB lower.
Both are way above our scan ceiling of `0x20000000` (512 MB). The counter object was
never in our scan window in the first place. Low-heap (< 512 MB) contains DLL images,
CLR stubs, and CRI audio ring buffers — not the game's primary heap.

### Fix 1 — Pivot-based high-heap scan

Instead of a fixed `[0x10000, 0x20000000]` range, pass the live session struct address
as a pivot and scan `[pivot - 256 MB, pivot + 256 MB]`. With the 64 MB total-bytes
cap, we snapshot the first 64 MB of committed+RW pages in that 512 MB window, which
are the pages closest to the session struct. The counter object, being allocated from
the same heap arena, is very likely within 64 MB of the session struct.

### Fix 2 — PtrFollow cumulative baseline

The existing `PointerFollowScanner` diffs each sub-object tick-to-tick, showing
which bytes flip between ticks. That's good for finding volatile pointers but misses a
slowly incrementing counter (it only changes by +1 per press, which is often
indistinguishable from noise in a single-tick diff).

The fix: when a target is first captured, save its bytes as a *baseline*. Then at
mid-session and session-end, `CumulativeDiff()` compares *current* bytes to the
*baseline* — showing the total change since capture. A counter that went 0→15 over 15
presses appears as `+0x18: 0→15 (+15)` even if each tick only produced `+0x18:14→15`.

This turns PtrFollow from a one-tick diff tool into a session-scoped counter detector,
targeting the exact sub-objects the session struct already pointed us to.

### Why PtrFollow is better than a full heap scan for this

PtrFollow scans only the three 256-byte windows at `session+0xE0`, `session+0xE8`, and
`session+0xF0` — the pointers the game engine wrote into the session struct. Any
sub-system that the session struct delegates to lives at one of those targets. The
dialogue counter, being part of the dialogue sub-system, is almost certainly reachable
from one of these pointers. Scanning 3 × 256 B = 768 B is far cleaner than scanning
64 MB of heap noise.

---

## Chapter 30 — DLL Address Poisoning and Timer Array Filtering

### How PtrFollow lost the stable target

`PointerFollowScanner.Update()` accepts any user-mode address in the range
`0x10000–0x7FFF_FFFF_FFFF`. That range includes DLL image sections (`0x7FF8...`).

When the session struct's +0xF0 slot briefly holds a vtable pointer or code pointer
from a DLL, Update() sees a valid user-mode candidate, overwrites the previous target
(`0x41DBC05050`), and resets `_hasSnapshot` — which also resets `_hasBaseline`.
The cumulative history is destroyed. At the mid-session check, no baseline exists.

The fix is a tighter upper bound. Windows x64 loads DLLs at the *top* of user space
(around `0x7FF80000_00000` and above). Game heap objects live in the `0x40...-0x43...`
range — well below `0x7F00_0000_0000`. Adding a `HeapAddressMax = 0x7F00_0000_0000`
ceiling keeps DLL pointers from evicting stable heap targets.

### Why all HeapScan results form regular arrays

Looking at mid-session addresses: `0x423FE55F9E`, `0x423FE55FA2`, `0x423FE55FA6`...
each exactly 4 bytes apart. These are contiguous fields within a struct array — the
game's animation or audio sub-system keeps many timer counters in a flat struct array,
all ticking at the same rate (0.75 Hz in this session). All entries hit the same delta
(+15 in 20 seconds) and dominate the result set.

The dialogue counter is a *single* value, not an array member. Post-processing the
scan results with a stride filter — marking any two hits within 48 bytes and aligned
to a 4-byte stride as array co-members, then removing both — leaves only isolated
addresses. An isolated byte that went up by some small delta matching the number of
lines pressed is the counter.

### The extended SubScanBytes

PtrFollow scanned 256 bytes (offsets 0x00–0xFF) of each sub-object. If the sub-object
header occupies ~200 bytes and the dialogue line index is at offset +0x100 or above,
it was never diffed. Extending SubScanBytes to 512 (offsets 0x00–0x1FF) gives a full
512-byte window into each pointed-to sub-object — covering the range where game-engine
dialogue managers typically store line counters.

---

## Chapter 31 — Recognizing Sunk Cost: Deleting the Scanner Triad

### What we built vs. what we needed

Three scanners were written over several sessions:

| Scanner | Purpose | Verdict |
|---|---|---|
| `LineCounterMonitor` | Monitor a CE-discovered byte counter | Dead: 0x6FFC28 is a stale heap address that changes every boot |
| `HeapCounterScanner` | Snapshot the game heap and find counter by delta | Dead: timer arrays dominated every result; counter obscured |
| `PointerFollowScanner` | Follow session-struct pointers to sub-objects | Dead: DLL address poisoning kept resetting cumulative baselines |

### Why they were unnecessary from the start

`CmmExecEvent` already fires **once per dialogue line** — that is the per-line trigger.
The hook is already wired, already dispatches to the LLM, and already throttles correctly.
We spent several sessions trying to build a second per-line trigger via memory scanning
when a working one existed in the codebase the whole time.

This is the sunk-cost trap in game modding: once you commit to a memory-scan approach
it is easy to keep patching it instead of stepping back and asking whether the goal is
already met by another path.

### What this means for the architecture

The cleanup leaves a simpler `Mod.cs` poll loop:

```
tick: TryResolve session
  → new session: dispatch one LLM call (hook fallback) + start StructDiff
  → same session: StructDiff for passive discovery only
  → session gone: reset
```

The hook path is the primary per-line trigger. The poll loop is purely for session
lifecycle management and text-pool discovery (Phase 3).

### The real Phase 3 blocker

`TextPoolFinder.Find()` returns 0 every session. The LLM generates dialogue but
cannot inject it because there is no confirmed write-back target. Fixing that —
finding where the game stores the displayed dialogue string — is the only remaining
work before Phase 3 is complete. It does not require the counter.

---

## Ch32 — Why TextPoolFinder Fails: Three Root Causes and the Fix

### Root cause 1: Wrong anchor address

`DialogueTextPoolFinder` Phase 1 searches for text by probing the counter struct at
`0x006FFC10`. That address came from Cheat Engine: `mov [rcx+18],eax` — the write
instruction for the line counter — put `rcx = 0x006FFC10` during one specific session.

Two problems:

1. **It is an absolute heap address.** It is not derived from `moduleBase` or the static
   pointer chain. On the next game launch, ASLR places the heap at a completely different
   base. `0x006FFC10` is now garbage or unmapped.

2. **It is the wrong object anyway.** The counter struct tracks *position* within the
   dialogue sequence. The *text* lives in the BF script text pool, which is managed by a
   different subsystem. The counter struct may not hold a pointer to the text pool at all.

The correct anchor is the **session struct** — the object we already own via the static
pointer chain `[moduleBase + 0x2A63EF0] → [CMM + 0x48]`. `GameMemory.cs` confirms that
offsets `+0x10` onwards are "internal CMM pointers." Those pointers are our starting
point, not a CE-discovered absolute address.

### Root cause 2: Fingerprint too strict

The current text pool fingerprint requires every byte to be printable ASCII (0x20–0x7E).
P5R's AtlusScript / FlowScript compiled format embeds control codes inside dialogue
strings:

```
"Dude, what's up?\x01\x00\x00\x04I was thinkin'...\0"
                  ^^^^-- escape code sequence
```

These control codes (bytes 0x01–0x1F) signal pause markers, line-break characters,
speaker name injections, and color changes to the BF interpreter. They are *inside* the
string — between printable characters — so the fingerprint check fails as soon as it sees
`0x01` and abandons the entire string candidate.

The fix: a string is "valid" if at least 50 % of its non-null bytes are printable
(0x20–0x7E) and it contains at least 3 printable characters. Any byte below 0x20 is an
escape code and is tolerated, not rejected.

### Root cause 3: Find() called at session-detection time, not hook-fire time

`Mod.cs` calls `DialogueTextPoolFinder.Find()` inside the poll loop when it first sees a
new session pointer. At that moment, the BF interpreter may have just started loading the
scene script — the text pool may not yet be decompressed into a heap allocation.

`CmmExecEvent` fires **after** the BF interpreter has rendered the current line. By the
time `lineIndex = 1`, the text pool is guaranteed to be in memory. The finder should
be retried inside `OnCmmExecEvent` for the first 3–5 fires if `poolBase` is still 0.

### The pointer-traversal pattern

After anchoring to the session struct, the strategy is:

```
sessionBase
 +0x10: ptr_A  →  probe ptr_A for pool fingerprint
 +0x18: ptr_B  →  probe ptr_B for pool fingerprint
 +0x20: ptr_C  →  scan first 256 bytes of ptr_C for sub-pointers → probe each
 ...
```

This is bounded: the session struct is ~256 bytes → at most 30 candidate pointers
(256/8). Each pointer is probed once (16 KB scan). Two-level traversal doubles the
candidates but is still O(1) relative to session size.

Compare this to the Phase 3 heap scan (±128 MB) which walks thousands of VirtualQuery
regions. The pointer-traversal path is faster and more targeted.

### Diagnostic dump

When all three phases fail, the mod logs a pointer map of the session struct:

```
[TextPoolFinder] Diag: session=0x424... (+0x10)=0x51... HEAP printable=38%
[TextPoolFinder] Diag: session=0x424... (+0x18)=0x52... HEAP printable=12%
[TextPoolFinder] Diag: session=0x424... (+0x20)=0x53... HEAP printable=71% ← candidate?
```

One session of log output will show us exactly which pointer leads to text. Once we have
that offset, it becomes a constant in `GameMemory.cs` and the pointer traversal collapses
to a single dereference.

---

## Ch33 — CMM Struct Population Timing and Poll-Retry Strategy

### The timing gap

`Find()` was called once at session-detection time — the exact moment the poll loop
first saw a non-zero session pointer. At that moment, the CMM struct at the session
address looks like this:

```
+0x00: 0E 00 00 00 00 00 00 00  ← ConfidantId=14, SessionPhase=0
+0x08: 0E 00 02 03 28 00 00 00  ← CmmIdRepeat, EventType, RankLevel, SceneNumber
+0x10: 00 00 00 00 00 00 00 00  ← NOT YET POPULATED
+0x18: 00 00 00 00 00 00 00 00  ← NOT YET POPULATED
```

The internal CMM sub-object pointers (at +0x10+) are populated asynchronously by the
BF interpreter during scene initialization — typically within one poll interval (≤1 s)
of session detection.

Evidence from the StructDiff log: when the first diff tick fired for Ryuji Scene 51,
it showed massive changes including:

```
+0x90:49→10 +0x91:0C→A1 +0x92:CE→3D +0x93:E5→12 +0x94:5C→42 ...
```

Reading those "to" bytes as a little-endian 64-bit int:
`[10][A1][3D][12][42][00][00][00]` = `0x0000_0042_12_3D_A1_10` — a valid heap address.

The struct IS full of pointers, they're just not there yet when `Find()` runs.

### Poll-retry pattern

The fix: track a retry counter per session, reset it on new-session and session-end,
and retry `Find()` on every poll tick while `PoolBase == 0` and retries < 10.

```
Session detected → Find() attempt 0 (struct empty → nothing found)
1 tick later    → Find() attempt 1 (struct populated → Phase1 finds pool!)
```

This costs one extra poll cycle (usually ≤1 s) to discover the pool, which is fine
because the player almost never advances dialogue within the first second of a hang-out.

### Phase 3 false-positive threshold

The Phase 3 heap scan found `0x4210DC6000` with exactly 4 strings (threshold = 4)
for Ryuji Scene 51. A real dialogue pool has 20–50 strings; 4 is incidental data.

Fix: Phase 3 requires ≥8 strings (targeted Phase 1/2 keep 4). The heap scan must be
much more confident before we write-back into an arbitrary memory region.

---

## Ch34 — CE Reveals: No Pool, Per-Line Allocations, and the Real Hook Address

### What CE actually showed us

After tracking the dialogue string "It's a gym over in Shibuya" (Ryuji Scene 51)
at address `0x41DBE3B059`, the "Find what accesses this address" list contained:

1. **Dozens of System.Text.Unicode.dll instructions** (different addresses, different bytes)
   — These are all our own mod's Phase 3 backward heap scan. The JIT-compiled
   `CountPoolStrings` byte loop generates many native instructions that appear in the
   .NET BCL's JIT memory range. Our scanner was reading directly through the dialogue
   text region without recognising it.

2. **One p5r.exe instruction: `0x1405A857B - mov rsi,r10`**
   — This is `p5r.exe + 0x5A857B`. The only genuine game-side access to the text buffer.
   (`mov rsi,r10` is register-to-register, so CE may be showing the adjacent instruction
   rather than the memory access itself. The real memory-touching instruction is within
   a few bytes of `0x5A857B`.)

### Key finding: no contiguous text pool

As dialogue advanced through ~15 lines, four distinct text buffer addresses appeared:
```
0x41DBE3B059  ← "It's a gym over in Shibuya..."
0x4178E70D99  ← next line
0x41DCFCA895  ← next line
0x4214A00069  ← next line
```

Each dialogue line is a **separate heap allocation**. The BF interpreter allocates a
fresh buffer per displayed string. There is no contiguous pool of strings to pre-fill
and no single `poolBase` address to anchor write-back to.

This explains every Phase 3 failure: we required ≥8 consecutive strings in one region.
That structure simply does not exist in P5R's dialogue system.

### CmmExecEvent confirmed wrong

CmmExecEvent fires **once at hang-out initialisation** — before the save file loads us
into an ongoing scene. In all logs with dialogue advancing (choices selected, lines
skipped), not a single `CmmExec #N:` line appeared. The hook is active but the function
is never called again after scene setup. Using it as a per-line trigger was incorrect.

### The real hook: p5r.exe + 0x5A857B

The instruction at `0x1405A857B` is the only p5r.exe code that accessed the dialogue
text buffer. To understand whether it fires once per line or every frame — and what
register holds the text pointer — the function must be inspected in Ghidra.

**Next step:** Open p5r.exe in Ghidra → navigate to `0x1405a857b` → identify the
enclosing function. Once we know the function signature and calling frequency, we write
an `IAsmHook` that captures the text buffer address and overwrites it with LLM output
before the game renders the line.

### Write-back strategy revision

Old strategy (scrapped):
- Find a contiguous text pool → pre-fill all slots at session start

New strategy:
- Hook the game function at p5r.exe+0x5A857B
- At hook entry, one register = current line's text buffer address
- Overwrite `[reg]` with LLM-generated text
- Existing 3-second throttle prevents duplicate calls
- No pool finder, no CmmExecEvent, no per-session scanning

---

## Chapter 35 — Why the CE Call Stack Showed .NET Frames, and How Source-Address Filtering Solves It

### What the CE breakpoint actually showed

Setting a software breakpoint at `p5r.exe+5A857E` (the `REP MOVSB` inside `FUN_1405a8570`)
caused CE to break immediately with these call stack entries:

```
P5R.exe+4FFF1C   System.Net.Net...
P5R.exe+4EC675   Reloaded.Mod...
P5R.exe+4EC221   0AD44BD0,Syste...
```

And the source register was `RSI = 0x0AD453E8` — a 32-bit address well below 4 GB.

This is **not** the game's dialogue copy. Reloaded-II hosts the .NET CLR inside the
P5R process. The CLR runtime performs its own internal memory operations — GC compaction,
string interning, JIT code emission — and many of those paths call into the same native
memcpy that P5R uses. A static breakpoint at that function fires for all of them
indiscriminately.

**Key insight:** The dialogue text addresses we confirmed with CE were all in the range
`0x41XXXXXXXX` — above `0x4000000000` (256 GB). The CLR and runtime copies are in the
low 4 GB range (`0x0XXXXXXXX`). A single 64-bit comparison separates them with zero
false positives.

### Hooking `FUN_1405a8590` directly (the memcpy dispatcher)

`FUN_1405a8590` at `p5r.exe+0x5A8590` is the game's `memcpy` dispatcher. Its Microsoft
x64 calling convention maps perfectly:

```
RCX = dst   (destination buffer)
RDX = src   (source — the dialogue text buffer)
R8  = count (byte count, typically 512 for dialogue)
```

Reloaded-II's `CreateHook<T>` with `[Function(CallingConventions.Microsoft)]` wraps
this function cleanly: we call `OriginalFunction(dst, src, count)` first (so the copy
always completes), then inspect the destination for dialogue content.

### The three-tier filter

The hook must be extremely fast because memcpy is called constantly. Three cheap guards
eliminate almost all non-dialogue calls before any string inspection:

```
1. src <  0x4000000000  →  return  (CLR/runtime low-heap copy, skip)
2. count < 10 || count > 600  →  return  (too small or too large for dialogue)
3. !_inActiveSession  →  return  (no social link in progress)
```

Only if all three pass does the hook scan the destination buffer for printable text
(≥5 printable bytes in the first 64 bytes). That check costs ~64 iterations at most,
and only during confirmed dialogue scenes.

### `_inActiveSession` — cached boolean avoids session-chain lookup in hot path

`TryResolve()` walks a pointer chain on every call. That is fine in the 500 ms poll
loop but would be expensive called from inside the game's memcpy thousands of times
per second. Instead, the poll loop sets a `volatile bool _inActiveSession` whenever it
detects or loses a session. The hook reads this single boolean — one cache-coherent
load — with no lock.

### Why we read from `dst` (destination), not `src` (source)

The `OriginalFunction` call copies `count` bytes from `src` to `dst`. After that
call returns, `dst` has the same content as `src` — but accessing `dst` is safer:

- `src` is the BF interpreter's internal buffer. Writing to it could corrupt the
  interpreter's own state if it re-reads from that address later.
- `dst` is the display buffer. Writing to it is exactly what LLM write-back needs
  to do — overwrite the destination so the renderer sees our text.

For Phase 1 (observation), we only log from `dst`. For Phase 2 (write-back), we
will `Marshal.Copy` the LLM byte array into `dst` in-place, immediately after the
native copy completes and before the renderer reads from it.

### Next step: confirm captures match dialogue, then add write-back

After building and deploying, run a hang-out and check the log for:
```
[MemcpyHook] (512B src=0x41DBE3B059): "It·s a gym over in Shibuya····"
```
Once we confirm dialogue text appears there (and not noise), we pipe `dst` through
`_bridge.DispatchAsync()` and write the LLM response back to `dst`.

This is simpler than everything we built. One mid-function hook, one buffer overwrite.

---

## Chapter 36 — Why Dialogue Text Shows as "" and How Hex-Prefix Logging Reveals BF Format

### The problem
Our dedup diagnostic logged many entries at addresses like `0x41E19DCC75 n=168: ""` — sizes
(132–268 bytes) and timing (around StructDiff state changes) that look exactly like dialogue,
but the content field is completely blank.

### Why the content shows "" when text is present
Our display loop was:
```csharp
for (int i = 0; i < n && p[i] != 0 && sb.Length < 64; i++)
    sb.Append(p[i] >= 0x20 && p[i] <= 0x7E ? (char)p[i] : '·');
```
Two bugs that compound:
1. **Early termination on `p[i] != 0`**: BF buffers routinely have null bytes inside control-code
   sequences BEFORE the text starts. A single 0x00 in the header stops the scan immediately.
2. **`·` is never appended to `sb`**: non-printable bytes produce `·` in the char, but the
   ternary is inside the loop — we append `·` for non-printable, but only if `p[i] != 0`.
   If byte 0 is a control code (0x01–0x1F), that's ≥0x20? No — 0x01 < 0x20 → `·` appended,
   but if byte 0 IS literally 0x00, the whole loop terminates with `sb = ""`.

### What BF format actually looks like
P5R's BF (Binary Format) dialogue buffers have a header section:
```
[01] [len] [type] [00] [00] [00] ... actual Shift-JIS or ASCII text ... [00]
```
Byte 0 is 0x01 (message start code). Byte 1 is a length. After several control code bytes,
the readable text begins. Our `p[i] != 0` guard terminates at the very first null in the
header — long before reaching the text.

### The hex-prefix fix
Show the first 4 bytes as hex regardless:
```csharp
string hex4 = n >= 4
    ? $"[{p[0]:X2} {p[1]:X2} {p[2]:X2} {p[3]:X2}]"
    : "[short]";
```
Remove the `p[i] != 0` early-termination guard and scan the FULL buffer:
```csharp
for (int i = 0; i < n; i++)
    if (p[i] >= 0x20 && p[i] <= 0x7E) sb.Append((char)p[i]);
```
Now the output looks like:
```
[MemcpyHook][NEW] 0x41E19DCC75 n=168 [01 2A 00 03]: "It's a gym in Shibuya."
```
The `[01 2A 00 03]` prefix immediately identifies the buffer as BF dialogue (0x01 start code),
while the full-buffer scan finds the ASCII text that comes after the control codes.

### Why we read from `dst` not `src`
The original buffer at `src` is fine to read but we use `dst` because after the original
memcpy runs, `dst` holds the completed copy in a stable location we own for the duration of
the hook. Reading `src` is equally valid here, but using `dst` is the habit we build for
write-back: we will eventually need to WRITE to `dst`, so reading from it first confirms we
can address it correctly.

### What happens next
If the hex prefix shows `[01 xx]` for those 132–268B entries, we have confirmed dialogue text
flowing through the hook. The next step is filtering to `p[0] == 0x01` (BF message start) and
capturing the full text from `dst`, which is the buffer we will overwrite with LLM output.

---

## Chapter 37 — Why the Memcpy Hook Missed Dialogue and How BF Program-Counter Probing Works

### Why the memcpy hook never showed dialogue text
P5R loads the BF script for a hang-out scene **once**, before the dialogue UI appears.
The copy sequence:
1. User selects "Hang out with him" from the menu
2. The BF interpreter calls `FUN_1405a8570` (our hooked function) to DMA the compiled
   script file from the asset archive into heap memory
3. The game transitions to the hang-out scene and sets up the session struct
4. **Our poll loop now detects the session** → `_inActiveSession = true`
5. For every line of dialogue thereafter, the BF interpreter reads DIRECTLY from the
   already-loaded heap buffer — no further memcpy calls are made

So step 2 (the copy) happens BEFORE step 4 (session detection). `_inActiveSession = false`
during the copy → we skip it. We see every subsequent asset copy but never the one that matters.

### Two confirmed anchor points from prior instrumentation

| Anchor | Source | Value |
|--------|--------|-------|
| `session + 0x20` = BF PC | StructDiff log (16 StructDiff events matched 16 dialogue advances) | uint16, starts at 0x04E1, advances ~53 bytes/line |
| `*(session + 0x0E8)` = BF buffer base | TextPoolFinder Diag `+0x0E8 → 0x4247624840 = session + 0x80` | nuint pointer into heap |

The BF program counter at `session+0x20` tells us WHERE in the BF script the interpreter
is currently executing. The buffer pointer at `*(session+0x0E8)` tells us the base address
of the BF script in heap memory. Together: **current dialogue line = bfBase + pc**.

### The BF instruction at bfBase+pc
P5R's BF format stores a text-display instruction as roughly:
```
[opcode 1B] [speaker 1B] [flags ...] [text bytes...] [00 terminator]
```
The first 8 bytes are binary opcodes; the actual dialogue text starts somewhere in the
middle of the instruction payload. By logging "all printable bytes in the first 64 bytes
at bfBase+pc", we surface the English text from the current instruction.

### Change detection (avoiding per-tick spam)
The PC at `session+0x20` is stable between dialogue advances (the game waits for the
player to press X before advancing). We hash `pc + first 8 bytes at lineAddr` to get a
32-bit snapshot. Only when the snapshot changes (= a new line loaded) do we log.
This gives exactly one `[BFLine]` entry per dialogue line, at 200 ms latency from the
button press — fast enough for LLM write-back.

### Write-back plan (next milestone)
Once `[BFLine]` entries confirm we're reading the correct text:
1. At CmmExecEvent fire: read current line text from bfBase+pc
2. Send to LLM (`DialogueBridge.DispatchAsync`)
3. When LLM responds: find the text region in bfBase+pc (skip past opcode bytes)
4. `Marshal.Copy(llmBytes, 0, (nint)(lineAddr + textOffset), llmBytes.Length)`
5. Write null terminator at llmBytes.Length

---

## Chapter 38 — Session Struct Layout Varies; Robust BF Buffer Discovery via ptr+PC Scan

### What broke
We hardcoded `*(session + 0x0E8)` as the BF buffer base from one test session
(`0x4247624760`). A different Ryuji hangout session (`0x418A26AAE0`) has the heap
pointers at completely different offsets (`+0x1B8`, `+0x1C0`, `+0x1F0`, `+0x1F8`).
Hardcoding a single offset breaks whenever the game allocates a different session
struct layout (different scene type, different confidant rank, different play order).

### Why session struct layout varies in P5R
P5R uses C++ inheritance for its social link / event system. Different scene types
derive from a common `SocialLinkSession` base class but add their own virtual method
tables and fields. A gym hangout might be `GymHangoutSession` (adds a gym-data pointer
at +0x80), while a festival scene might be `FestivalHangoutSession` (completely different
field layout). The base class fields (like the BF PC at +0x20) are fixed; the derived
fields (like the BF script buffer pointer) are at class-specific offsets.

### The robust approach: ptr+PC scan
Two things are STABLE across session types:
1. **PC at session+0x20** — StructDiff confirmed it advances ~53 bytes per dialogue line
   in BOTH sessions tested. It's in the base class.
2. **The BF buffer, when indexed by the PC, contains printable English text.**

Instead of guessing which heap pointer is the BF buffer, we probe ALL of them:
```
for each 8-byte value V in first 512 bytes of session struct:
    if V is a heap address (> 256 GB):
        probe = V + pc
        if MemoryGuard.IsReadable(probe, 64) AND printable_chars(probe, 64) >= 8:
            → this is the BF buffer pointer; log it
```

The BF script + pc lands on the current dialogue instruction, which contains ASCII text.
Every other heap pointer + pc lands on binary data (animation frames, texture headers, etc.)
which has <8 printable bytes in 64. The separation is strong because dialogue text has
~50-70% printable bytes while packed float/bone data has <10%.

### Expected output
```
[BFLine] pc=0x04F7 [sess+0x1B8+pc] [01 21 00 05 ...]: "It's a gym over in Shibuya."
```
The `[sess+0x1B8+pc]` part tells us which pointer offset in the struct is the BF buffer —
which we can then use directly for write-back.

---

## Chapter 39 — Transient Pointers, Long Text Runs, and Session-Tick Caching

### Why BFLine only found garbage (the ≥8 threshold was too low)
The probe looked for ≥8 consecutive printable bytes at `[ptr + pc]`. Binary data (textures,
animation matrices, packed floats) routinely has accidental 8-byte printable runs — `"UUXp"`,
`"rbrb"` etc. Those fired as false positives from `sess+0x0A0`. The pointer there is a game
C++ struct, not a BF script. Dialogue lines have LONG runs (20-50 chars) because English
sentences are long. Binary data almost never exceeds 12 consecutive printable bytes.

### Why the BF buffer pointer disappears before BFLine fires
The early StructDiff events (during scene load) show bytes `session+0x40` through `+0x44`
changing as a group:
```
+0x40:20→F0  +0x41:00→61  +0x42:00→A1  +0x43:00→10  +0x44:01→42
```
As little-endian uint64: `0x4210A161F0` — a valid P5R heap address. This is the BF script
buffer, live during scene initialization. Two StructDiff ticks later it's cleared back to a
non-heap value. By the time `[BFLine]` first fires (when PC = 0x001F), the pointer is gone.

### The fix: scan every tick, cache on first hit
Instead of probing `[ptr + pc]` at the moment of dialogue advance, we scan ALL heap pointers
in the session struct on EVERY poll tick and look for any whose target contains a run of
≥20 consecutive printable bytes. An English dialogue script always satisfies this (even the
shortest Ryuji line is 20+ chars). Binary data almost never does. We save (`_bfBufferBase`)
the first match, and then `ProbeBfLine` uses the cached address forever after.

### Why 20 consecutive printable bytes works as the discriminator
A BF script file has dialogue lines like "It's a gym over in Shibuya. Pretty damn cheap too."
= 50 chars in a row. The BF binary header before the first line is ~31 bytes of binary, then
30-50 chars of ASCII text. Our 0-to-511 scan of each candidate region will always find this
run within the first 100 bytes. Texture data, vertex buffers, and C++ structs very rarely
have runs longer than ~12 printable bytes — "DDS ", "GFS0", "FBN0", etc. are exactly 4 chars.

## Chapter 40 — Self-Referential Pointers Are False Positives; String-Count Selects the Real BF Script

### What we found (and why it was wrong)

`TryFindBfBuffer` fired with `[BFBuffer] FOUND sess+0x098 → 0x41DADBA5A0`. The session struct
was at `0x41DADBA510`. The "buffer" address is `0x41DADBA510 + 0x090 = 0x41DADBA5A0`.
The pointer at session+0x98 pointed to session+0x90 — **eight bytes before the pointer itself,
still inside the session struct**. We accepted it because it had maxRun=34, but the text
("Strongest, most powerful Personas.") is the Chariot arcana's social link *ability description*,
stored inline as a field in the C++ session object. It is NOT the BF dialogue script.

### What a self-referential pointer means in C++

In a C++ class layout, a pointer at field +0x98 that points to the same object's field +0x90
is usually one of three things:

1. **A vtable or interface pointer** — the object's secondary vtable table (IUnknown-style).
2. **A pointer to an embedded sub-object** — the social link session inherits from a base class
   that contains the ability description as a fixed-size char array at offset +0x90. The pointer
   at +0x98 lets code treat that sub-object polymorphically.
3. **A back-pointer** — for doubly-linked structures.

In all three cases, **the target is data that lives inside the session struct itself**, not a
separately allocated BF script buffer. Real script buffers are allocated by the BF interpreter
when it opens a `.bf` file: they are `malloc`/`new` calls that land at heap addresses completely
unrelated to the session address.

### The correct filter: skip any target within the session scan window

```
if (ptr >= session && ptr < session + sessionScan) continue;
```

If the target address falls inside the first N bytes of the session struct, the pointer is
self-referential. Skip it. A real BF script buffer is a separate allocation — its address will
be thousands or millions of bytes away from the session struct.

### Why we also need to scan ALL candidates, not just the first

`TryFindBfBuffer` returned immediately after the first hit. That is correct for SELECTION but
wrong during DIAGNOSIS. With the self-ref filter in place, the first non-self-referential
candidate with ≥20 printable bytes might still not be the BF script (there could be other
string pools or asset metadata reachable from the session). We log ALL candidates with:
- `maxRun` — length of the longest printable run
- `strings` — count of null-terminated segments ≥ 4 printable chars each

### Why null-terminated-string count is the right discriminator

A real BF dialogue script has ONE null-terminated string per dialogue line:

```
"It's a gym over in Shibuya. Pretty damn cheap too.\0"
"C'mon, I'll show you the way.\0"
"Here we are... Protein Lovers gym!\0"
```

→ many short strings → high `strings` count.

The ability description field might be ONE long string ("Strongest, most powerful Personas.") or
two strings (the label + the body). Either way, `strings` is low (1–3).

By selecting the candidate with the HIGHEST `strings` count (after applying the self-ref filter),
we pick the allocation with the most dialogue-like structure, regardless of what scene is active
or what the dialogue text says. This heuristic works across all confidants and all scenes because
every BF script is a sequence of null-terminated dialogue lines.

### The expanded scan window: 512 → 1024 bytes

The previous 512-byte scan covered offsets +0x000 through +0x1F8 (64 pointer slots). The session
struct for complex scenes (gym, festival) appears to be 3 KB+ based on the maximum BF PC value
observed (0x0C4C = 3148). The BF script pointer could easily be at offset +0x300 or +0x400.
Expanding to 1024 bytes (128 slots, offsets through +0x3F8) costs one extra VirtualQuery call
and gives us more surface area to find the script pointer.

---

## Chapter 41 — Three Bugs That Hid the BF Script Pointer (And How We Fixed All Three)

### Background: what we confirmed with CE

CE memory viewer confirmed the BF script at runtime:

- BF buffer base: `0x4178E79D48`
- Opcode byte `0x05` (dialogue) at `+0x20` offset: `0x4178E79D68`
- First dialogue byte `'g'` of "gym over in Shibuya": `0x4178E79D69`
- PC at the moment that line fired: `0x0020`
- Therefore bfBase = opcodeAddr − PC = `0x4178E79D68 − 0x0020 = 0x4178E79D48` ✓

Despite this confirmed address, our C# scanner logged zero `[BFBuffer] CAND` entries.
Three overlapping bugs were hiding the buffer.

---

### Bug 1: `CountPoolStrings` breaks on the first bad segment — and BF files start with null bytes

The BF file format begins with a **32-byte binary header** before the first instruction:

```
0x4178E79D48:  00 00 00 00 00 02 00 00 00 00 5D 08 00 00 5D 73
0x4178E79D58:  00 00 00 A8 F2 05 FF FF 00 00 00 00 00 00 00 00
0x4178E79D68:  05 67 79 6D 20 6F 76 65 72 ...   ← first instruction
```

The very first byte at `+0x00` is `0x00` (null). `CountPoolStrings` reads:

```csharp
// scan until null
if (b == 0) { strEnd = i; break; }
// after loop:
if (printable < MinPrintableChars) break; // ← hits this, printable=0
```

Because the first "string" is zero-length with zero printable chars, the function
hits `break` immediately and returns `count = 0`. Every pointer to the BF buffer
fails the `count >= MinPoolStrings` check.

**The fix:** change all three `break` exits in `CountPoolStrings` to `continue` with
position advancement. A bad/empty segment is skipped; only the good ones are counted.
This mirrors what `CountNullTermStrings` in `Mod.cs` already does correctly.

```csharp
// OLD:
if (printable < MinPrintableChars) break;
// NEW:
if (printable < MinPrintableChars) { pos = (strEnd >= 0 ? strEnd + 1 : pos + 1); continue; }
```

With the fix, the scanner skips past the 32-byte header, reaches the first null-terminated
dialogue string "gym over in Shibuya\0", and counts it.

---

### Bug 2: `minRun = 20` threshold is one character too high

The second gatekeeper in `TryFindBfBuffer` is:

```csharp
if (maxRun < minRun) continue;   // minRun = 20
```

"gym over in Shibuya" = **19 printable characters** — one short of the cutoff.
The BF script is discarded before `CountNullTermStrings` is even called.

Other strings in the same scene ("C'mon, I'll show you the way", "Here we are...")
might be longer, but those appear deeper in the BF file. Our `probeScan = 512` covers
only the first 512 bytes. The first 32 bytes are the header; the first instruction
occupies bytes 32–~70; subsequent instructions are at 70–512. Even if there is a 20+
char string further in, **the maxRun filter must survive the header region** to ever
reach those strings.

**The fix:** lower `minRun` to **12**. This still excludes pure-binary objects (which
rarely have 12 consecutive printable bytes) while passing any English dialogue sentence.
Also expand `probeScan` from 512 → 2048 so more of the BF file is covered.

---

### Bug 3: Session struct scan range too narrow

The session struct scan range was 512 bytes in `StructDiffScanner` and 1024 bytes in
`TryFindBfBuffer`. Both C++ game engines (Unreal-adjacent, AtlasEngine) store the BF
interpreter as a **separate object** reached through a pointer chain:

```
session struct  +0x?? → BFInterpreter object  +0x?? → BF script buffer
```

The interpreter pointer within the session struct can be at ANY offset; there is no
reason to assume it lands in the first 128 pointer slots (1024 bytes). Real game session
structs are hundreds to thousands of bytes. We expand the scan to **4096 bytes** (512
pointer slots), which covers sessions up to ~4 KB without a noticeable performance hit
(the VirtualQuery + memcpy cost per tick is negligible at a 200ms interval).

---

### Why Phase 3 (bidirectional heap scan) cannot help here

Phase 3 scans ±128 MB / ±32 MB around the session struct address. But:

- session: `0x420F6B77C0`
- BF buffer: `0x4178E79D48`
- gap: `0x420F6B77C0 − 0x4178E79D48 ≈ 2.57 GB`

The buffer is **2.57 GB below** the session struct — far outside the scan window.
Phase 3 must remain as a hook-based safety net (fires only once per session start,
not every poll tick). The only path to the BF buffer from our code is through the
pointer chain in the session struct (Phase 1) or through intermediate objects (Phase 2).

---

### Summary: the three fixes together

| Bug | Symptom | Fix |
|-----|---------|-----|
| `CountPoolStrings` breaks on null header | BF buffer scores 0 strings | Skip bad segments (`continue`) instead of `break` |
| `minRun = 20` too high | BF buffer discarded before string counting | Lower to 12; expand `probeScan` to 2048 |
| Session scan too narrow (512–1024 B) | Interpreter pointer at offset > 1024 | Expand to 4096 B in both scanner and diff tracker |

---

## Chapter 42 — CMM Event Script vs. BF Dialogue Buffer; Why the Memcpy Hook Is the Real Path

### What the expanded scanner found (and why it is NOT dialogue)

After the three-bug fix, `TryFindBfBuffer` found four candidates:

```
CAND sess+0x330 → 0x4244BC59B0  strings=11  preview: "]C fZDBc]~]}...]Ryuji]HO@\DBRyuji Sakamoto]"
CAND sess+0x600 → 0x4244BC5A10  strings=11  preview: same buffer, +0x60 offset
CAND sess+0xA60 → 0x4244B0D9F0  strings=10  preview: "&BiB4-,KC``f Bb..."
CAND sess+0xCE0 → 0x41DB8E9A90  strings=9   preview: "MODEL/CHARACTER/3001/FIELD/BF3001_200.GAP"
```

The SELECTED buffer (`sess+0x330 → 0x4244BC59B0`) contained:

- The character name "Ryuji Sakamoto" and "Chariot" (arcana) as isolated strings
- A repeating `]` (0x5D) byte as what appears to be a control code
- "DB", "HO", "HOZ" patterns — CMM instruction mnemonics
- ZERO complete English dialogue sentences

If this were the BF dialogue buffer, we would expect to see "It's a gym over in Shibuya", "Yo let's go", etc. Instead we see encoded identifiers and arcana labels.

The buffer at `sess+0xCE0 → 0x41DB8E9A90` contains the literal file path
`MODEL/CHARACTER/3001/FIELD/BF3001_200.GAP`. That is the **asset descriptor** —
a metadata structure that tells the CMM *where* to load the BF script from disk.
It is not the script itself.

### The two-layer architecture of P5R dialogue

P5R uses a two-layer scripting system:

```
Layer 1 — CMM event graph (what we found at 0x4244BC59B0):
  ┌─────────────────────────────────────────────────────┐
  │  Social link state machine: rank checks, flags,     │
  │  branching, character identifiers, arcana labels    │
  │  Format: proprietary Atlus CMM opcodes + 0x5D tags  │
  └─────────────────────────────────────────────────────┘
         │
         ▼ loads from file path in descriptor
Layer 2 — BF dialogue buffer (what CE confirmed at 0x4178E79D48):
  ┌─────────────────────────────────────────────────────┐
  │  Raw BF script: 32-byte header, then instructions   │
  │  Opcode 0x05 + null-terminated English text         │
  │  "gym over in Shibuya\0", "Yo let's go\0", etc.     │
  └─────────────────────────────────────────────────────┘
```

The CMM event graph orchestrates which scenes play. When it reaches a dialogue
node, it triggers the BF interpreter to execute a LINE from the BF dialogue buffer.
The BF dialogue buffer is loaded from `BF3001_200.GAP` on demand, into a heap
region ≈2.57 GB below the session struct — unreachable by proximity-based scanning.

### What the memcpy hook at 0x5A8570 intercepts

When the BF interpreter reaches a `0x05` (dialogue) instruction, it:
1. Reads the null-terminated string from the BF script buffer
2. Calls the inner copy function at `p5r.exe+0x5A8570` (REP MOVSB) to transfer
   the text to a freshly-allocated render buffer
3. The renderer reads the text from the render buffer and writes it to the dialogue box

The memcpy hook is already wired and active — it was just doing nothing:

```csharp
private unsafe void OnGameMemcpy(nuint dst, nuint src, nuint count)
{
    // Diagnostic logging disabled — ...
    _memcpyHook!.OriginalFunction(dst, src, count);
}
```

**This hook IS the correct interception point.** We need to:
1. Filter calls where `dst >= HeapLow` (destination in game heap, not CLR)
2. Filter calls where the copied content contains ≥10 consecutive printable ASCII bytes
3. Log `src`, `dst`, `count`, and the copied text preview

The `dst` of such a call IS the render buffer the game reads to display dialogue.
To replace the text, we write our LLM output into `dst` before returning.

### Why not write back to the BF script buffer directly?

Writing to the BF script buffer (if we found it) would be fragile:
- The buffer is the DECODED static script; modifying it changes ALL future occurrences
  of the same string (any repeat line or scene reuse)
- The render buffer at `dst` is per-line: allocated fresh, filled, rendered, freed —
  no aliasing problems, no stale state

The memcpy hook write-back approach is per-line, stateless, and safe.

### What the PC advance pattern means in the CMM layer

The PC at session+0x20 is the **CMM program counter**, not the BF script PC.
It advances through the CMM event graph nodes:
- PC=0x0033: CMM node 0x33 (possibly a "start dialogue" node)
- PC=0x006D, 0x00A9, ...: subsequent CMM nodes (conditionals, flag sets, etc.)

The CMM PC and the BF PC are different counters. CE confirmed the BF PC was also
at `session+0x20` *during the specific BF scene loading event* we were watching —
but in the steady-state gameplay, what we're reading is the CMM PC.

---

## Chapter 43 — Why the Memcpy Filter Caught Float Data and How Vowel Counting Fixes It

### What we saw in the log

```
[MemcpyDialogue] src=0x41F0D1973F dst=0x41EE9D11A0 n=280 run=14:
  "==>>*>L>n>>>>>>>?????;?DDD?L?UUU?]?fff?n?www??DD????UU???..."
```

Every single logged entry had:
- count = 280, 288, 384 — multiples of 16 (SIMD alignment)
- Patterns: `>>>`, `???`, `===`, `fff`, `www`, `DDD`, `UUU` repeating
- Zero lowercase vowels (a, e, i, o, u)

None of it was dialogue.

### Why IEEE 754 floats look like printable ASCII

A 32-bit float in P5R's vertex buffers commonly looks like:

```
  0.0f  → 00 00 00 00   (not printable)
  0.5f  → 00 00 00 3F   → '?' = 0x3F is printable
  1.0f  → 00 00 80 3F   → '?' printable
  2.0f  → 00 00 00 40   → '@' printable
  0.25f → 00 00 80 3E   → '>' printable
  0.333f→ AA AA AA 3E   → '>' + 0xAA (non-printable)
```

The UPPER byte of small positive floats is always 0x3E–0x44, which maps exactly to
`>?@ABCD` — all printable. A vertex buffer of N×(x,y,z) floats produces a dense run
of printable `>`, `?`, `@`, `A`, `B`, `C`, `D` bytes. Our `maxRun ≥ 10` filter accepted
these as "looks like text."

Additionally, colour channels stored as float (0.0–1.0) produce the same 0x3F–0x40
range, and UV coordinates produce `<`, `=`, `>`, `?` (0x3C–0x3F). This accounts for
the repeated `fff` (0x66 = 'f' is the exponent byte for floats near 2²³) and `www`
(0x77 = 'w').

### Why English dialogue text is different

English text like "It's a gym over in Shibuya":

```
I  t  '  s     a     g  y  m     o  v  e  r     i  n     S  h  i  b  u  y  a
49 74 27 73 20 61 20 67 79 6D 20 6F 76 65 72 20 69 6E 20 53 68 69 62 75 79 61
```

- Has SPACES (0x20) scattered throughout (roughly every 4–6 bytes)
- Has many LOWERCASE VOWELS: a=0x61, e=0x65, i=0x69, o=0x6F, u=0x75
- "gym over in Shibuya" alone contains: a, o, e, i, i, u, a = **7 vowels**

Float vertex data in the `>?@ABC` range (0x3E–0x44) contains NONE of: a, e, i, o, u.
Colour/UV data in the `<=>?` range (0x3C–0x3F) also contains none.

### The fix: require ≥ 3 lowercase vowels in the copied content

```csharp
int vowels = 0;
for (nuint i = 0; i < count; i++)
{
    byte b = d[i];
    if (b == 'a' || b == 'e' || b == 'i' || b == 'o' || b == 'u') vowels++;
}
if (vowels < 3) return;
```

This single check eliminates every false positive from the previous run (they all had
0 vowels) while passing any real English dialogue sentence (minimum ~2–3 vowels even
for the shortest lines).

### Secondary fix: reduce max count from 512 to 150

A P5R dialogue line is never 384 bytes of text. The largest lines observed are ~80
characters. Setting max count to 150 eliminates the 280/288/384-byte vertex copies
even before the vowel check runs.

---

## Ch44 — BF Script Load Timing: Why count > 150 Was the Wrong Filter

### The access type problem

Cheat Engine's "find what accesses this address" captures any instruction that
touches the watched address — read OR write. When CE watched `0x4178E79D48` (our
confirmed BF dialogue buffer) and reported `mov rsi, r10` at `p5r.exe+0x5A857B`,
it could mean two very different things:

- **Case A (write access)**: R10 = source of data, RDI = `0x4178E79D48` = destination.
  The game is LOADING the BF script into that buffer. `count` = size of entire BF
  script (several KB — far larger than our 150-byte cap).
- **Case B (read access)**: R10 = `0x4178E79D48` = source, RDI = some render buffer.
  The game is reading one line's text OUT of the BF buffer per display tick.
  `count` = string length (10–80 bytes — well inside our cap).

Our hook fired ZERO times during dialogue display despite lines advancing. Case B
would fire PER LINE — so zero fires proves Case B is wrong. The correct reading is
**Case A**: the BF script is loaded in one bulk copy (or a small number of them)
BEFORE dialogue starts, and our `count > 150` filter silently dropped every one.

### Timeline of BF script loading

```
T=0    Player triggers hang-out
T+5ms  P5R loads BF scene script from GAP archive:
           FUN_1405A8570 called with r10=src, rcx=dst_bfBuffer, r8=script_size
           ← our hook fires here, count >> 150, REJECTED by filter, dst never stored
T+15ms Session struct pointer becomes valid → our poll loop detects session
T+20ms TryFindBfBuffer() runs → no pointer path from session struct to bfBuffer
```

The 150-byte cap was designed for per-line interception (Case B), but the actual
access is a one-shot bulk load (Case A). The buffer address we need is the `dst`
of that bulk load — and we were throwing it away.

### The large-copy-log approach

Instead of filtering by content (vowels, printable runs), record the **destination
address** of every large heap-to-heap copy. Then, when the session is detected,
probe each recorded destination for BF content:

```csharp
// In OnGameMemcpy:
if (count >= 500 && count <= 500_000)
{
    lock (_largeCopyLock)
    {
        if (_largeCopyDsts.Count < 150)
            _largeCopyDsts.Add(dst);   // record dst; probe at session start
    }
}
```

At session start, `TryFindBfBuffer` scans `_largeCopyDsts` in reverse (most recent
copy first) and runs a BF content probe on each:

```
BF fingerprint probe:
  1. maxRun ≥ 12 contiguous printable bytes (rules out pure binary/compressed data)
  2. ≥ 3 null-terminated strings of ≥ 4 chars with ≥ 2 vowels each
     (rules out mesh IDs like "mesh_920", texture keys, vertex layout names)
```

Mesh names ("Ryuji_Hair", "field_gym") fail criterion 2 because they have too few
vowels (0–1) per token. English dialogue ("gym over in Shibuya", "You've been
training here, huh?") has 3–7 vowels per sentence — passes cleanly.

### Clearing the log between hang-outs

`_largeCopyDsts` is cleared when a hang-out ends so the next session starts with a
fresh list. The new hang-out's BF script load fires AFTER the clear but BEFORE the
poll loop detects the new session — by construction the list always contains the
current session's BF script by the time `TryFindBfBuffer` runs.

### Why the session struct scan still fails (and stays as a fallback)

The confirmed BF buffer (`0x4178E79D48`) and the session struct are in different
heap arenas — the session struct has NO reachable pointer to the BF buffer within
any scan window. The copy-log approach bypasses this completely: it records the BF
buffer address directly at the moment it's written, without needing a pointer chain.


---

## Chapter 44 — Reading the BF Script: base pointer, PC, and extracting a msg_id

### The two offsets that matter

We confirmed two fields in the social-link session struct:

| Offset | Width | Meaning |
|--------|-------|---------|
| `+0x18` | 8 bytes (pointer) | Base address of the compiled BF script in memory |
| `+0x20` | 4 bytes (uint32) | PC — byte offset from that base to the *current* instruction |

`bfBase = *(nuint*)(session + 0x18)` is **constant** for the entire hang-out; it is the load address of the `.bf` scene file.  
`pc = *(uint*)(session + 0x20)` is written by `mov [rbx+20], eax` (confirmed by CE hardware write-trap) and advances with every BF opcode dispatch.

Reading `bfBase + pc` gives us the raw bytes of the instruction the interpreter paused at **after each dialogue box advance**.

### Why `bfBase` appears "below the heap"

Our earlier `HeapLow` filter (`> 0x4000000000`) silently rejected `bfBase` because the BF file is mapped into a non-heap region (~`0x700D7038` range).  Removing the HeapLow guard and accepting any non-zero value was the fix.

### The 32-byte instruction window

Each pause captures a 32-byte snapshot starting at `bfBase + pc`.  
Empirical data from a Ryuji gym hang-out (msgId = 0x0348 = 840):

```
pc=0x109: 00 00 00 00 00 00 00 09 00 01 03 05 01 [48 03] 00 ...
pc=0x139: 00 00 00 00 00 00 00 09 00 01 06 05 01 [48 03] 00 ...
pc=0x16A: 00 00 00 00 00 00 09 00 01 09 05 01    [48 03] 00 ...
pc=0x2D9: 00 00 00 00 00 00 00 0A 00 01 03 5B 00 [48 03] 00 ...
```

`48 03` in little-endian = **0x0348 = 840** — the BMD message index for this scene.  
The varying bytes (`03`, `06`, `09`) are the sub-line index within the same message.

### The msg_id extraction heuristic

Scanning the 32-byte window for the **last** little-endian uint16 in the range `[0x0200, 0x07FF]` robustly identifies the msg_id:

- Sub-line indices (`03`, `06`, `09` read as `0x0003`, `0x0006`, `0x0009`) are below 0x0200 — filtered out.
- Opcode bytes (`09`, `0A`, `05`) interpreted as LE pairs (`0x0009`, `0x000A`, `0x0005`) are also below 0x0200 — filtered out.
- The actual msg_id (`0x0348`, `0x02D6`, `0x02C5`) is always in the 0x0200–0x07FF band.

Taking the **last** occurrence in the 32-byte window avoids the varying sub-line bytes that appear *before* the stable msg_id.

**Confirmation threshold = 3 consecutive windows** with the same value.  This eliminates false positives (values that appear in 1–2 instruction windows but are not real msg_ids).

### The full scene break-down (Ryuji gym, rank-1)

| PC range     | Confirmed msg_id | Interpretation |
|---|---|---|
| 0x109–0x16A  | 0x0348 = 840   | Ryuji's gym invite speech (3 sub-lines) |
| 0x1C1–0x21E  | 0x02D6 = 726   | Follow-up / response segment (4 sub-lines) |
| 0x360–0x3BE  | 0x02C5 = 709   | Third dialogue segment |

### What comes next: BMD lookup and write-back

The BMD file (Binary Message Data) is the static string table that maps msg_id → null-terminated dialogue text.  The game computes `bmd_base + offset_table[msg_id]` — it never stores a raw string pointer, which is why CE write-breakpoints on the text address found nothing.

To inject LLM text we must:
1. Find `bmd_base` in memory (the loaded BMD file).
2. Read `offset_table[msg_id]` to get the string offset.
3. Write LLM text in-place at `bmd_base + offset`.

**Short-term beta path**: The social-link description text is written *inline* in the session struct at `session+0x9B0` and is readable/writable.  Writing LLM text there lets us display generated content in the hang-out UI while the full BMD injection is wired up.

---

## Chapter 45 — P5R Memory Layout and Finding the BMD

### Two distinct memory regions

P5R uses two address-space zones for runtime data:

| Zone | Address range | What lives here |
|------|---------------|-----------------|
| **Lower 4 GB** (memory-mapped) | `0x0–0xFFFF_FFFF` | PAK-file assets loaded by the game's resource manager. Non-heap. VirtualQuery shows `MEM_MAPPED`. |
| **Upper heap** | `> 0x4000_0000_0000` (HeapLow) | CLR runtime, game-engine objects, session structs, dialogue string pools. `VirtualQuery` shows `MEM_PRIVATE`. |

The BF script is at **`0x702594D8`** (≈ 1.8 GB) — solidly in the mapped-file region.  
The BMD for the same scene is loaded from the same PAK archive, so it **must be nearby** in the same 4 GB window.

Our earlier `HeapLow` filter silently rejected `bfBase` because we assumed all game data is in the upper heap. It isn't.

### The BF instruction format (confirmed from 32-byte windows)

Every BF dialogue instruction is exactly **12 bytes**:

```
Offset │ Size │ Field
───────┼──────┼──────────────────────
   0   │ 2 B  │ opcode (LE uint16) — 0x0009, 0x000A, 0x000B … increments per exchange
   2   │ 2 B  │ arg1   (LE uint16) — low byte = 0x01 (const), high byte = line_idx
   4   │ 2 B  │ arg2   (LE uint16) — varies (speaker param / sub-message index)
   6   │ 2 B  │ msgId  (LE uint16) — BMD message index (0x0348, 0x02C7, …)
   8   │ 4 B  │ trailing zeros
```

The 32-byte window captures **two complete instructions** back-to-back, confirming the 12-byte stride.

### BMD search strategy

The BMD file is somewhere near `bfBase` (±32 MB) in the mapped-file region.  
Characteristics that distinguish BMD from the BF script (binary bytecode):

| Property | BF script | BMD |
|---|---|---|
| Printable ratio | Low (binary opcodes) | High (>60% — all text) |
| Null-separated strings | 0–5 | 20–200+ |
| Content | opcode bytes | English dialogue sentences |

Scan using `VirtualQuery` to walk committed regions near `bfBase`.  
For each region: compute printable%, count English sentences (≥8 chars, ≥2 spaces, ≥3 vowels).  
The region with the highest sentence count that is NOT the BF script itself is the BMD.

Once found, log the first 30 null-separated strings to:
1. Confirm it's the right BMD (contains rank-specific dialogue).
2. Reverse-engineer the offset table format (likely `uint32 count` + `uint32 offsets[count]` + strings).

---

## Chapter 46 — False Positive Analysis and Heuristic Refinement

### What went wrong

The scanner found `0x6FE52000` (size 0x11000, 68 KB) and declared it the BMD with 34 "sentences".  
The logged strings were 3D skeleton bone names:

```
"b l b seifuk02"   (len 14)
"Bip01 R Toe0"     (len 12)
"b r Blur_asi01"   (len 14)
```

Why they passed the old filter (`len >= 8`, `spaces >= 2`, `vowels >= 3`, `ascii >= 90%`):

| Check | Bone name "b l b seifuk02" | Verdict |
|---|---|---|
| len >= 8 | len = 14 ✓ | passes |
| spaces >= 2 | 3 spaces ✓ | passes |
| vowels >= 3 | e,i,u in "seifuk" ✓ | passes |
| ascii >= 90% | fully ASCII ✓ | passes |

Result: every bone name with an underscore prefix counted as a "sentence".

### The discriminating property: average word length

| Content type | Example | Average word length |
|---|---|---|
| Bone names | "b l b seifuk02" | 14 / 4 = 3.5 (padded by 3 single-char tokens) |
| Real dialogue | "Dude, you're seriously the only one" | 34 / 6 ≈ 5.7 |
| Labels, filenames | "Bip01 R Toe0" | 12 / 3 = 4 |

The correct discriminators for dialogue:
1. **Minimum string length 25** — bone names top out around 20 chars.
2. **Average word length ≥ 4** — `len / (spaces + 1) >= 4` eliminates "b l b …" style token sequences.
3. **Vowel count ≥ 4** — raised from 3 to reduce borderline passes.

Also noted: `bfBase` is NOT static — it changed from `0x702594D8` to `0x6FFC1258` between two game sessions. It must always be read live from `session+0x18`, not cached across sessions. The current code already does this correctly.

---

## Chapter 47 — Diagnostic Scan: Why the BMD Is Still Hidden

### Two root causes from the ±32 MB scan failure

**Cause 1 — Scan fires every poll tick.**  
The gate `if (_confirmedBfBase != 0 && _bmdBase == 0)` fires once per poll interval (~500 ms) because `_bmdBase` stays 0. Walking ±32 MB of address space on every tick is wasteful and produces log spam. Fix: add a `_bmdScanDone` bool that flips true on first attempt.

**Cause 2 — BMD header precedes strings.**  
P5R `.bmd` (message) files start with a binary header: a magic word, string count, then an offset table (4 bytes per string). That header section can be several KB. If the BMD region is, say, 68 KB and the offset table takes the first 10 KB, scanning only the first 8 KB of the region will hit zero real sentences. Fix: scan 32 KB per region.

**Cause 3 — Window may be too narrow.**  
`bfBase` shifts between `0x6FAA35F8` and `0x702594D8` across runs — the BF script is not at a fixed address. The BMD for the same event is usually in the same CPK archive, but after decompression it may land in a different VirtualAlloc range. Expanding to ±128 MB covers almost the entire lower-4 GB mapped-file zone.

### Diagnostic approach

Instead of stopping at the first 10-sentence region, log EVERY committed readable region that has ≥1 qualifying sentence. Output its base, size, protect flags, first 4 raw bytes (to identify file magic), and sentence count. This gives a full map of what text regions exist in the window and why the threshold isn't being met.

---

## Chapter 48 — BMD Control Codes and Printable-Run Scanning

### Why null-terminated scanning fails on BMD data

P5R BMD dialogue strings are NOT simple null-terminated strings. They use embedded control codes where `\x00` followed by a non-zero byte is a formatting command (speaker label, line break, color change), not a string terminator. A single displayed line like:

```
"Strongest, most powerful Personas."
```

is stored in the BMD as something like:

```
53 74 72 6F 6E 67 65 73 74 2C  "Strongest,"
00 01                           [control: line break?]
6D 6F 73 74 20 70 6F 77 65 72 66 75 6C  "most powerful"
00 02                           [control: next segment]
50 65 72 73 6F 6E 61 73 2E     "Personas."
00 00                           [true terminator]
```

Our null-terminated scanner stops at the first `\x00 01` and sees "Strongest," (10 chars) — too short to count as a sentence. The full string is never reconstructed.

### The diagnostic signal: `[0D 00 xx xx]` regions

Two PAGE_READONLY (prot=0x2) regions appeared with this magic:

| Region | Magic | LE uint16[0] | LE uint16[1] |
|--------|-------|------|------|
| `0x190000` | `0D 00 E4 04` | 13 (version?) | 1252 (string count?) |
| `0x1B0000` | `0D 00 B5 01` | 13 (version?) | 437 (string count?) |

1252 strings in one file, 437 in another. msgId=840 < 1252 so it fits in the first file. These are the BMD files — but they scored `s=1` because the control-code nulls broke our scanner.

### Printable-run scanning

Instead of looking for null-terminated strings, scan for **contiguous runs of printable ASCII bytes** (0x20–0x7E), ignoring the intervening control code bytes:

```
Scan byte-by-byte.
When byte is printable (0x20–0x7E): accumulate into run.
When byte is non-printable (<0x20 or ≥0x7F): end run, evaluate.

If run length ≥ 12 AND letters ≥ 60% AND spaces ≥ 1 → count it.
```

"Strongest," + control code + "most powerful" would produce TWO qualifying runs instead of being killed by the null. A BMD region would score 50–100+ runs vs. 0–3 for model/shader assets.

---

## Chapter 49 — BMD Offset Table Lookup and In-Place Write

### The BMD binary layout

Now that we have `_bmdBase = 0x190000`, we need to find exactly where msgId N's text lives inside the file. The BMD format is:

```
Offset 0x0000:  uint16  version    = 13  (0x0D 0x00)
Offset 0x0002:  uint16  msgCount   = 1252 (0xE4 0x04)
Offset 0x0004:  uint32[msgCount]   offset table
                  Each entry is a byte offset FROM the start of the BMD
                  to the beginning of that message's data.
Offset 0x138C:  <string data>      Shift-JIS text + control codes
```

To get the bytes for msgId N:
```
uint* offsetTable = (uint*)(bmdBase + 4);
uint  msgOffset   = offsetTable[N];          // e.g. 0x2A10
byte* msgData     = (byte*)bmdBase + msgOffset;
```

`msgData` now points to the first byte of that dialogue entry.

### Why the region is PAGE_READONLY and how to write to it

The game's asset loader maps BMD files from disk with `CreateFileMapping` + `MapViewOfFile(FILE_MAP_READ)`, which gives PAGE_READONLY pages. You cannot write to them directly — the CPU raises an access violation.

`VirtualProtect` changes the memory protection flags on a page range at runtime without unmapping the view:

```
VirtualProtect(addr, size, PAGE_READWRITE, out oldProtect)
// ... write your bytes ...
VirtualProtect(addr, size, oldProtect, out _)  // restore
```

This works because the OS kernel only checks protection flags on access — the underlying file mapping stays intact. After restoring PAGE_READONLY, any further writes AV again (good — prevents accidental corruption).

### In-place length constraint

The original string at `msgData` has a fixed byte length determined by the gap between consecutive offset table entries:

```
slotSize = offsetTable[N+1] - offsetTable[N]   (for N < msgCount - 1)
```

We can only overwrite up to `slotSize` bytes. If the LLM text is longer, we truncate. If shorter, we null-terminate and leave the trailing bytes as-is (the game reads until its own terminator).

### Write strategy

1. Encode LLM text as ASCII bytes (plain Latin fits; no Shift-JIS needed for our generated output)
2. VirtualProtect → PAGE_READWRITE
3. `Marshal.Copy` or a direct `byte*` loop to write bytes
4. Write a null terminator at `min(llmLen, slotSize - 1)`
5. VirtualProtect → restore original protect

---

## Chapter 50 — Why 0x190000 Was the Wrong BMD: Glyph Tables vs. Dialogue Scripts

### The false positive

Our scan locked on address 0x190000 because it matched three filters:
1. Magic `[0D 00]` at byte 0
2. `msgCount = 1252` ≥ our 1000-threshold
3. PAGE_READONLY memory-mapped region

But the reverse-anchor scan returned **zero hits** — values 0x0120, 0x013E, 0x0158 did not appear in the first 1024 bytes. This is the smoking gun.

Then look at what TryLogBmdStrings found in the big 15837-byte block at +0x0223:

```
............................... !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwx.
```

The leading dots are bytes ≥ 0x80 (Shift-JIS lead bytes). Then there's a space (0x20), followed by the entire ASCII printable sequence 0x21–0x7E. That IS the ASCII character table, ordered by code point. This region is a **glyph descriptor table** (probably font metrics or character-set mapping), not dialogue BMD. The count 1252 = number of glyphs, not messages.

### Why the scan-all-memory approach breaks

Scanning the full lower 4 GB for `[0D 00]` + large count was always fragile: the magic bytes and size threshold are coincidentally satisfied by non-BMD resources. The correct approach is to go directly to the source — the BF (FlowScript Binary) that the game loaded for this conversation.

---

## Chapter 51 — Parsing the BF Section Table to Find the Embedded BMD

### BF binary format (P5R, little-endian, version 3)

A BF file is a structured binary with a fixed header followed by a section table. Every field is little-endian:

```
Offset  Size  Field
0x00    4     field00 (= 0)
0x04    1     compressionFlag
0x05    1     userId
0x06    2     version (int16)
0x08    4     fileSize
0x0C    4     magic: "FLW\0" = 0x00574C46  ("FLF\0" for some variants)
0x10    2     sectionCount (int16)
0x12    2     localIntVariableCount
0x14    2     localFloatVariableCount
0x16    2     padding
0x18    ...   SectionHeader[sectionCount], each 16 bytes
```

Each `SectionHeader` is 16 bytes:

```
+0x00  int32  firstElementOffset   ← byte offset from BF buffer start to section data
+0x04  int32  elementSize          ← bytes per element
+0x08  int32  elementCount         ← number of elements (= section byte length when elementSize=1)
+0x0C  int32  reserved (= 0)
```

The sections appear in a fixed order by type:

| Index | Type               | elementSize | Content                          |
|-------|--------------------|-------------|----------------------------------|
| 0     | ProcedureLabelTable| 0x20        | Procedure descriptor entries     |
| 1     | JumpLabelTable     | 0x04        | Jump target offsets              |
| 2     | Text (bytecode)    | 0x04        | BF instruction stream            |
| 3     | MessageScript      | 0x01        | Embedded BMD (raw bytes)         |
| 4     | StringTable        | 0x01        | Null-terminated C strings        |

**Section 3 is the BMD.** Its data starts at `bfBase + sections[3].firstElementOffset`.

### Why this is better than memory scanning

The BF binary is the authoritative source. The game engine already knows where the BMD is — it loaded it from the same BF file. By reading the section table, we get the exact address with zero ambiguity, no false positives, and no dependence on magic-byte heuristics.

### Implementation

```csharp
private unsafe void TryScanForBmd()
{
    nuint bfBase = _confirmedBfBase;
    if (bfBase == 0) return;

    byte* bf = (byte*)bfBase;
    uint  magic        = *(uint*)(bf + 0x0C);
    short sectionCount = *(short*)(bf + 0x10);

    for (int s = 0; s < sectionCount && s < 8; s++)
    {
        byte* sh = bf + 0x18 + s * 16;
        int firstOff  = *(int*)(sh + 0);
        int elemSize  = *(int*)(sh + 4);
        int elemCount = *(int*)(sh + 8);

        // Section 3 = MessageScript, elementSize=1 means raw bytes
        if (s == 3 && elemSize == 1 && elemCount > 0)
        {
            _bmdBase = bfBase + (nuint)(uint)firstOff;
            // → now _bmdBase points to the real embedded BMD
        }
    }
}
```

Once `_bmdBase` is set correctly, `TryWriteToBmd` can read the actual offset table and write LLM text to the right message slot.

---

## Chapter 52 — bfBase Points to the Instruction Buffer, Not the File Header

### What the BF hex dump revealed

Dumping 64 bytes at `bfBase=0x620F4FF8` showed a repeating 16-byte pattern:
```
08 00 02 04 33 00 C3 02 00 00 00 00 00 00 00 00
08 00 02 05 3C 00 C3 02 00 00 00 00 00 00 00 00
```

This is NOT a BF file header — it's the **in-memory instruction buffer**. Each 16-byte record is one expanded BF instruction: [opcode][type][value][metadata][padding]. The game's BF runtime expands compact 4-byte file instructions to 16-byte in-memory structs for faster dispatch.

`session+0x18` = bfBase = **start of the TEXT section** (instruction buffer), NOT the BF file start. So our section-walk code was parsing instruction data as section headers, producing garbage (magic=0x00000000, counts like 100 million).

### Three-strategy discovery

Since we can't use bfBase as a file header, we use three approaches in order:

**Strategy 1 — Session struct scan**: The BF runtime likely stores a pointer to the BMD somewhere in the session struct (the object that holds all dialogue state). We walk session+0x00..0xF8 in 8-byte steps, and for each value that's a readable pointer, check the first 4 bytes for `[0D 00]` with a sane message count.

**Strategy 2 — Backward FLW magic scan**: The BF file was memory-mapped as a contiguous block. The instruction buffer (`bfBase`) is the Text section inside that file. Scanning backward from `bfBase` through the same VirtualQuery region for `FLW\0` (magic=0x00574C46) at offset +0x0C finds the file base. From there, the standard 16-byte section table at fileBase+0x18 gives us section[3] = BMD.

**Strategy 3 — Region forward scan**: Fallback — scan the entire VirtualQuery region containing bfBase for any `[0D 00]` pattern with 5 ≤ msgCount ≤ 3000.

### Lesson: verify pointer provenance before parsing

`session+0x18` is documented as "BF script base", but the BF system distinguishes between:
- The FILE base (where the file mapping starts, contains the header)
- The TEXT base (where the instruction bytecode lives, = what `session+0x18` holds)

Always verify what a pointer actually points to by dumping a few bytes before assuming it's the start of a structured format.

### The problem: offset table location is unknown

We have a binary blob (`_bmdBase`, 0x11000 bytes). We know the first string of actual message text sits at `bmdBase+0x0120`. But reading `bmdBase+8+msgId×4` as a flat `uint32[]` offset table returns `0x3F3F3F3F` — four `3F` bytes, which is just the ASCII `?` character repeated. That region is not an offset table; it's literal content.

The header hex gives us facts:
```
0x00: 0D 00        → version = 13 (LE uint16)
0x02: E4 04        → count  = 1252 (LE uint16)
0x04: 01 00        → unknown field = 1
0x06–0x0D: 3F 00 × 4  → four LE uint16 = 63, not offsets
0x0E–0x19: all 00
0x1A: 03 01        → unknown
0x1C+: 00 00, 01 00, 02 00, 03 00 ...  → sequential uint16 (index list?)
```
First real string text: `bmdBase+0x0120`. Next strings at `+0x013E`, `+0x0158`.

### Reverse-anchor scan strategy

Instead of guessing the table format, we exploit a known invariant:

> **The offset table must contain the byte offsets of each message.** For message 0, that value is `0x0120`. For message 1, `0x013E`. For message 2, `0x0158`.

So we scan the entire first 1024 bytes of the BMD looking for these exact values as both `uint16` and `uint32`. Wherever we find them clustered, that IS the offset table, and the position of the match tells us:
- **Table base address** (e.g., `bmdBase + 0x??`)
- **Entry size** (distance between consecutive hits for msg0 and msg1)
- **Entry type** (uint16 or uint32)

#### Why this is guaranteed to terminate

The game engine has to look up each message by index to render dialogue. That means a data structure mapping `index → byte_offset` exists somewhere in the BMD or an adjacent table. The string at `+0x0120` is message zero; its offset (`0x0120 = 288`) is a concrete 2-byte or 4-byte value. The scan will find it.

#### Code

```csharp
byte* scan = (byte*)_bmdBase;
uint[] knownOffsets = { 0x0120u, 0x013Eu, 0x0158u };
for (int pos = 4; pos < 1024; pos++)
{
    if (!Memory.MemoryGuard.IsReadable(_bmdBase + (nuint)pos, 4)) break;
    byte* sp = (byte*)(_bmdBase + (nuint)pos);
    uint v32 = (uint)(sp[0] | sp[1]<<8 | sp[2]<<16 | sp[3]<<24);
    foreach (uint ko in knownOffsets)
        if (v32 == ko)
            _modLog!.Info($"[BMD] OffsetScan: 0x{ko:X} as uint32 at bmd+0x{pos:X}");
    if (pos < 1023) {
        ushort v16 = (ushort)(sp[0] | sp[1]<<8);
        foreach (uint ko in knownOffsets)
            if (v16 == ko)
                _modLog!.Info($"[BMD] OffsetScan: 0x{ko:X} as uint16 at bmd+0x{pos:X}");
    }
}
```

#### What to do with the output

Once we see lines like:
```
[BMD] OffsetScan: 0x120 as uint16 at bmd+0x1C
[BMD] OffsetScan: 0x13E as uint16 at bmd+0x1E
[BMD] OffsetScan: 0x158 as uint16 at bmd+0x20
```
That tells us:
- Table starts at `bmdBase + 0x1C`
- Entry size = 2 bytes (uint16), stride = 2
- `msgId N` → `offset = *(uint16*)(bmdBase + 0x1C + N*2)`

We then update `TryWriteToBmd` to read offsets using those discovered constants, and the SKIPPED log lines turn into successful writes.
6. The game's renderer re-reads from the same address next frame → shows new text

---

## Chapter 53: Deferred Pointer Capture — Why "Not Yet Populated" Isn't the Same as "Doesn't Exist"

### The Bug in One Sentence

Our hook fires when the BF interpreter *advances the PC*. The game's C++ dialogue system populates `session+0xD0` (the message descriptor pointer) slightly *later*, when the renderer actually constructs the speech bubble. These are two separate code paths, and ours runs first.

### Why This Happens

The BF virtual machine and the rendering layer are **decoupled**. The BF interpreter runs ahead — it steps through opcodes, encounters a "SHOW MESSAGE" instruction, calls into the C++ dialogue manager, and then *blocks waiting for player input*. But that blocking happens at the C++ level, not at the BF level. Our poll loop samples the BF PC while the C++ layer is still in the middle of its own setup.

Timeline:
```
t=0ms   BF interpreter hits SHOW_MSG opcode at pc=0x8B1
t=0ms   ProbeBfLine fires (PC changed), starts 3-streak countdown
t+~1s   3rd streak confirms: ProbeBfLine reads session+0xD0 → ZERO (not yet set)
t+1.5s  C++ dialogue manager allocates text buffer, writes ptr to session+0xD0
t+5s    LLM inference completes — TryWriteToBmd runs
```

On warm subsequent messages, the C++ setup happens faster (memory already allocated and reused), so the 3-streak window (≈1s at 500ms poll) overlaps with a populated +0xD0. On the very first message of a session (cold path), the allocation happens after our confirmation window closes.

### The Fix: Poll-Loop Lazy Retry

Instead of treating the captured-at-confirmation value as final, treat it as a *best-effort snapshot*. The poll loop runs every 500ms for the entire session. If we have a confirmed `_currentMsgId` but `_currentMsgTextAddr` is still 0, we re-probe `session+0xD0` every tick until it fills in.

```
Every 500ms tick:
  if _currentMsgId != 0 AND _currentMsgTextAddr == 0:
      attempt TryReadTextAddr(_capturedSession)
      if non-zero: store it, log "[MSG] TextAddr recovered"
```

By the time the LLM inference returns (3–30 seconds), the game has had 6–60 additional ticks to populate the field. The lazy write in `TryWriteToBmd` is a final safety net for the case where the recovery tick fires just before the LLM response arrives.

### Session Struct Dumps: The Debugging Tool

When `textAddr == 0` after confirmation, we dump all pointer-shaped values (≥ `HeapLow = 0x4000000000`) in the readable portion of the session struct. This shows which offsets ARE populated and point into the game's heap — letting us locate the descriptor via exploration rather than just trial-and-error at fixed offsets.

```
[MSG] SessPointers: +0x18→0x61B7F5F8  (skip: below HeapLow)
[MSG] SessPointers: +0x40→0x420A8B3C10  ← candidate
[MSG] SessPointers: +0x58→0x41DCE9A000  ← candidate
```

Any pointer ≥ 0x4000000000 is heap-allocated dialogue data. We can dereference each one and look for text or for the descriptor layout (zeros + two heap pointers at +0x18/+0x20).

### Why First-Message vs. Later-Message Timing Differs

| Condition | C++ dialogue setup time | Our 3-streak window | Result |
|---|---|---|---|
| First message (cold path) | ~1.5s (allocates heap buffer) | 0–1s after PC change | Miss |
| Later messages (warm path) | ~50ms (reuses existing buffer) | 0–1s after PC change | Hit |

The poll-loop retry collapses this distinction: both paths converge to "populated within 2–3 ticks," which is well inside the LLM's inference window.

---

## Chapter 54: C++ Polymorphic Session Types and Adaptive Pointer Scanning

### The Problem: Fixed Offsets Break Across Subclasses

P5R's dialogue session is a C++ object. Like most game engine objects, it uses inheritance — the base class stores common fields (BF script base, PC, confidant data), and subclasses add dialogue-specific fields at higher offsets. The problem: **two runs of the same scene can land on different subclasses**.

Why? Because the game reuses a pool of session objects. The previous run left a "full" session (which had +0xD0 populated) at address 0x424D8E1190. This run got a "lite" session at 0x4250321640 that ends or has a null pointer at +0xD0. Same scene, different C++ type from the pool.

The `SessPointers` output shows this clearly — the current run has no pointer-shaped value at +0xD0, but does have a stable external heap pointer at +0x90.

### The Fix: Adaptive Scan Instead of Fixed Offset

Instead of hardcoding `session+0xD0`, we scan ALL pointer-sized slots in the readable portion of the session struct and test each external heap pointer as a potential descriptor:

```
For each offset in [0x00, 0xC8) step 8:
    ptr = session[offset]
    if ptr is external heap (≥ 0x4000000000) and not self-referential:
        candidate = FollowTextObjChain(ptr)
        if candidate != 0: return it
```

`FollowTextObjChain` applies the known 3-hop pattern:
```
descriptor → descriptor+0x18 → textObj → *(textObj) → charPtr
```

Only returns non-zero if every link is non-null, readable, and the final charPtr is in heap range. This is safe — a false positive (accidentally valid chain) is checked at write time by `IsWritable`.

### Why the Scan Is Safe

The scan doesn't write anything — it only reads. If a random pointer accidentally passes all three chain checks, the worst case is a spurious `[MSG] TextAddr recovered` log. The write gate (`IsWritable`) prevents actual corruption. And in practice, pointer chains with 4 valid heap dereferences are very rare by accident.

### Boundary-Scan Logging

We also add a targeted dump of session[0xB8..0xE0] — one 8-byte slot at a time, checking each independently — to show exactly which offsets are readable and what values they hold. This lets us see the exact VirtualQuery boundary and understand WHY +0xD0 fails even when +0xC0 passes.

```
[MSG] sess+0xB8: R 0x42503215A0
[MSG] sess+0xC0: R 0x0000000000000000
[MSG] sess+0xC8: F                     ← boundary here
[MSG] sess+0xD0: F
```

If +0xC8 is the first unreadable slot, the struct is exactly 0xC8 bytes. If +0xD0 is unreadable but +0xC8 is readable-but-null, the struct exists but hasn't been populated yet at that offset.

---

## Chapter 55: Memory-Mapped Files vs. Heap — Two Totally Different Pointer Ranges

### What We Missed

We assumed dialogue text lives on the game heap (> 256 GB / `0x4000000000`). That was true for one type of session object. But `session+0xD0 = 0x610A3768` — that's in the **memory-mapped file region** (all `0x61...` addresses), the same range as `bfBase`. This is where P5R maps BMD dialogue files directly off disk into process memory via `MapViewOfFile`.

### Memory-Mapped Files in Windows

`MapViewOfFile` maps a file into the process address space. The OS picks an address in user space — typically in the 1–4 GB range for 32-bit compatible mappings, or wherever VirtualAlloc finds space. P5R uses `0x61...` for most of its game data files.

Protection is `PAGE_READONLY` (0x02) by default — the CPU MMU enforces this; any write triggers an access violation. To write to a mapped region without modifying the file on disk, call `VirtualProtect` with `PAGE_WRITECOPY` (0x08). The OS switches that page to copy-on-write: your process gets a private writable copy, the on-disk file is untouched.

### The False Positive Bug

`0x65007300610062` decoded as UTF-16 LE is "base" — a fragment of a shader filename. It sneaked through because our `charPtr >= HeapLow` check only has a **lower** bound (256 GB) but no **upper** bound. Valid Windows user-mode addresses cap at `0x7FFFFFFFFFFF` (48-bit VA space, 128 TB). Any value above that is data being misread as a pointer. Adding `charPtr <= 0x7FFFFFFFFFFF` rejects all such garbage.

### The Fix: Three-Path Text Discovery

```
Path A (primary heap path):
  session+0xD0 → descriptor (heap) → descriptor+0x18 → textObj → *(textObj) → charPtr

Path B (direct mapped-file path):  ← NEW
  session+0xD0 → value < HeapLow → bytes there look like English? → return it directly

Path C (fallback heap scan):
  scan session[0x00..0xC8) for external heap ptrs (with 48-bit cap) → follow chain
```

Path B handles the mapped-file case. If `*(session+0xD0)` is in low memory (`0x1000 < val < HeapLow`) and the bytes at that address contain ≥8 printable ASCII chars with ≥1 space, we treat it as the dialogue text directly.

### Writing to a Mapped Page

If textAddr < HeapLow (mapped file), `IsWritable` returns false (it's PAGE_READONLY). Before writing, call:
```csharp
VirtualProtect(textAddr, MaxWrite, PAGE_WRITECOPY, out oldProtect)
```
This makes the page copy-on-write. The write succeeds. The game's renderer, reading from this address, now sees our patched text instead of the original. No file on disk is modified.

---

## Chapter 56: Abandoning Pointer Chains — Content-Based Memory Discovery

### Why Pointer Chains Failed

The session struct has at least three C++ subclass variants in the wild. Each has the dialogue text pointer at a different offset, or not at all. Every fix for one variant breaks another. The session address itself can be in totally different memory regions (2GB vs 256TB) depending on game state. This is the wrong anchor.

### The Right Anchor: bfBase

`bfBase` is found reliably every single run via the large-copy hook. It's always in the `0x61...–0x62...` range (memory-mapped game data files). The dialogue BMD file for this scene is in the same mapped-file region — P5R loads paired `.bf` and `.bmd` assets from the same PAK archive, so they live close together in the virtual address space.

### Strategy 1: bfBase-Vicinity Scan for Dialogue Strings

Scan a ±512KB window around `bfBase` page-by-page. For each committed readable page, skip the first 0x100 bytes (possible binary header) and count null-terminated ASCII strings that have ≥2 spaces and ≥10 printable chars. A BMD file's string section will have many such strings (one per dialogue line). The first page that accumulates ≥5 such strings is the dialogue text pool.

Once found, we cache it as `_bmdTextPool`. We write the LLM text there using `VirtualProtect(PAGE_WRITECOPY)`, overwriting the original dialogue bytes in the game's private copy of the mapped page.

### Strategy 2: Cross-Region MemcpyText

The current MemcpyText hook drops any copy where `src < HeapLow`. But dialogue text flows FROM the mapped BMD file (src ≈ 0x61...) TO the render pipeline buffer (dst anywhere). Dropping the HeapLow restriction and filtering by spaces instead of vowels catches exactly this transfer — giving us the render buffer destination address directly.

### Why Both Strategies Together

- Strategy 1 finds the source (the BMD string pool, writable via WRITECOPY)
- Strategy 2 finds the destination (the render buffer the GPU reads from)

Whichever fires first tells us the correct write target. Both bypass the session struct entirely.

---

## Chapter 57: Synchronous Inline Interception — The Right Architecture

### Why Async Write Was Always Wrong

Every previous attempt wrote the LLM text AFTER the LLM responded (3–30 seconds later). By then the game had already rendered the original text and the player was reading it. Even if we found the exact right memory address, the write was too late.

### The Right Model: Hook the Copy, Own the Buffer

The game's text pipeline is:
```
BMD file (mapped memory) → memcpy → render buffer → GPU → screen
```

Our memcpy hook fires DURING that copy, on the game thread, before the frame is drawn. If we overwrite the render buffer (`dst`) immediately after the copy completes, the GPU sees our text instead of the original.

```csharp
// Game copies: BMD → dst (render buffer)
_memcpyHook!.OriginalFunction(dst, src, count);  // original text is in dst now
// We immediately overwrite:
MemoryCopy(cachedLlmText, dst, count, writeLen);  // our text is in dst now
// Frame renders → player sees our text
```

### Cache Strategy

LLM inference takes 3–30 seconds. We can't do it synchronously on the game thread. Instead:
- When a msgId is confirmed, fire the LLM call async as before
- When it responds, store the result in `_lastLlmText`
- The inline memcpy interceptor uses `_lastLlmText` to overwrite any subsequent dialogue copy

Result:
- Message 1: original text (LLM warming up)  
- Message 2+: LLM text (cache hit, instant inline write)

This is the correct beta architecture. The LLM "looks ahead" one message, which is imperceptible in normal play.

---

## Chapter 58: Writing the Pool, Not the Pointer

### What the memcpy log ruled out

Chapter 57's inline interception assumed dialogue text flows through the hooked
`memcpy`. The log disproved it. Every capture with `sp>=2` looked like this:

```
[MemcpyText] src=0x134C6E8F0 dst=0x418F0BFE00 n=192 sp=3: "·S·D(5·D········$··D···D·······"
```

Those `·D` pairs are the high bytes of `0x...44......` heap pointers — this is an
array of pointer structs, not a sentence. The space characters that passed the
`sp>=2` filter were coincidental `0x20` bytes inside binary fields.

Conclusion: P5R's text renderer does not bulk-copy dialogue. It walks the BMD
glyph-by-glyph, dereferencing directly. There is no copy to intercept.

### The remaining lever: mutate the source

If the renderer reads the BMD in place, then the BMD *is* the render buffer. We
don't need to find the render target — we need to edit the source before it's read.

### Two properties that make this safe

**1. Write within the original length.** Each entry is null-terminated and the
BMD's offset table stores absolute byte offsets to each entry. If we write a
shorter string and re-terminate in place, every offset in that table still points
where it did. Overrun the original length and we'd clobber the next entry's first
bytes, desynchronizing the table.

```csharp
int wl = Math.Min(enc.Length, len - 1);   // len - 1 leaves room for the terminator
System.Buffer.MemoryCopy(src, p + off, len, wl);
p[off + wl] = 0;
```

**2. Capture slot lengths once, before the first write.** This is the subtle one.
If we re-measured entry lengths on each write pass, the second pass would measure
the *shortened* string we just wrote:

```
original:  "So what do you want to do today?"   len = 32
write #1:  "Hey there."                          len becomes 10
write #2:  measures 10, can only write 9 bytes
write #3:  measures 9, can only write 8 bytes ...
```

The usable space ratchets down to nothing. `CapturePoolSlots` snapshots
`(offset, length)` at discovery time and every later write measures against those
originals.

### PAGE_WRITECOPY and why the .bmd on disk is safe

BMD files are memory-mapped `PAGE_READONLY`. `VirtualProtect(..., PAGE_WRITECOPY)`
flips the page to copy-on-write: the first write triggers the OS to allocate a
private physical page, copy the contents, and point our process's page table entry
at the copy. Every subsequent read in this process sees our text; the file on disk
and every other mapping of it are untouched. This is the same mechanism that backs
`fork()` semantics and DLL relocation.

### Known limitation

Every dialogue-looking slot in the pool gets the same text, so a scene will repeat
one line. That is intentional for now — it proves the write reaches the renderer.
Targeting the single entry for the current `msgId` requires parsing the BMD offset
table, which is the next step once we have visual confirmation.

---

## Chapter 59: Stop Guessing From Content — Parse the Format

### What the ranked scan actually proved

Chapter 58's prose detector worked exactly as designed. It found real English:

```
cand#0 score=60: "The highest quality shoes! | Are you prepared if disaster strikes!?"
cand#1 score=43: "Changed Shido's heart | Started Maruki's Palace"
cand#2 score=32: "Which Persona will it be? | Select the base Persona."
```

Shoe shop ads, the journal, the Velvet Room fusion menu. All genuine game text,
none of it from the scene on screen. The detector answered "is this English?"
correctly and that turned out to be the wrong question — the address space holds
dozens of loaded message files and prose-likeness cannot distinguish them.

A heuristic that is working correctly and still gives you the wrong answer is a
sign the question is wrong, not the thresholds.

### The format has an answer built in

Atlus message files carry a `MSG1` magic and a 32-byte header:

```
+0x00  fileType(1)  format(1)  userId(2)
+0x04  fileSize(4)
+0x08  magic "MSG1"          <- the searchable anchor
+0x0C  extSize(4)
+0x10  relocationTable(4)   +0x14 relocationTableSize(4)
+0x18  messageCount(4)
+0x1C  isRelocated(2)  version(2)
```

The magic sits at +0x08, so a hit implies the header began 8 bytes earlier. That
one subtraction converts a byte-pattern match into a structured record: declared
file size, message count, and a validity check (`0x40 <= fileSize <= 4MB`,
`0 < messageCount <= 20000`) that binary noise essentially cannot pass by accident.

### Why bfBase is the right anchor

In P5R the message script is normally *embedded inside* the `.bf` flowscript rather
than shipped as a separate file. The script and its dialogue are one archive entry.
So the first valid `MSG1` header at or after `bfBase` is this scene's dialogue —
not the shoe shop's. `bfBase` is the one address this project resolves reliably on
every single run, which makes it the right thing to anchor to.

### Walking regions instead of probing pages

Scanning 5 MB by probing each 4 KB page costs ~1280 `VirtualQuery` syscalls. Walking
regions costs a handful:

```csharp
var (ok, regionBase, regionSize, state, protect) = MemoryGuard.QueryRegion(addr);
// ... scan this region if committed ...
addr = regionBase + regionSize;   // jump the whole region, mapped or not
```

`VirtualQuery` reports the extent of the *entire* run of pages sharing a state, so
one call skips an unmapped span of any size. Region walking is the general pattern
for "search a wide address range" — page probing is for "is this one pointer safe."

### ReadableLen: never trust a declared size

The header's `fileSize` describes the file on disk. Its tail pages may not be
resident in memory. Reading the declared length would fault, so the length is
resolved down in page steps until it is actually readable, and the scan uses that.
Declared sizes are a claim about the format; `VirtualQuery` is the truth about
this process.

---

## Chapter 60: The Encoding Was the Blind Spot

### What 21 valid message files actually told us

The MSG1 scan worked perfectly and every file scored well:

```
msgs=696  score=1188: "ARestores 20 HP to one ally. | AA new drug develop"
msgs=1050 score=856:  "ALight Fire dmg to | one foe. Chance of"
msgs=464  score=2081: "AThe greatest angel | legend. He is known"
```

Weapons, clothes, healing items, skills, Persona lore. Every one of the 21 files
is a *global data table*. None is scene dialogue. The entry dump made it explicit:

```
[MSG1] entry[709] kind=0 offset=0xCBE4
[MSG1] fileBase+off: "ABLANK·····skill_268···"
```

Message 709 is named `skill_268`. The `msgId` extracted from the BF instruction
window indexes skills in that file, not conversation lines. Selecting the file by
`msgCount > msgId` was therefore selecting on a coincidence — any table with
enough rows would have matched.

### The line that reframed everything

The session's message object dumped this:

```
52 00 4F 00 52 00 00 00     ->  "R\0O\0R\0"
```

That is UTF-16LE. Every scanner written for this project so far — `LooksLikeText`,
`ProbeForBfContent`, `IsEnglishSentence`, `CountEnglishSentences` — reads one byte
per character. Against UTF-16 they do not merely score badly, they are
*structurally incapable* of producing a hit:

```
UTF-16 "Hello":  48 00 65 00 6C 00 6C 00 6F 00
ASCII run walk:  ^^ run ends here (0x00 is not printable)
```

Every run is exactly one character long, so `len < 12` rejects all of them. A
UTF-16 dialogue buffer would look like empty space to all of this code. The text
may have been sitting in a scanned region the whole time.

The general lesson: when a detector finds nothing anywhere, suspect the
representation before tuning the thresholds. Thresholds produce weak signals;
a wrong representation produces exactly zero, which is what we kept seeing.

### Reusing the test across encodings

Rather than write the ratio test twice, decode first and share one predicate:

```csharp
IsEnglishString(string s)   // the ratio test, encoding-agnostic
IsEnglishSentence(byte*, n) // ASCII: bytes are already characters
FindUtf16English(...)       // decode even bytes, then IsEnglishString
```

The UTF-16 scan keys on the pattern "printable in even byte, 0x00 in odd byte",
accumulates the run, then hands the decoded string to the same predicate.

### session+0xC8 is the real handle

Also new this run: `session+0xC8 = 0x41D6D21780`, a live heap object with a vtable
at +0x00, and `session+0xD0 = 0x175E` — far too small for a pointer, so an offset
or index. That object is the game's own handle on the displayed message, populated
exactly while a message is on screen. Following it beats guessing at pool
addresses, which is what `ProbeMessageObject` now does — dumping every heap pointer
it holds and testing each target as both ASCII and UTF-16.

---

## Chapter 61: TOCTOU, and Why the Scan Killed the Game

### The crash

```
Fatal error. System.AccessViolationException: Attempted to read or write protected memory.
   at P5RGenSocialLinks.Mod.FindAsciiEnglish(UIntPtr, Int32, Int32)
   at P5RGenSocialLinks.Mod.TryFindHeapDialoguePool()
```

The scan validated a region and then read it:

```csharp
if (MemoryGuard.IsReadable(regionBase, len))     // check
    foreach (var hit in FindAsciiEnglish(regionBase, len, 8192))  // ...then use
```

That is a textbook time-of-check/time-of-use race. `IsReadable` is a single
`VirtualQuery` describing the address space *at that instant*. The walk that follows
covers up to 8 MB and takes real time, on a background poll thread, while the game's
own threads continue allocating and freeing heap. Somewhere mid-scan the region was
unmapped and the next dereference hit unmapped memory.

The window scaled with exactly the thing that had been increased to improve coverage:
larger regions meant longer scans meant a wider window to lose the memory.

### Why this one is fatal rather than catchable

`AccessViolationException` is a *corrupted-state exception*. Since .NET Core, it
cannot be caught at all — not by `catch (Exception)`, not by
`catch (AccessViolationException)`. The legacy
`<LegacyCorruptedStateExceptionsPolicy>` escape hatch does not exist on modern
runtimes. The process dies, taking the game with it.

So this cannot be handled defensively after the fact. It has to be made impossible.

### The fix: copy instead of dereference

```csharp
internal static bool TryRead(nuint addr, byte[] buffer, int len)
    => ReadProcessMemory(GetCurrentProcess(), addr, buffer, (nuint)len, out var got)
       && (int)got == len;
```

`ReadProcessMemory` against the current process performs the same validation the
CPU would, but reports failure through a return value instead of a fault. If the
region vanished, the call returns `false` and the scan skips that chunk.

Reading 256 KB at a time is a deliberate balance: small enough that a chunk copy is
effectively atomic against the game's allocator, large enough that scanning 2 GB
costs roughly 8000 syscalls rather than millions.

### The general rule

Tightening the check does not fix a TOCTOU race — it only narrows the window.
Checking every 4 KB instead of every 8 MB would have made the crash rarer and
therefore harder to diagnose, not absent. The race is only eliminated by making the
check and the use the *same operation*, which is precisely what `ReadProcessMemory`
does: validation and copy happen together inside one kernel call.

A corollary worth carrying: raw pointer walks over another subsystem's live memory
are safe only for small, immediate reads. Anything that scans in bulk should copy
first and parse the copy.

## Chapter 62: Process Boundaries Beat Language Bindings

### The wall

The server had been answering `/health` with `"mock"` for weeks. The assumption was a
stray config flag. The actual cause:

```
>>> import llama_cpp
ModuleNotFoundError: No module named 'llama_cpp'
```

It was never installed. And it could not be, because of a detail that has nothing to
do with this project:

| | |
|---|---|
| The venv | Python 3.13 |
| Official CUDA wheels | cp39–cp312. No cp313. |
| PyPI | sdist only — no binaries at all |

`llama-cpp-python` is not Python. It is a C++ library with a thin Python jacket, and
that jacket is stitched to a specific CPython ABI. A `cp312` wheel contains machine
code linked against `python312.dll`; loading it into 3.13 is not a version warning,
it is a missing symbol. So "just install it" decomposes into: install the CUDA
Toolkit, install MSVC build tools, and compile llama.cpp from source — a multi-hour
setup that then has to be repeated by every person who installs the mod.

### The thing worth noticing

The dependency was never really on *a Python package*. It was on **llama.cpp**, a C++
program. The Python package was a delivery mechanism, and it was the delivery
mechanism imposing every constraint: the ABI pin, the missing wheel, the compiler
requirement. None of those are properties of the inference engine.

Upstream already publishes exactly what we want, prebuilt:

```
llama-b10408-bin-win-cuda-12.4-x64.zip     # llama-server.exe + friends
cudart-llama-bin-win-cuda-12.4-x64.zip     # bundled CUDA runtime DLLs
```

An `.exe` has no opinion about the Python interpreter that talks to it. Swapping the
binding for a process boundary deletes the entire class of problem — not by solving
the ABI mismatch, but by arranging for there not to be an ABI.

### Why the CUDA runtime ships in the zip

`cudart-*.zip` is 391 MB of CUDA runtime DLLs, and shipping it is what removes the
"install the CUDA Toolkit" step. The split matters because two different things are
often both called "CUDA":

- **The driver** (`nvidia-smi` reports 13.1 here) — kernel-mode, installed with the GPU driver.
- **The runtime** (`cudart64_*.dll`) — user-mode, versioned with the toolkit, and
  legal to ship next to the application.

NVIDIA guarantees *backward* compatibility: a newer driver runs applications built
against older toolkits. That is why the build tagged 12.4 is the safe pick on a 13.1
driver, and the 13.3 build is not obviously safer despite the closer number — it
would rely on minor-version compatibility within CUDA 13, which holds only as long as
the binary avoids driver APIs newer than the installed driver. Backward compatibility
is a guarantee; forward compatibility is a hope.

### What the boundary buys beyond the install

The original design loaded the model inside the FastAPI process:

```python
model     = load_model(_cfg)          # ~15s, blocking, 4.9 GB into VRAM
_pipeline = InferencePipeline(model, _cfg)
```

Three consequences, all of which the split fixes:

1. **Startup coupling.** The dialogue server was unavailable until the weights
   finished loading. Separated, the HTTP server answers `/health` immediately and
   reports the model as not-yet-ready — a much more useful signal to the mod than a
   connection refusal.
2. **Failure coupling.** A CUDA OOM inside the binding takes down the process that
   the game is talking to. As a child process, the same OOM leaves our server up and
   able to say *why* it degraded.
3. **Lifecycle coupling.** Reloading the model, or swapping to a different GGUF,
   meant restarting the API. Now it does not.

There is a cost, and it should be stated plainly: an extra process to supervise, one
more thing to fail, and an HTTP hop (sub-millisecond on loopback, against a ~2s
generation — irrelevant here, but not in every system).

### The interface was already the right shape

The reason this rewrite is small is visible in the existing call:

```python
response = self._model.create_chat_completion(
    messages=[{"role": "system", ...}, {"role": "user", ...}],
    max_tokens=..., temperature=..., top_p=...,
)
raw = response["choices"][0]["message"]["content"]
```

That is the OpenAI chat-completions schema, which is also precisely what
`llama-server` exposes at `/v1/chat/completions`. `llama-cpp-python` was mirroring
that schema in-process; the same request shape crosses a socket instead. The port is
therefore a change of *transport*, not of contract — `build_prompt` and
`clean_response` on either side of the call never learn that anything moved.

### The general rule

When a dependency is painful, check whether the pain belongs to the thing you need or
to the wrapper you are getting it through. Native bindings couple you to the host
language's ABI, its version, and its build toolchain, and they charge that cost to
every user at install time. A subprocess speaking a documented protocol couples you to
a wire format — which is versioned far more slowly and can be reimplemented by hand if
it ever breaks.

Prefer a process boundary to a language binding when the dependency is a program that
already knows how to be a server.

## Chapter 63: Nobody Runs Your Cleanup Code

### The orphan

With inference moved into a child process, shutdown looked handled:

```python
yield                    # FastAPI lifespan: everything after this is teardown
if _process is not None:
    _process.stop()      # terminate llama-server
```

Then the parent was force-killed during testing, and:

```
$ netstat -ano | grep 8766
  TCP  127.0.0.1:8766  LISTENING  133716        # still there
```

llama-server survived, holding the port and ~4.9 GB of VRAM. The next start then
failed its port check — a second, more confusing symptom caused by the first.

### Why the handler did not run

`stop()` is ordinary Python. It runs when the interpreter reaches it, which requires
the process to still be alive and cooperating. None of these give it that chance:

- `taskkill /F` — `TerminateProcess`, no notification to the target at all
- Closing the console window — the parent can be killed before its handler finishes
- Task Manager "End task"
- A hard crash in the parent

This is the same shape as the `AccessViolationException` in Chapter 61, and it has the
same moral. There, the fix was not a better `catch` — an uncatchable exception cannot
be caught — but making the fault impossible by moving validation into the kernel with
`ReadProcessMemory`. Here the fix is not a better handler. A handler that never runs
cannot be improved. The cleanup has to happen *somewhere that is not our process*.

### Job Objects

Windows has exactly that mechanism:

```python
job = kernel32.CreateJobObjectW(None, None)
info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ...)
kernel32.AssignProcessToJobObject(job, child_handle)
```

A Job Object is a kernel container for processes. `KILL_ON_JOB_CLOSE` says: when the
last handle to this job closes, terminate everything inside it.

The load-bearing detail is that handles are closed by the **kernel**, as part of
tearing down a process, whatever killed it. There is no callback, no unwind, no code
of ours involved. Our process dying *is* the trigger. That is why this survives
`TerminateProcess` when a `finally` block does not.

Two practical notes:

- **Hold the handle.** The job dies with the last handle, so it lives on the supervisor
  object for the process lifetime. Closing it early kills the child immediately —
  which is also exactly how `stop()` uses it as a backstop.
- **Adopt before waiting for readiness.** Model load is a ~20 s window and the longest
  stretch in which a force-kill could strand a child. Adopting after the health check
  would leave precisely the riskiest period unprotected.

### Testing something that requires killing yourself

The real behaviour — child dies when force-killed parent dies — cannot be asserted
in-process, because demonstrating it means killing the test runner. It was verified
with a spawned parent/child pair instead:

```
parent=162732 child=161360 adopted True
child alive before kill: True
child alive AFTER force-kill of parent: False
```

The suite then covers what it usefully can: that closing the job kills its members
(the same limit, triggered directly), that adoption failure returns `False` rather
than raising, and that the whole thing no-ops off Windows so CI stays green. When a
property cannot be tested automatically, test its mechanism and record the manual
verification — do not pretend the untestable part is covered.

### The general rule

Cleanup that depends on your own code running is not cleanup, it is an optimistic
default. It handles the polite exits, which are the ones that were never going to
cause trouble. For any resource whose leak actually hurts — a port, GPU memory, a
lock, a temp directory — ask what happens when the process is shot in the head, and
push the release down to something that outlives you: the kernel, the OS, the
supervisor, the database's session timeout.

"The exit handler didn't run" is not an edge case. It is the normal outcome of every
abnormal exit, and abnormal exits are the ones worth designing for.

## Chapter 64: The Data You Destroy Is the Data You Need

### The symptom

Generated dialogue was in character and in the right register, and still felt wrong:

```
Ryuji (scripted):  "A towel?"  →  "Nah, they'll lend you one of them."
Ryuji (generated): "Time to crush it, let's go all-out on these reps!"
```

Nothing about that line is incorrect. It is a gym line, in his voice, at the right
rank. It is also completely deaf to the conversation it replaced.

The context being sent explains why:

```
"Hang-out with Ryuji Sakamoto (rank 4/10) at Protein Lovers gym.
 This is a Social Link conversation — ..."
```

Who, where, how close. Nothing about *what is being said*. The model was writing
ambience because ambience is all it had the information to write.

### Where the missing context was

In the buffer we were overwriting.

```csharp
fixed (byte* src = enc)
    System.Buffer.MemoryCopy(src, p + off, len, wl);
```

Before that copy, `p + off` holds the scripted line. The write is the only operation
in the system that touches the script, and it discards it without reading it. The
information needed to respond to the scene was being destroyed by the act of
responding to the scene.

This is worth sitting with, because it is a shape that recurs: a destructive operation
standing exactly where the missing input lives. Migrations that drop the column whose
old values would validate the migration. Caches invalidated before the recomputation
that could have reused them. The fix is never to reconstruct the data later — it is to
notice that the last moment it exists is *right here*, and read it first.

### Ordering is the whole design

The capture has to happen at arm time, before the first write:

```csharp
if (_sceneScript.Count == 0)
{
    var script = CaptureSceneScript(ranked[0].Base, ranked[0].Len, maxLines: 12, maxChars: 480);
    ...
}
foreach (int i in selected) { /* arm regions for writing */ }
```

Reading lazily — when the context is first needed — would look equivalent and be
silently wrong. By then the first line has been written, so every slot holds our own
generated text. The model would receive its own previous output labelled as the
scene's script, and each generation would drift further from a scene it could no
longer see. A feedback loop that produces plausible output is far harder to notice
than one that crashes.

The same reasoning drives an explicit guard, because arming can recur mid-session:

```csharp
if (_lastLlmText != null && line.Contains(_lastLlmText)) continue;
```

### Handing over dialogue invites imitation

The first framing was simply the lines. The obvious failure followed: the model
paraphrased them, which reads worse than generic filler, because a near-copy of the
original looks like a bug rather than a variation.

```
"The scene's original dialogue, for subject matter only — do not repeat or
 paraphrase these lines: ..."
```

Context is not neutral. Anything placed in a prompt is an implicit instruction, and
examples are the strongest implicit instruction there is — a model shown text will
tend to produce text like it. When the intent is *topic* rather than *style*, that has
to be said, or the examples win.

### Result

```
before   "Time to crush it, let's go all-out on these reps!"
after    "What's with the look, you think I'm scrawny?"
         "Dude, stop tryin', you'll just end up hurtin' yourself."
```

The second set engages with the actual banter — the ribbing, Ryuji's defensiveness
about his build — without reusing a line from it.

### The general rule

When output is generically correct but contextually hollow, the question is not which
model or which sampling parameters. It is what the model was given. Trace the inputs
before touching the generation, and check specifically whether the thing you need is
being thrown away somewhere upstream — most often by the very step that needed it.

---

## Chapter 65: Ask the Consumer, Not the Heap

### What is still wrong

The pipeline works end to end. A generated line reaches the speech bubble. But three
symptoms survived, and they are all the same bug wearing different clothes:

1. **Both rows of the bubble show the same line.** We write all 36 slots in the region,
   because we do not know which one is live.
2. **The ranking picks a different buffer between sessions.** One run rendered from
   `0x421167C000` (266 KB); the next from `0x41EF589000` (1 MB). Same scene, same
   build.
3. **The twin arming fired in one run and not the other.** Content matching found a
   second copy once, and once did not need to.

Every one of these traces to the same root: *we are inferring which bytes the renderer
reads by scoring how much they look like dialogue.* Similarity is a heuristic. It
ranked `ternTableOffset: -1` above the live scene once, and that crashed the game.

### The inversion

There is a source of truth we have not asked. The renderer *knows* which address it
reads — it reads it, every frame, with a specific instruction at a specific offset in
a specific module. We have been guessing the answer to a question the CPU can be made
to answer exactly.

That is the whole idea behind "Find out what accesses this address":

```
guessing:  scan heap → score regions → rank → hope #0 is the live one
asking:    pick the address that is on screen → break on reads → read the instruction
```

The second one cannot be wrong about which buffer matters, because the evidence *is*
the read.

### How the CPU does this

x86-64 has four debug address registers, `DR0`–`DR3`, plus a control register `DR7`.
Each `DRn` holds a linear address; `DR7` says, per slot, what access triggers a fault
and over what width (1, 2, 4, or 8 bytes):

```
DR0 = 0x421167C0A0      ← the address of the line in the bubble
DR7 = ...R/W bits: 11   ← 11 = break on read or write, 01 = write only
      ...LEN bits: 11   ← 8 bytes wide
```

When the CPU executes an instruction that touches that range, it raises `#DB`, and the
debugger — Cheat Engine, here — records `RIP`, the register file, and moves on.

Two consequences fall directly out of that mechanism:

- **There are exactly four.** Hardware breakpoints are a fixed CPU resource, not a
  software list. Watch a fifth address and something must be evicted.
- **It costs nothing until it fires.** Unlike a software breakpoint (which rewrites
  the target instruction to `0xCC`), a hardware watchpoint modifies no memory. The
  game does not know it is being watched, and unmodified code paths run at full speed.

### Why "accesses", not "writes"

Cheat Engine offers both, and the wrong one produces a confident, useless answer.

- **"Find out what *writes* to this address"** catches the *producer* — whatever
  filled the buffer from the MSG archive. That code runs once, before the scene, and
  tells us nothing about which of two copies the renderer prefers.
- **"Find out what *accesses* this address"** catches reads too, and the renderer is a
  reader. It is the only one of the two that answers our actual question.

This distinction is the payoff of Chapter 60's finding that the renderer reads the
buffer **in place** — there is no `memcpy` into a render-side scratch buffer to
intercept. In-place reading is why the write works at all, and it is also why a read
watchpoint is the correct instrument.

### Reading the results

The access list will not be one entry. Expect noise, and expect it to be
distinguishable by *frequency*:

| Pattern | What it usually is | Useful? |
|---|---|---|
| Thousands of hits/sec, `movzx` byte loops | `strlen`, `memcpy`, CRT internals | No |
| A handful per frame, `[rcx+rdx*1+0x20]`-style | Layout / glyph iteration | **Yes** |
| One or two hits *only when the line advances* | The dispatch that binds a slot | **Best** |

The last row is the prize. An instruction that fires exactly once per line advance is
reading the slot *as a choice* — which means the register arithmetic in that
instruction contains the index we have been unable to compute.

`[rcx+rdx*1+0x20]` is not decoration; it is the answer written out. `rcx` is the pool
base, `0x20` is the header, and `rdx` is the slot index we have been brute-forcing
across all 36 slots. Hook that instruction and `rdx` hands us the live slot for free.

### Why the module + offset matters more than the address

Cheat Engine will report something like `P5R.exe+1A3F2C`. The absolute address is
worthless across runs — ASLR rebases the module every launch. The *offset* is stable
for a given build, and that is exactly what `Reloaded.Memory.SigScan` consumes: we
turn the bytes at that offset into a signature, scan for it at load, and hook the
result. The address is a fact about today; the offset is a fact about the binary.

### What this buys, concretely

All three symptoms collapse at once:

- Duplicated rows: gone, because we write one slot instead of 36.
- Ranking instability: gone, because we stop ranking. The instruction tells us the
  base.
- Blast radius: gone. A single-slot write cannot arm a data table by accident, so the
  crash class from Chapter 63 stops being reachable.

And the scene-script capture gets sharper too: instead of the whole pool deduped down
to 12 lines, we get the actual preceding line, in order.

### The general lesson

When you are scoring candidates to guess which one a system uses, check first whether
the system will simply tell you. Runtime has information static inspection does not,
and a watchpoint, a `LD_PRELOAD` shim, an `strace`, or an access log are all the same
move: stop modelling the consumer and go observe it. Heuristics are what you build
when observation is impossible — not before you have tried.

---

## Chapter 66: The Buffer Was a Format, Not a Pile

### What a hex dump showed that a scanner could not

Every conclusion so far about the dialogue pool came from an ASCII scanner: walk the
bytes, collect printable runs of 8+ characters, score them for English. That tool
answers exactly one question — *where is text?* — and it silently discards everything
that is not text. Which turned out to be the part that mattered.

Opening the same region in a hex viewer:

```
424F054547  54 68 65 20 65 71 ... 62 75 74 0A 74 68 65 79   "The equipment's kinda crappy,but they got"
424F054577  ... 74 79 2E 0A F2 23 00 00 00 F1 21 F2 05 FF   "ty.  #.. ! ..."
424F0545D7  ... 4D 53 47 5F 30 30 31 5F 35 5F 30 00         "...MSG_001_5_0."
424F054637  ... 49 74 27 73 20 70 61 79 ...                 "It's pay per visit, so you don'"
424F054757  ... 4D 53 47 5F 30 30 32 5F 30 5F 30 00         "...MSG_002_0_0."
424F0547E7  ... 53 45 4C 20 30 30 33 ...                    ".SEL 003"
```

This is not a heap of loose strings. It is Atlus's BMD message archive, decompressed,
**with its symbol table intact**.

### Three corrections, in increasing order of consequence

**The separator is `0x0A`, not `0x00`.** Consecutive "slots" sat one byte apart, and I
read that byte as a null terminator without checking it. It is a newline:

```
62 75 74 0A 74 68 65 79
b  u  t  \n t  h  e  y
```

So `"The equipment's kinda crappy, but"` and `"they got tons of variety."` were never
two strings. They are **one string with a line break** — the two rows of a single
speech bubble. The scanner split on non-printable bytes and manufactured a boundary
that does not exist in the data.

That single byte explains a bug we had filed as unsolved: *both rows of the bubble
show the same generated line.* Of course they do. We treat the halves as independent
slots and write the same text into each. The fix is not per-slot targeting — it is to
stop splitting, and write one string containing its own `\n`.

**Each line is stored more than once.** The same three lines appear in two blocks, with
different wording:

```
0x424F0544D0   "It's pay per visit though, so you don't ..."
0x424F054637   "It's pay per visit, so you don't ..."        ← on screen
```

The heuristic scanner reported both as excellent dialogue, because both *are* excellent
dialogue. Nothing in a similarity score can distinguish the take the game rendered from
the take it did not. This is why the first watchpoint recorded zero accesses: it was
armed on prose the renderer never reads.

**The archive knows its own names.** `MSG_001_5_0`, `MSG_002_0_0`, `SEL 003` are
labels, not text. The mod already logs `msgId=0x2C7 (711)` from the BF hook, and it has
been carrying that number around with no way to spend it. A label table is exactly the
thing an ID indexes into.

### The shape of the mistake

The scanner was written to answer *"is this English?"* and it answered well. The bug
was mistaking an answer for **the** answer. `0x0A` is not English, so it was invisible;
`F1 21 F2 05` is not English, so it was invisible; `MSG_002_0_0` failed the vowel test,
so it was invisible. Every structural fact about the format lived precisely in the
bytes the filter was designed to drop.

The general form: a filter tuned to find signal will, by construction, hide the frame
the signal sits in. When a heuristic keeps *almost* working — right region, wrong copy;
right text, wrong boundaries — that is the signature of a format being read as a pile.
Stop refining the score and go look at the raw bytes.

### Where this leads

Ranking regions by how much they look like dialogue was always a stand-in for parsing.
With labels and control codes visible, the real path opens:

```
msgId (already have)  →  label table  →  message block  →  text + its \n
```

That is deterministic. No scores, no anchor, no twin, no 36-slot blast radius — and
the blast radius is where the crash lived. The watchpoint hunt is still worth
finishing, because the renderer's own read is the fastest confirmation that a block is
live. But the destination changed: from *guess better* to *read the format*.

---

## Chapter 67: A Signature Is a Claim About the Whole Binary

### What a debugger actually told us

Cheat Engine reported an instruction at `P5R.exe+17A3D27` that reads the message being
rendered. Turning that into something the mod can use at runtime means writing down the
bytes around it:

```
48 63 53 30    movsxd rdx, dword ptr [rbx+30]
48 8B 43 20    mov    rax, [rbx+20]
0F B6 3C 02    movzx  edi, byte ptr [rdx+rax]
85 FF          test   edi, edi
0F 85          jne    <loop>
```

The reason this indirection exists at all is ASLR. `0x1417A3D27` was true for one
launch and will be false for the next: Windows rebases the image every time, so the
only durable fact is the *offset*, and the only way to recover an offset without a
symbol table is to search for the bytes.

### The sample-of-one problem

Here is the trap. `Scanner.FindPattern` returns `PatternScanResult`, which has `Found`
and `Offset` — and nothing else. It stops at the first hit. So a pattern occurring twice
is indistinguishable at runtime from a pattern occurring once: both resolve, both look
successful, and one of them hooks an instruction you have never seen.

Worse, the failure is quiet and late. The scan succeeds at load. The hook activates. The
log says `sig OK`. The wrongness only shows up as garbage data much later, in a code
path with no obvious link to a scan that happened at startup.

A signature is not "bytes that match my target". It is a claim that **no other point in
386 MB of image matches**. Nothing at runtime checks that claim, so it has to be checked
somewhere else.

### Checking it offline

`scripts/verify-signature.py` reads the PE from disk and counts every occurrence:

```
occurrences  1
  file 0x17A331F -> p5r.exe+17A3D1F  (section .sdata)
```

One hit. And run against the older `CmmExecEvent` pattern it independently reproduces
`p5r.exe+E0D0B0`, which matches the `0x140E0D0B0` recorded in that signature's comment
from a Ghidra session months ago — the tool validated against a known answer before
being trusted on an unknown one.

Two details in that output are worth reading carefully. `p5r.exe+17A3D1F` is the RVA, not
the file offset (`0x17A331F`) — sections are laid out differently on disk than in memory,
so the two differ by the section's own delta, and comparing the wrong one against a
disassembler produces a confident mismatch. And `.sdata` holding executable code is odd
enough to notice; what matters for us is that the bytes are present in the file rather
than produced at runtime by a packer, which the successful match proves.

### What to wildcard, and what not to

The usual advice is to wildcard anything that might shift between builds: RIP-relative
displacements, jump targets, compiler-chosen stack slots. That is right, but it is only
half a rule, because every wildcard also widens the match.

Here nothing is wildcarded except by omission — the pattern stops at the `0F 85` opcode
and excludes the jump displacement that follows. The struct offsets `0x20` and `0x30`
are deliberately literal, and that is the important decision:

```
48 8B 43 20    mov rax, [rbx+0x20]     <- 0x20 is the record pointer field
48 63 53 30    movsxd rdx, [rbx+0x30]  <- 0x30 is the cursor field
```

Those two bytes are not incidental encoding, they are *the thing we learned*. If a game
update moves the record pointer to `+0x28`, a wildcarded pattern still matches — and the
mod then reads a neighbouring field and dereferences whatever it finds. A literal
pattern fails the scan loudly instead, and the mod falls back to the heuristic it
already has.

So: wildcard what you do not care about, and pin what you are relying on. A signature
that cannot fail is not robust, it is just silent.

### Separating resolution from hooking

The scan and the hook land as two commits, not one, and the first logs an address
without touching it. This is not tidiness — it is that each attempt costs a game
restart, a load, and walking back to a hang-out. When "sig OK" and "hook active" appear
together and the game then misbehaves, you have two suspects and one expensive
experiment. Split, and the first run answers "did the pattern survive this build?"
before the second run asks anything harder.

---

## Chapter 68: Six Instructions in Somebody Else's Loop

### The budget

The hook goes here:

```
P5R.exe+17A3D1F   movsxd rdx, dword ptr [rbx+30]
P5R.exe+17A3D23   mov    rax, [rbx+20]
P5R.exe+17A3D27   movzx  edi, byte ptr [rdx+rax]   <- our stub runs just before this
P5R.exe+17A3D2B   test   edi, edi
P5R.exe+17A3D2D   jne    P5R.exe+17A367B
```

RAX holds the record pointer and RDX the cursor, so all we need is to copy two registers
somewhere we can read them from C#:

```asm
push rcx
mov  rcx, <slab>
mov  [rcx],   rax
mov  [rcx+8], rdx
pop  rcx
```

Five instructions. Everything interesting about this step is a constraint on what those
five are allowed to do.

### Constraint 1: registers are borrowed, not owned

We need a scratch register to hold the destination address, because x86-64 has no
`mov [imm64], r64` — only RAX gets that privilege, and RAX is exactly what we are trying
to save. So RCX gets pushed and popped.

The instinct to grab "a register that looks free" is how these hooks go wrong. There is
no free register. The function is mid-computation; every register belongs to the caller's
plan. Push and pop is not defensive style, it is the only correct move.

### Constraint 2: flags are a register too, and the loop reads them

Look at what runs immediately after our stub returns control:

```
test edi, edi
jne  <loop top>
```

`TEST` sets the flags and `JNE` consumes them. If the stub disturbed the flags register
between the two, we would be steering the interpreter's control flow — not corrupting
data, but rewriting the program's branches, which fails in a way no memory dump explains.

The chosen five instructions do not touch flags: `PUSH`, `POP`, and `MOV` all leave them
alone. That is not a happy accident, it is why the stub uses `mov` and not, say, `xor
rcx, rcx` to zero a register, or `add` to compute an offset. In injected code, the flags
register is part of the calling convention whether the ABI says so or not.

### Constraint 3: this is a per-character loop

The stub runs once for every byte of every message the game displays. That immediately
rules out the obvious design — log from inside the hook — because a managed callback,
a string format, and a file write would run tens of thousands of times a second inside a
render path.

So the split is: the stub writes two words to a fixed address and does nothing else, and
a 500 ms poll tick reads that address and reports changes. The stub has no idea anyone is
watching.

The honest cost of sampling is that short-lived records are missed. For verification that
is acceptable — what needs proving is that the pointer is real, in the heap, and changes
line to line. It would be the wrong mechanism for driving writes, which have to key off
the record changing, not off a timer.

### Constraint 4: the slab outlives everything

The destination address is an immediate operand baked into machine code:

```asm
mov rcx, 0x1F4A2C60      ; not a variable — a constant inside the instruction stream
```

Two things follow. It cannot be a managed array, even pinned, because the address has to
stay valid for as long as the code exists rather than as long as a handle does — so it is
`Marshal.AllocHGlobal`.

And it must never be freed. `Dispose` disables the hook, which restores the original
bytes, but restoring bytes does not evict threads already executing inside the stub.
Freeing the slab there would trade 16 bytes for a write through a dangling pointer at
process shutdown. The leak is the correct choice, and worth a comment saying so, because
it will otherwise look like a bug to the next reader.

### Where the risk actually is

Everything above is reasoned, not tested; the first honest test is the game launching.
That is why the hook is behind `msg_hook_enabled` with a default that can be turned off
without a rebuild, and why installation is wrapped so an assembler failure degrades to
the heap heuristic instead of aborting mod startup.

A broad `catch` around exactly one call, with a written reason, is the right shape here:
the assembler is third-party, its failure type is not part of the interface we reference,
and every path downstream still works without the watch. Broad catches are bad when they
hide failures you could have handled. This one converts an unknown failure into a named,
logged degradation.

---

## Chapter 69: The Hook Answers, and the Answers Rewrite the Model

### What one log run replaced

Eleven lines of a gym scene, each one caught at the moment the game read it:

```
[WATCH] in-pool   record=0x424CAC6BDC "Here we are... Protein Lovers gym!"
[WATCH] in-pool   record=0x424CAC6E90 "It's pay per visit, so you don't gotta worry"
[WATCH] in-pool   record=0x424CAC6F14 "The equipment's kinda crappy, but"
...
[WATCH] in-pool   record=0x424CAC74B8 "Anyways, let's head in."
```

Every one inside the region the heuristic armed, in the order they appeared on screen.
Compare that to what the mod had before: a 1.4 GB heap scan producing a ranked list, an
anchor whose address had been different in each of the last four runs of the same scene,
and no way to tell a correct ranking from a lucky one.

The point is not that the hook is faster. It is that the previous system could not be
*checked*. `in-pool` versus `elsewhere` is the first line of output in this project's
history that can falsify a claim about the dialogue buffer.

### The twin was never a variant

Each in-pool record arrived paired with one outside the pool, read within a millisecond:

```
0x424CAC6BDC - 0x42104BB2EC = 0x3C60B8F0    "Here we are..."
0x424CAC6E90 - 0x42104BB5A0 = 0x3C60B8F0    "It's pay per visit..."
0x424CAC7008 - 0x42104BB718 = 0x3C60B8F0    "Oh yeah! You bring..."
0x424CAC70E8 - 0x42104BB7F8 = 0x3C60B8F0    "Nah, they'll lend..."
0x424CAC74B8 - 0x42104BBBC8 = 0x3C60B8F0    "Anyways, let's head in."
```

A constant delta across every pair. The two regions are not two takes of the script and
not two stages of a pipeline — they are parallel allocations with identical internal
layout, where message *k* sits at the same offset in both.

That retroactively explains an earlier fix. Arming the twin by matching sample text
worked, and matching by rank never did, and the reason was structural: the copies are
byte-identical, so content finds them and a similarity score has no reason to place them
near each other. The fix was right for a reason we had not yet earned.

### One byte, three bugs

The oldest cosmetic complaint — both rows of the bubble showing the same sentence — came
from a single misread byte:

```
0x4250B2FBC7 len=33  "The equipment's kinda crappy, but"
0x4250B2FBE9 len=25  "they got tons of variety."
```

`0xFBC7 + 33 = 0xFBE8`, next run at `0xFBE9`. One byte between them, and it is `0x0A`.
These were never two strings. They are one message with a line break, and the scanner
manufactured the boundary by splitting on any non-printable byte.

So the write was wrong at the level of *units*. It wrote the full generated line into
fragment one, then the full generated line into fragment two, because it believed they
were independent slots. They are rows of one bubble. Rows are not independent, so they
cannot be written independently.

Grouping is trivial once seen: consecutive runs one byte apart are one record; a genuine
record boundary shows a gap of tens of bytes, because a header sits there. Then the
generated line is word-wrapped *across* the fragment widths, and any fragment the text
does not reach is blanked — a row of scripted dialogue under a row of generated dialogue
is worse than a short line.

### Checking a boundary without spending a restart

There is no C# test project here, and every in-game verification costs a relaunch, a
load, and a walk back to a hang-out. So the wrapping logic was mirrored in twenty lines
of Python and run against the cases that actually break wrappers: exact fit, empty
input, a word wider than its fragment, text far longer than all fragments combined.

That is not a substitute for a test, and it does not live in CI. It is a way to spend
thirty seconds instead of five minutes on the class of bug — off-by-one at a boundary —
that is both the most likely and the least visible in a screenshot.

### What is left

The hook now reports the address of each line as it displays. It does not yet drive the
write: generation still happens once at scene start and fills every record with the same
sentence. Per-line generation is the next thing the hook makes possible, and it is a
scheduling problem rather than a discovery one — which is a considerably better kind of
problem to have.

---

## Chapter 70: Blast Radius Is a Scheduling Problem

### The symptom, in the player's words

The text log showed one generated sentence three times, at three different widths:

```
Ryuji   Your gym game's comin' along,
        ain't ya?

Ryuji   Your gym game's comin' along, ain't ya?

Ryuji   Your gym game's comin' along,
```

The wrapping is correct — that is the previous chapter's fix working, one sentence
flowing across two rows instead of the same sentence twice. What is wrong is that three
*different messages* now hold the same text. The log says why:

```
[POOL] wrote 93 records across 2 regions <- "Your gym game's comin' along, ain't ya?"
```

Ninety-three records, one sentence. Every past and future line of the scene, overwritten
with whatever was generated most recently.

### Why it was ever written that way

Not laziness — ignorance, of a specific and now-fixed kind. The mod could not tell which
record the game would render, so it wrote all of them and let the renderer pick. That is
a rational strategy when you cannot observe the consumer. It stops being rational the
moment you can.

But knowing *which* record is only half of it. The other half is *when*, and that
question had never been asked. Timestamps answer it:

```
15:23:09.300  [MSG]   msgId=0x348                              dispatch
15:23:10.883  [LLM]   "You still comin' here with me, then?"   +1.583s
15:23:10.885  [POOL]  write                                    +0.002s
15:23:12.299  [WATCH] record=0x421532EB82 rendered             +1.414s
```

Inference takes 1.6 seconds and the write lands 1.4 seconds before the renderer reads
the bytes. There was never a race. The whole 93-record blast radius existed to cover a
timing risk that measurement shows does not exist.

That is the general shape worth keeping: **an over-broad write is often a scheduling
question in disguise.** Before widening a fan-out to be safe, timestamp the actual
sequence. "We write everything because we do not know which one" and "we write everything
because we might be too late" are different problems, and only one of them was real here.

### Which record is next

Two facts from the log make the target unambiguous. Reads run monotonically upward
through the region — `0x421532EB82` then `0x421532EC18` — so the record after the one
just rendered is the next in address order. And the hook reports the record *base*, which
sits a header ahead of the text, so matching is not containment but "the first record
whose text begins at or just after this address".

One subtlety costs a line if missed: the cursor advances **on render, not on write**. A
write that never reaches the screen — the player skipped, the scene branched — must not
consume a record, or the mod runs one line ahead of the game for the rest of the scene.
And the cursor never moves backwards, because scrolling the backlog re-reads earlier
records and that is not progress.

### What one record buys

Three problems close together, and they were never really three:

- The text log stops repeating, because only the upcoming line is touched.
- The blast radius becomes one record. The crash from Ch. 63 came from writing 211 slots
  into a region containing a data table; a single-record write cannot reach one.
- The write becomes cheap enough to do per line rather than once per scene, which is what
  makes per-line *generation* the next step rather than a rewrite.

The unifying point: precision about the target was worth more than any amount of care
about the write itself. We had spent chapters making the write safer — bounds checks,
word-boundary trimming, kill switches — while it was still aimed at ninety-three places
at once.

---

## Chapter 71: Three Symptoms, Three Different Bugs

A single play session produced three complaints — the text changed under the player, the
generated lines stopped reaching the bubble, and the game hitched when the backlog
opened. It is tempting to look for the one cause behind all three. There wasn't one.
They shared a session, not a mechanism, and each needed its own fix.

### Symptom 1: the sentence changed while it was on screen

```
15:34:32.729  [POOL] region 0 record 8/22   <- "You're really gonna start comin' here..."
15:34:36.535  [POOL] region 0 record 8/22   <- "This gym's not bad for the price, though."
15:34:39.098  [POOL] region 0 record 8/22   <- "Your gym skills are gettin' better, Joker!"
```

Same record, three times, four seconds apart. Whatever was displaying record 8 got
rewritten twice under the player.

The cause is a design decision from the previous chapter, and the reasoning behind it was
sound: advance the write cursor **on render**, so that a line which never reaches the
screen cannot leave the mod permanently one record ahead of the game. That protects
against drift. What it does not do is stop the *next* write landing in the same place
when no render happens in between.

The fix is to treat both as floors. A record is spent when it is written, whether or not
it is ever displayed; render observation still pushes the cursor forward when the game
gets ahead. Progress has two sources and the cursor takes the larger.

The general lesson: "advance on X" is rarely a complete rule. Ask what happens when X
does not occur, because that is the branch the bug lives in.

### Symptom 2: the bubble kept the original script

```
[POOL] 1 regions armed for write
...
15:34:42.469  [WATCH] elsewhere record=0x41DB1930C8  "Oh yeah! You bring your stuff?"
15:34:42.470  [WATCH] in-pool   record=0x4209925338  "Oh yeah! You bring your stuff?"
```

Two copies of the line, read a millisecond apart, and only one of them inside an armed
region. The write went to the copy the text log reads; the bubble read the other and
showed the script.

This is the twin problem, which we thought was solved by arming any region whose sample
text matched the anchor's. It resolved in earlier runs and silently failed in this one —
`1 regions armed`. Ranking never had a way to find the twin: the relationship between the
two copies is a fixed address offset, not a similarity of score, and no amount of tuning
a text-similarity metric will surface a fact the metric does not measure.

The hook does measure it. Two consecutive distinct records with identical preview text
are the pair, and their difference is the delta — `0x3C60B8F0` in one run, `0x2E67F270`
in the next, constant within each. So the twin is now learned by observation and written
by arithmetic.

Arithmetic on an address is exactly where a bad assumption becomes a memory corruption,
so the mirror write never trusts the delta alone. The original bytes are captured before
the target is overwritten, and the mirror is written only if it still contains them, byte
for byte. A wrong delta fails the comparison and writes nothing. This is the Ch. 64
ordering constraint reappearing in a new place: the evidence you need is destroyed by the
operation you are about to perform, so capture it first.

### Symptom 3: the game hitched when the backlog opened

The backlog re-reads every past line at once. That is not a bug — but each read produced
a log line, and each log line opened the file, appended, and closed it.

That per-line open was a deliberate earlier choice: a buffered writer loses its tail
exactly when the process crashes, which is when the tail matters most. The reasoning was
right and the conclusion was wrong, because `AutoFlush` on a held-open writer provides the
same crash-safety without the syscall per line. The cost was invisible at a 500 ms tick
and became a stutter under a burst.

The log volume needed capping too, but note which fix goes where: the *processing* of
every record still happens — cursor, twin delta — and only the *reporting* is limited to
six per tick with a count of the rest. Throttling the work would have introduced a third
bug to fix later.

### Why they looked like one bug

All three surfaced in the same minute, and two of them made the mod's output look wrong
in the same way. The instinct to find a common root is usually right and was wrong here,
and the thing that separated them was timestamped evidence: three writes to one record
is a cursor problem, two addresses holding one line is a coverage problem, and a burst
correlated with a keypress is a cost problem. Without the log they would have been one
vague report of "it's glitchy", and any single fix would have left two of them live.

---

## Chapter 72: Generate Ahead, Because You Cannot Outrun a Button

### The problem latency actually creates

The pipeline today is reactive:

```
[MSG] dispatch  ->  generate (1.6-2.2s)  ->  write the next record  ->  player reads it
```

That works while the player reads at a human pace and falls apart the moment they do
not. Hold fast-forward and the write lands after the line is drawn. Skip a scene and the
record is consumed without ever being seen. Read slowly and nothing is wrong, which is
the trap: the bug is invisible in exactly the conditions you test under.

The instinct is to make generation faster. Two seconds is already good for an 8B model on
a laptop GPU, and it will never beat a held button. Latency is not the problem; *waiting
until the last moment to start* is.

### What the hook changed about what is knowable

Before the interpreter hook, reacting was the only option, because the buffer was
discovered as it was used. That is no longer true. At arm time the mod holds:

- every record in the scene, in order,
- each record's exact capacity in characters,
- each record's original scripted text.

Nothing in that list requires the player to have reached the line. There is no reason to
wait for a dispatch event to know what Ryuji says four lines from now — the script is
already in memory, and so is the space his replacement has to fit into.

So the design inverts. Instead of *an event triggering a generation*, a background
scheduler keeps the next few records filled, and the player's arrival is merely when the
work becomes visible:

```
arm  ->  plan all 22 records
     ->  keep K ahead generating
     ->  each result written seconds or minutes early
     ->  player pacing stops mattering
```

### Why this does not preclude reacting to the player

The obvious objection is that the end goal is the player typing something and Ryuji
answering — which is reactive by definition, and cannot be pre-generated.

Both fit, because the two cases are disjoint and the reactive one is where latency is
free:

| Case | Player input | Time available | Strategy |
|---|---|---|---|
| Ryuji monologue | none | ~0, they can hold FFWD | pre-generate |
| Choice prompt (`SEL`) | menu navigation | 1-3s | reactive |
| Typed line | typing a sentence | 5-15s | reactive |

The player can only outrun inference when they are contributing nothing. The moment they
contribute, they have spent seconds doing it. You cannot fast-forward through your own
typing.

And the two compose rather than compete, because the hook reports when a record has been
*read*:

```
pre-generated line sits in record N            <- floor
player types; reactive generation starts
  arrives before N is reported rendered        -> overwrite
  arrives after                                -> discard; the floor stands
```

That second branch is the "text changed under the player" bug from Ch. 71, restated as a
rule: **a record is writable until the hook says it was read, and frozen afterwards.** The
bug becomes the safety property.

### The state a record needs

Pre-generation means a record is no longer something written once at an event. It has a
life cycle, and every transition matters:

```
Pending  -> InFlight -> Ready -> Written -> Rendered
   ^                                          |
   |                    (never goes backwards)|
```

`Ready` is separate from `Written` because generation completing and the bytes landing
are different moments with different failure modes. `Rendered` is the freeze point. And
the arrows only run one way — Ch. 71 was two bugs caused by a cursor that could be moved
by the wrong event, so this one advances on explicit transitions and never recomputes
itself from a timer.

### The cost of being early

Pre-generation buys pacing-independence and gives up immediacy: a line written four lines
ahead cannot respond to what happened three lines ago. For scripted monologue that costs
almost nothing, because the model is working from the scene script either way. It would
cost a great deal for a reply to player input — which is precisely the case that stays
reactive.

Being early is not free, and it is worth naming the trade rather than discovering it
later as a vague sense that the dialogue feels less connected.

---

## Chapter 73: Deleting the Search

### What the scan cost

```
15:33:54  Hang-out begins
15:34:27  [POOL] scanned 4030 regions, 1433 MB, 18 scored
15:34:27  [POOL] ARM #0 (anchor) 0x42098F1000
```

Thirty-three seconds, 1.4 GB walked, to end up with a ranked list whose top entry was
right most of the time. During those thirty-three seconds the scene was playing and
nothing could be written, so the opening lines were always the scripted ones. That was
never a bug report, because it looks exactly like "the mod warming up".

The scan also could not start until a message had been dispatched, since that was the
only signal that a conversation existed. So the latency was structural: discover late,
then discover slowly.

### The replacement is one call

```csharp
(bool ok, nuint regionBase, nuint regionSize, uint state, uint _) =
    MemoryGuard.QueryRegion(record);
```

`VirtualQuery` on an address the interpreter just read. Microseconds, and exact rather
than ranked, because the address came from the game reading it.

The interesting part is not the speed. It is that the question changed. The scan asked
*"which of 4030 regions looks most like dialogue?"* — a search, answered with a
similarity heuristic that had to be tuned, budgeted, and periodically re-explained when
it picked a data table. The hook asks *"what region is this address in?"* — a lookup,
answered by the operating system.

Searching for a thing you can be handed is the expensive kind of mistake, because it
does not feel like a mistake. It feels like engineering: there is a scoring function to
improve, a budget to tune, a ranking to inspect. All of that work was real and none of
it was necessary once something else in the system knew the answer.

### Validation is not the same as searching

Removing the search does not remove the need to be careful. The interpreter serves every
string in the game — menu labels, item names, the newspaper — so the first record it
reads is not necessarily dialogue. Four gates remain:

- an active hang-out,
- a preview at least twelve characters long, because a scene line is a sentence,
- a committed, writable region of plausible size,
- at least six text runs inside it.

That looks like the old heuristic and is doing something different. The scan used
scoring to *choose among thousands of candidates*; this uses thresholds to *reject one
candidate the game pointed at*. The first needs to be right about relative ranking across
a huge space and is fragile in proportion. The second only has to notice that a menu is
not a conversation.

### Keeping the corpse warm

The scan is switched off, not deleted, for one release. The hook path has been exercised
on exactly one scene, and a slow, occasionally-wrong pool beats no pool at all in a scene
it turns out to miss.

That is a real judgement rather than sentiment, and it comes with an expiry: it goes when
the hook has been through a few more scenes. Dead code kept "just in case" with no
condition for removal is how a codebase accumulates two ways of doing everything, and the
second way rots until the day someone needs it.

---

## Chapter 74: A Session Is Not a Scene

### The wrong pool, armed confidently

The hook-driven arming worked exactly as designed and produced a completely wrong result:

```
16:51:43  [ARM] 0x41DBE73000 len=552960 slots=402
16:51:43  [PLAN] 207 records
16:51:43  [PLAN]   #0 "Just believe in yourself, dude!"
...
16:51:57  [TWIN] ... 0x4250C8B13C "Here we are... Protein Lovers gym!"
```

The armed pool is the street conversation that opens the hang-out. The gym pool appears
fourteen seconds later and is never armed, because arming had already happened.

The bug is a modelling error, not a coding one. `_sessionActive` was treated as the unit
of work — one hang-out, one pool, arm once — and a hang-out turns out to contain several
scenes, each with its own pool. Every assumption downstream inherited it: the plan, the
cursor, the whole generation queue, all bound to a buffer the player had walked away from.

The fix is to stop asking "have we armed yet" and start asking "is the game still reading
from what we armed". Reads landing outside the armed region are the scene change
announcing itself. Two in a row, because one stray read is a menu or a name plate while a
genuinely moved scene keeps reading from its new pool.

One subtlety makes or breaks that rule: the twin copy is always outside the armed region,
so without excluding it explicitly the pool and its copy take turns evicting each other
for the length of the scene. A re-arm trigger has to know which "outside" reads do not
count.

### Two regions were never copies

```
[POOL] region 0 record 0/207
[POOL] region 1 record 0/180
```

Arming two regions and writing the same record index into both assumed they held the same
script. 207 records against 180 says otherwise — two unrelated pools, where index N names
a different line in each.

What makes this worth recording is that the correct mechanism was already present and
working. `MirrorToTwin` locates the copy at a learned offset and verifies byte-for-byte
before writing. The second armed region was a second, weaker answer to a question that
already had a good one, kept because it had been written first.

Two mechanisms for one job is not redundancy. The weaker one runs too, and its failures
are attributed to the system rather than to itself.

### Late answers land in the past

```
[PREGEN] #0 written, -32 ahead of the player
```

The request went out with the cursor at 0. The player advanced thirty-two records during
the round trip. The answer arrived and overwrote a line spoken half a minute earlier.

There was already a guard for this — records freeze once the interpreter reports reading
them. It did not fire, because the interpreter only reports what it actually read, and a
scene moved through quickly leaves gaps nobody looked at. The freeze is a record of
observation, not a claim about everything behind the cursor.

So the write path needs its own floor: never write behind the cursor, regardless of what
was observed. Two guards for one hazard is fine when they fail differently — unlike two
mechanisms for one job, which is the previous section's problem.

### The pattern across all three

Each of these is the same shape: a rule that was true when written and quietly stopped
being true.

- "One session, one pool" was true while every test was a single scene.
- "The armed regions are copies" was true for the run where content matching found a
  genuine twin.
- "Freezing on render covers everything behind the player" was true while the player read
  every line.

None of them failed loudly. They produced confident, well-logged, wrong behaviour — which
is the argument for logging the *reasoning* and not only the outcome. `record 0/207`
against `record 0/180` is the entire second bug, visible in one line, and only because
the log prints the denominator.

---

## Chapter 75: The Header You Have Been Skipping Over

Every record the interpreter hands us starts with twenty-four bytes we have spent the
whole project stepping around.

`ReadRecordPreview` opens with a loop whose entire purpose is to get past them:

```csharp
for (int start = 0; start < Window; start++)
{
    if (!IsPrintable(_recordBuf[start])) continue;
    ...
    if (end - start < 8) { start = end; continue; }
```

Skip the unprintable bytes, skip anything shorter than eight characters, take the first
run that reads like English. It works. It also throws away the answer to the question we
sat down to solve today.

### The bug, stated precisely

A Takemi rank-up scene is not Takemi talking for twelve records. It is Takemi, a patient,
the patient's father, and a narrator, interleaved. The mod rewrites all of them in
Takemi's voice, because the only thing it knows about a record is that it is a record.

The failure is not subtle in play. The father says something dry and clinical about his
daughter's treatment; a nurse asks a question and answers herself in a doctor's cadence.
The lines are individually decent — the model is doing its job — and collectively wrong,
because they are attributed to whoever happens to be on screen.

There is no heuristic fix here. You cannot infer the speaker from the text, because the
text is the thing being replaced. Whatever tells us who is talking has to be read before
the first write, from the game's own data.

### What is actually at record+0

Atlus ships dialogue in `.BMD` files (`MSG1` format, shared across Persona 3/4/5 and
documented by the AtlusScriptToolkit project). A dialogue message inside one is a struct,
not a string:

```
offset  size  field
0x00      24  Name          e.g. "MSG_001_5_0", NUL-padded ASCII
0x18       2  PageCount     int16 — speech bubbles in this message
0x1A       2  SpeakerId     uint16 — index into the file's speaker table, 0xFFFF = none
0x1C   4*n    PageStartAddresses   int32 each, n = PageCount
 ...       4  TextBufferSize
 ...       ?  TextBuffer    the bytes we have been rewriting
```

Read that layout against the one measurement we already have. Ch. 69 recorded the first
character of a live record at `+0x28`:

```
0x1C + 4*PageCount + 4 = text
       PageCount = 1  ->  0x24
       PageCount = 2  ->  0x28   <-- the one we measured
       PageCount = 3  ->  0x2C
```

A two-page message. The layout predicts the number we already had, which is the only kind
of confirmation worth anything before a scene has been run.

It also explains the cursor. `RDX` at the hook site is not "characters printed so far
from the top of the text" — it is a byte offset from the *record* base, and the first
value it can hold is the page start address, which is `0x28` for a two-page message. We
have been reading the page table's contents out of a register without recognising them.

And `MSG_001_5_0`, the symbol we have been seeing in hex dumps and treating as pool
scenery, is not scenery. It is field one of the struct we are standing on.

### Why the name matters as much as the speaker id

`SpeakerId` is an index, and an index needs its table. That table lives in the `MSG1`
file header, which is somewhere behind us in the region — findable, but a second problem.

The name is free. It is right there at offset zero, it is ASCII, and it carries structure:
`MSG_001_5_0` is a message id, and `SEL_003` (which the interpreter also serves) is a
selection prompt — a menu, not a line of dialogue. Rewriting a menu option is a different
and worse bug than rewriting the wrong speaker, and one string comparison rules it out.

So the first cut is: read the header, log the name and the speaker id for every record we
watch, and change nothing else. Two scenes of evidence, then decisions.

This is the discipline Ch. 74 arrived at the hard way. Every bug in this project has been
a rule that was true when it was written — "one session, one pool", "the armed regions are
copies", "reads run monotonically". Each one was reasonable, each one was tested against
exactly one scene, and each one was wrong the first time a different scene ran. The layout
above is a claim from documentation plus one confirming offset. That is a hypothesis, and
it gets logged before it gets trusted.

### The part that is not a bug fix

Attribution is also the missing half of the context.

Right now a generated Takemi line is given the scripted line it replaces and Takemi's own
previous lines. It is never told that the line before hers was the father asking a
question, because the mod has no concept of "the father". Once records carry a speaker,
the ones we do *not* rewrite stop being obstacles and become the conversation: the model
can answer the question that was actually asked.

The scene stops being a list of slots to overwrite and becomes a script with parts.

---

## Chapter 76: An Index Needs Its Table

`SpeakerId` is a number. Ch. 75 got it out of the record header, and on its own it is
worth almost nothing: knowing that record 4 and record 9 share speaker 2 tells you they
are the same person, not who that person is.

The mod needs the name, because the decision it has to make is "is this the confidant we
are generating for?" — and the only handle it has on the confidant is a name, read out of
a completely different structure (the CMM session struct, `ConfidantId` → `arcana.py`).

### Where the table lives

The `MSG1` file that contains the records also contains the table:

```
0x00       1  FileType
0x01       1  Format
0x02       2  UserId
0x04       4  FileSize
0x08       4  Magic                "MSG1"
0x0C       4  ExtSize
0x10       4  RelocationTable
0x14       4  RelocationTableSize
0x18       4  DialogCount
0x1C       2  IsRelocated
0x1E       2  Version
0x20     8*n  DialogEntry[n]       { int Kind; int Address; }
 ...      16  SpeakerTable         { int ArrayAddress; int Count; int ExtAddress; int _ }
```

`ArrayAddress` points at an array of `Count` addresses, each pointing at a NUL-terminated
name. Two hops from an index to a string.

### The problem with every address in that layout

None of them are file offsets. They are relative to a base, and the base is a convention
of the format rather than something stored in it. The one the toolchain uses is file start
plus `0x10`; whether P5R's runtime copy keeps that after relocation is not something this
mod can look up.

The tempting move is to pick one and move on. Do not. A wrong base here does not crash and
does not return empty — it walks into the middle of the dialogue data and reads whatever
NUL-terminated run it lands on. The failure mode of guessing is *plausible names for the
wrong speakers*, which is indistinguishable from success until someone plays the scene and
notices the nurse talking like a doctor. That is precisely the class of bug Ch. 74 is about.

### Let the data decide

There is a check available that costs nothing. The dialogue table is a list of addresses to
message records, and Ch. 75 already knows how to tell whether bytes are a message record.

So: try each candidate base, resolve the first few dialogue entries, and require that they
land on something `BmdMessage.TryParse` accepts — twenty-four bytes of printable ASCII with
exact NUL padding, followed by a plausible page count. A wrong base fails that immediately,
because the interior of a text buffer does not look like a header.

The base is not assumed, it is *recovered*, and the recovery is verified against a
structure we can independently parse. This is the same shape as the self-consistency test
in `TryFindHeader` — accept the interpretation that predicts data you can already check —
and it is going to keep being the answer, because reverse engineering never gives you a
spec, only agreement between structures.

### The magic is the anchor

One more circularity to break: the parse needs the file start, and all we have is a record
address somewhere inside the file.

`MSG1` is four bytes at file+8, so the file start is a backwards search away. Nearest match
wins, and that is not a tie-break — several BMDs are resident at once, and a record belongs
to the file immediately behind it. Searching to the front of the region and taking the
first hit would attribute a Takemi scene to whatever was loaded before it.

### What is still guesswork, stated plainly

Two things:

- Whether the runtime copy is relocated at all. If P5R rewrites the addresses into real
  pointers on load, neither candidate base resolves and the whole struct fails — which is
  fine, because it fails *closed*: no table, no names, no attribution, and the mod keeps
  behaving exactly as it did yesterday.
- Whether P5R's speaker names are plain ASCII. They carry inline formatting codes, which
  are stripped rather than rendered, so the worst case is a name with a character missing
  rather than line noise in a prompt.

Both are answered by one line of log from one real scene, which is where this stops being
a chapter and starts being evidence.

---

## Chapter 77: Two Names for the Same Person

The speaker table gives a label. The session struct gives a confidant id, which
`ConfidantNames` turns into a display name. Attribution is the question of whether those
two strings are the same person, and they are almost never the same string.

The game labels a bubble `Takemi`. The mod calls her `Tae Takemi`. It labels one `Dr.
Takemi` when the patient is speaking to her, `Ryuji` where the mod says `Ryuji Sakamoto`,
and `???` before an introduction.

### Why not just substring

`speaker.Contains(name) || name.Contains(speaker)` handles the easy cases and quietly
breaks on the interesting one. `Sakura` is inside `Futaba Sakura` and inside `Sojiro
Sakura`. In a Futaba scene, every line Sojiro speaks would be attributed to Futaba and
rewritten in her voice — which is exactly the bug this whole chapter exists to remove,
reintroduced by the fix for it.

`Niijima` is the same problem for Makoto and Sae, and those two share scenes constantly.

### Tokens, minus the parts that are not names

So compare word by word. Lowercase, drop anything that is not a letter (which handles
`Dr.` losing its full stop, and `???` reducing to nothing at all), and drop the words that
are titles rather than names — `dr`, `mr`, `ms`, `sensei`, `san`, `kun`, `chan`, `senpai`.
Tokens shorter than three letters go too; nothing distinguishing survives at two.

A shared token is then a candidate match. `Takemi` ∩ `{tae, takemi}` is non-empty. So is
`Sakura` ∩ `{futaba, sakura}` — which is where the interesting part starts.

### Let the table say which tokens are worth trusting

A token that appears in more than one confidant's name cannot identify anybody. And the
mod already holds every confidant's name, so it does not have to be told which tokens
those are — it can work them out:

```
sakura   -> Futaba Sakura, Sojiro Sakura     ambiguous
niijima  -> Makoto Niijima, Sae Niijima      ambiguous
takemi   -> Tae Takemi                       distinctive
futaba   -> Futaba Sakura                    distinctive
```

Match on distinctive tokens only. `Sakura` alone now matches nobody, which is the correct
answer — the label genuinely does not say which one — while `Futaba` and `Sojiro` both
still resolve. The rule maintains itself: adding a confidant whose surname collides with
an existing one silently moves that surname into the ambiguous set, rather than silently
creating a mis-attribution.

This is the second time in two chapters that the answer has been *derive it from data you
already hold* rather than *write down what you believe*. That is not a coincidence. Every
hand-written table in this project has been wrong at least once — `ConfidantNames` was
wrong about Takemi for months because two entries were missing above her.

### What a non-match means

Not "skip". Not yet.

A record that is not the confidant's is still part of the conversation, and up to now the
mod has had no way to say so. The scripted line is left alone *and* handed to the model as
context, attributed to whoever the table named. The line the confidant says next is then a
reply to a question that was actually asked, instead of a plausible sentence dropped into a
gap.

The scene stops being a list of slots and becomes a script with parts — which was the
closing sentence of Ch. 75, and is the point of the whole exercise.

### Failing closed

Every step here can fail: no archive, no speaker table, an unknown label, an ambiguous one.
All of them resolve to the same answer — *not attributable* — and the gate treats "not
attributable" as "do not rewrite".

That is the conservative direction, and it costs coverage. A scene whose archive does not
parse rewrites nothing at all, where yesterday it rewrote everything. So the gate ships
behind `speaker_filter_enabled`, off by default, and the first run produces the log that
says whether the attribution is trustworthy. Turning it on is a decision made with evidence
in hand, not a hope compiled in.

---

## Chapter 78: The Pointer Was Never the Record

Ch. 75 opened with "every record the interpreter hands us starts with twenty-four bytes we
have spent the whole project stepping around." A scene later, the count is nought for
sixty-five:

```
[SPEAKER] 0/65 records attributed; none
[SPEAKER]   #0 ? spk=65535
```

Not a partial failure, not a wrong speaker id. `BmdMessage.TryParse` never once returned
true against live memory.

### The line that settles it

```
[WATCH] in-pool record=0x4211C245E3 cursor=0x7 "Girl's Father"
```

`Girl's Father` is a speaker name, and a speaker name is a bare NUL-terminated string —
it has no page count, no page table, no header of any kind. The interpreter loaded its
address into RAX and started fetching characters, exactly as it does for a line of
dialogue.

So `mov rax,[rbx+0x20]` is not a message record. It is *the string currently being drawn*.
The struct that owns it is somewhere else, and everything Ch. 75 built on top of that
address was built on a misreading.

The `[GAP]` dumps say the same thing from the other side:

```
[GAP] 27B between #3 and #4: 0A F2 23 00 00 F1 21 F2 05 FF FF F1 41 F7 61 09 01 01 ...
```

Twenty-seven bytes between one bubble and the next, and not one of them is ASCII. A
twenty-four byte name would be unmissable there. There is no header between the texts.

### Where the wrong number came from

Ch. 75's confirmation was that the layout predicted the text at `+0x28`, and `0x1C + 4·2 +
4` is `0x28` for a two-page message. That felt like the layout agreeing with a measurement.

It was not a measurement. `+0x28` came from `AdvancePoolCursor`, which matches a record
address to a text run *within a window*:

```csharp
const int HeaderWindow = 0x80;
...
if (textOff - offset > HeaderWindow) break;
```

`0x28` was one observation of a distance that the code permits to be anything up to `0x80`.
A tolerance got written down as a fact, and then a documented file format was accepted
because it reproduced the fact.

This is not the Ch. 74 failure — a rule that was true and stopped being true. It is worse
and simpler: a number that was never load-bearing, promoted to evidence because it agreed
with what I already wanted to believe. The right question about `+0x28` was "how was that
obtained?", and I did not ask it.

### What is still standing

Three things, all worth keeping:

- **The file is real and findable.** `MSG1 at 0x4211C22450 (8736B)` — the magic search
  works, the declared size is plausible, and the first line's text lands at `file+0x1D4`.
  Everything about locating the archive held; only the interpretation of its interior did
  not.
- **Speaker names live near the end of it.** `0x4211C245E3` is `file+0x2193` of `0x2220` —
  the last hundred and forty bytes. That is where a name array belongs, and the game
  reading strings out of it is direct evidence the array is there.
- **The rejection filter did its job.** `TryParse` was strict about NUL padding precisely
  so it would refuse rubbish, and when the layout turned out to be wrong it refused
  everything rather than inventing sixty-five plausible names. A loose parser would have
  attributed this scene confidently and wrongly, and the log would have looked like
  success.

### The thing the game handed over unasked

It draws the speaker's name. Through the same interpreter, from a fixed address, re-read
every time the bubble changes:

```
15:44:58.253  record=0x4211C22938 "So it seems her body's developed"
15:44:58.254  record=0x4211C245E3 "Girl's Father"
```

One millisecond apart. Who is speaking is observable without parsing anything at all —
which is Ch. 65's lesson arriving a second time, from a different direction: *ask the
consumer, not the format*.

It is not a complete answer, because pre-generation runs eight records ahead of the player
and this only reports the line being drawn now. But it is a free, exact ground truth for
the current line, and that makes it something better than a fallback — it is a way to
check whatever the format-parsing eventually claims.

### What happens next is a hex dump, not another hypothesis

Two hypotheses have now been offered for the interior of this file and one has been tested.
The next step is not a third. The file's first `0x60` bytes, the `0x60` bytes ahead of the
first line of text, and the tail where the names live are all cheap to log, and between
them they say what the dialogue table actually contains and where the headers actually are.

Guessing the layout has been tried. Reading it has not.

---

## Chapter 79: Every Address Points to Itself

One hex dump settled a layout that two hypotheses had failed to guess.

```
[BMDHEX] head +0000  07 00 00 00 20 22 00 00 4D 53 47 31 00 00 00 00  |.... "..MSG1....|
[BMDHEX] head +0010  AC 21 00 00 74 00 00 00 30 00 00 00 01 00 02 00  |.!..t...0.......|
[BMDHEX] head +0020  00 00 00 00 8C 01 00 00 00 00 00 00 1C 02 00 00  |................|
```

`FileSize` at `+0x04` is `0x2220`, and the file is 8736 bytes. `DialogCount` at `+0x18` is
`0x30` — 48. `IsRelocated` at `+0x1C` is 1. All as documented.

Then the dialogue table at `+0x20`: `Kind = 0`, `Address = 0x18C`. And the first message
header is at `0x1B0`, which `0x18C` is not, under any base I had tried.

### The rule

`0x1B0 − 0x18C = 0x24`, and `0x24` is where the address field itself sits — entry 0 begins
at `0x20`, its `Kind` occupies four bytes, its `Address` occupies the next four.

**A stored address is relative to the position of the field that stores it.**

```
dialog[0].Address  field @ 0x024, value 0x018C  ->  0x01B0   MSG_000_0_0
speakerTable.Array field @ 0x1A0, value 0x1FE0  ->  0x2180   the name array
speakerNames[0]    field @ 0x180, value 0x000C  ->  0x218C   "Takemi"
speakerNames[1]    field @ 0x184, value 0x000F  ->  0x2193   "Girl's Father"
speakerNames[2]    field @ 0x188, value 0x0019  ->  0x21A1   "Sick Girl"
```

Five for five, then fourteen more in a second file. That is what `IsRelocated = 1` means
here: the fields named by the relocation table have been rewritten in place as self-relative
deltas.

`RelocationTable` at `+0x10` is the exception and proves the rule — it is a plain file
offset, because it is the one field that cannot be listed in the table it points at. Both
files confirm it: `0x21AC + 0x74 = 0x2220`, and `0xDDC + 0x27 = 0xE03`, each landing
exactly on the end of its file.

### The message header was right the whole time

```
+01B0  4D 53 47 5F 30 30 30 5F 30 5F 30 00 ...   Name[24] = "MSG_000_0_0"
+01C8  01 00                                      PageCount = 1
+01CA  00 00                                      SpeakerId = 0  -> "Takemi"
+01CC  08 00 00 00                                PageStart[0], field-relative -> 0x1D4
+01D0  00 00 00 74                                TextBufferSize = 116
+01D4  F2 05 FF FF F1 41 ...                      text
```

Name, page count, speaker id, page table, text at `0x20 + 4·PageCount` — exactly Ch. 75.
The parser was never the problem. Ch. 78 concluded the layout was wrong when only the
addressing was, and the difference matters: `BmdMessage.TryParse` needed no change at all.

The confirmation is a system message from the other file, which has no speaker:

```
+00B8  03 00      PageCount = 3
+00BA  FF FF      SpeakerId = 0xFFFF
```

Three pages, nobody talking. That is a rank-up notification, and the sentinel is doing
precisely what the format says it does.

### Why the search found nothing

`TryFindHeader` walks back from a text run. It assumed the run begins where the text buffer
begins. It does not:

```
+01D4  F2 05 FF FF F1 41 F6 86 01 01 6D 01 01 01 01 01 ...   text buffer starts here
+01FC  2E 2E 2E 53 6F 2C 20 77 68 79 20 63 6F 6D 65 20      "...So, why come "
```

Forty bytes of control codes sit between the two, and the pool scanner — which looks for
printable ASCII — reports `0x1FC`. Every candidate header position was therefore shifted by
a distance that varies per message, so nothing was ever tested at the right offset.

Incidentally, that gap is where the notorious `+0x28` came from: `0x1FC − 0x1D4` is `0x28`.
The number was real. It was the length of a control-code prefix, and I read it as the size
of a header.

### Stop searching

There is no need to search backwards at all. `DialogCount` says how many messages there
are, and the dialogue table says where each one is. The entire scene enumerates in a loop:

```
for i in 0 .. DialogCount-1:
    field = 0x20 + 8*i + 4
    message = field + int32(file, field)
```

Exact, ordered, and it yields the name, the speaker id and the text offset of every message
in the file — including the ones the player never reaches, which is what pre-generation
needs and what the observation channel of Ch. 78 could never provide.

This is Ch. 73 again, one level down. That chapter deleted a heap scan because the game
would hand over the pointer if asked. This deletes a header search because the file has a
table of contents and I was walking backwards past it.

### One free consequence

The dialogue data ends where the speaker name array begins, at `0x2180`. Everything past
that is names and the relocation table — not dialogue, and never safe to overwrite.

`"Girl's Father"` is a thirteen-character run of printable ASCII sitting inside an armed
region. The pool scanner has been finding it, and the only reason it was never rewritten is
that thirteen characters fall below the generation floor. That was luck, not design, and
knowing where the dialogue ends turns it into design.

---

## Chapter 80: Two Files, Two Answers

The self-relative rule from Ch. 79 went in, and the next scene split cleanly down the
middle.

```
[SPEAKER] MSG1 at 0x424C630190: 14 messages, 0 speakers, dialogue ends at file+0xDDC
[SPEAKER] 35/35 records attributed; narrationx35
[SPEAKER]   #0 CMM_EVENT_OPEN spk=65535
[SPEAKER]   #4 CMM_EVENT_RANKUP_2 spk=65535
```

Fourteen messages, every one located, every name read, and the speaker sentinel correct for
a file that has no speakers at all. The rule is right.

```
[SPEAKER] MSG1 at 0x41DC897020 (8736B) did not resolve
```

The clinic file, same 8736 bytes as the dump that produced the rule, still fails.

### All or nothing was the wrong contract

`TryParse` refuses the archive if any one of the 48 dialogue entries does not resolve. That
was a deliberate choice in Ch. 76, made for a good reason — half a scene attributed under a
wrong rule is worse than none, because it looks like it worked.

The reason no longer applies. The rule is not in question any more; a second file confirmed
it end to end. What is in question is whether every *entry* in a file is the same kind of
thing, and one odd entry out of 48 should not cost the other 47.

But the fix is not simply to skip the odd one. Skipping leaves a hole in the offset map, and
containment matching would then hand that hole's text to the previous message — silently
attributing one character's lines to another, which is the original bug wearing a different
hat.

So an entry that does not parse becomes a **hole that is still on the map**: recorded at its
own offset, named nothing, speaker `NoSpeaker`. Text landing inside it resolves to
"unattributed", which the filter already treats as "leave this line alone". The archive
degrades to less coverage instead of to wrong coverage.

### What the odd entry probably is

The scene has a choice in it — `"Yes, I'm sure!"` appears in the watch log, and that is a
menu option, not a line. The format has a second record kind for exactly this, and the
dialogue entry's `Kind` field distinguishes them:

```
MessageDialog                    SelectionDialog
  char[24] Name                    char[24] Name
  int16    PageCount    @0x18      int16    Ext          @0x18
  int16    SpeakerId    @0x1A      int16    OptionCount  @0x1A
  int32[]  PageStarts             int32[]  OptionStarts
```

Same shape, but the count moves from `0x18` to `0x1A` and there is no speaker at all. Read
as a message, a selection offers whatever sits in `Ext` as its page count — and
`BmdMessage.TryParse` rejects an implausible one, which is exactly what it is built to do.

That is a hypothesis, not a measurement, and this chapter has learned not to promote one to
the other. It ships with the hole mechanism underneath it, so if it is wrong the file still
parses and the selections come out unattributed. And the log now names the first entry that
failed and dumps its bytes, so a second wrong guess costs one line of log rather than a
session.

Selections must never be rewritten regardless. Changing what the buttons say without
changing what they do is a worse bug than a mis-attributed line.

### The other thing this scene taught

Zero records were replaced, and the mod's log gave no reason. The server console did:

```
POST /generate HTTP/1.1" 503 Service Unavailable
```

The inference backend was down for the entire run. On the mod's side that path retries three
times at eight-second intervals and then throws, and the throw is caught by a handler that
logs nothing on the first attempt. So a dead backend looked, from the log, like a queue that
simply never produced anything — and the only visible trace was a thirty-second gap between
two `[PREGEN] requested` lines, which you have to already suspect the answer to notice.

This is the same defect as the 429 storm of Ch. 72, in the same place, found the same way:
by reading the *server's* log because the mod's log was silent. The rule that keeps being
relearned is that a failure which is handled is not a failure which is understood, and the
handler is exactly where the log line belongs.
