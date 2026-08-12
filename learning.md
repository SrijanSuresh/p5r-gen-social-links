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
