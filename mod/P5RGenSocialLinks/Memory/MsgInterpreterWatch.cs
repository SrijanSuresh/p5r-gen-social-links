using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;
using Reloaded.Hooks.Definitions;
using Reloaded.Hooks.Definitions.Enums;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Captures which message record P5R is rendering, by watching the interpreter read it.
///
/// The mod has spent its whole life inferring this. It scans the heap, scores regions on
/// how much they read like English, arms the best one plus any content-identical twin,
/// then overwrites every dialogue-shaped run inside it. That works often enough to be
/// convincing and fails in ways that look like anything but a ranking problem: a data
/// table outranking the scene and crashing the game, the same line stored in four takes
/// with no way to tell which one is live, both rows of a bubble showing one sentence.
///
/// The game never had that problem. At P5R.exe+17A3D27 the interpreter loads the record
/// pointer into RAX and the byte cursor into RDX, then fetches a character. Six
/// instructions injected ahead of it copy both into a fixed buffer, and the guessing
/// stops (learning.md Ch. 65-67).
///
/// Injected code, FASM syntax:
/// <code>
///   push rcx              ; the only register clobbered, and it is restored
///   mov  rcx, &lt;slab&gt;     ; absolute — the stub is not position independent
///   mov  [rcx],   rax     ; record pointer
///   mov  [rcx+8], rdx     ; byte cursor
///   pop  rcx
/// </code>
/// No instruction here writes flags, which matters: the original code's very next
/// instruction is TEST EDI, EDI followed by a conditional jump, so a stub that disturbed
/// the flags register would redirect the interpreter's control flow.
/// </summary>
internal sealed class MsgInterpreterWatch : IDisposable
{
    private const int RecordOffset = 0;
    private const int CursorOffset = 8;
    private const int SlabBytes    = 16;

    private readonly nint     _slab;
    private readonly IAsmHook _hook;

    internal MsgInterpreterWatch(IReloadedHooks hooks, nuint movzxAddress)
    {
        // Unmanaged on purpose. A pinned managed array would work, but its address has to
        // be baked into machine code that outlives any GC handle we could reason about;
        // unmanaged memory simply has no opinion about the collector.
        _slab = Marshal.AllocHGlobal(SlabBytes);
        Marshal.WriteInt64(_slab, RecordOffset, 0);
        Marshal.WriteInt64(_slab, CursorOffset, 0);

        string[] asm =
        {
            "use64",
            "push rcx",
            $"mov rcx, 0x{(ulong)_slab:X}",
            "mov [rcx], rax",
            "mov [rcx+8], rdx",
            "pop rcx",
        };

        // ExecuteFirst: the capture has to happen before the original MOVZX, because the
        // registers we want are its inputs. Running after would still see them here, but
        // only by accident of this particular encoding.
        _hook = hooks
            .CreateAsmHook(asm, (long)movzxAddress, AsmHookBehaviour.ExecuteFirst)
            .Activate();
    }

    /// <summary>
    /// Base address of the message record the interpreter last read from, or 0 before the
    /// first read. Writes are 8-byte aligned, so an x64 read cannot tear.
    /// </summary>
    internal nuint RecordPtr => (nuint)(ulong)Marshal.ReadInt64(_slab, RecordOffset);

    /// <summary>Byte offset the interpreter last read at, within <see cref="RecordPtr"/>.</summary>
    internal int Cursor => (int)Marshal.ReadInt64(_slab, CursorOffset);

    // --- sampling ------------------------------------------------------------------
    //
    // The slab holds one value: whatever the interpreter read most recently. Reading it
    // on the 500 ms poll tick therefore does not observe messages, it observes a clock —
    // a line that comes and goes between two ticks leaves no trace, and the log ends up
    // showing whichever records happened to survive until a tick landed.
    //
    // A dedicated thread reading at 5 ms turns a snapshot back into a sequence. It stays
    // separate from the poll loop because the poll loop does real work per tick (chain
    // resolution, heap scans) and cannot be sped up to match.

    private readonly ConcurrentQueue<(nuint Record, int Cursor)> _seen = new();
    private CancellationTokenSource? _samplerCts;
    private Thread?                  _sampler;

    /// Beyond this the queue is dropped rather than grown: nobody draining it means the
    /// consumer is gone, and an unbounded queue behind a 5 ms producer is a leak.
    private const int MaxQueued = 512;

    internal void StartSampling(int intervalMs = 5)
    {
        if (_sampler is not null) return;

        _samplerCts = new CancellationTokenSource();
        CancellationToken token = _samplerCts.Token;

        _sampler = new Thread(() =>
        {
            nuint last = 0;
            while (!token.IsCancellationRequested)
            {
                nuint record = RecordPtr;
                if (record != 0 && record != last)
                {
                    last = record;
                    if (_seen.Count < MaxQueued) _seen.Enqueue((record, Cursor));
                }
                Thread.Sleep(intervalMs);
            }
        })
        {
            IsBackground = true,          // must never hold up process exit
            Name         = "P5RGen.MsgSampler",
            Priority     = ThreadPriority.BelowNormal,  // never compete with the renderer
        };
        _sampler.Start();
    }

    /// <summary>
    /// Hand over every distinct record seen since the last call, oldest first.
    /// </summary>
    internal (nuint Record, int Cursor)[] DrainSeen()
    {
        var drained = new System.Collections.Generic.List<(nuint, int)>();
        while (_seen.TryDequeue(out (nuint Record, int Cursor) item)) drained.Add(item);
        return drained.ToArray();
    }

    public void Dispose()
    {
        _samplerCts?.Cancel();
        _sampler?.Join(TimeSpan.FromMilliseconds(200));
        _samplerCts?.Dispose();
        _hook.Disable();

        // The slab is deliberately never freed. Its address is an immediate operand inside
        // machine code the game may be executing on another thread at this instant, and
        // disabling a hook restores the original bytes without waiting for threads already
        // inside the stub to leave. Freeing 16 bytes would buy nothing and cost a write
        // through a dangling pointer at exactly the moment the process is shutting down.
    }
}
