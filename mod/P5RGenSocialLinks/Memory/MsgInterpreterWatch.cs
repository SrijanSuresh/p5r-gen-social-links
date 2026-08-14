using System;
using System.Runtime.InteropServices;
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

    public void Dispose()
    {
        _hook.Disable();

        // The slab is deliberately never freed. Its address is an immediate operand inside
        // machine code the game may be executing on another thread at this instant, and
        // disabling a hook restores the original bytes without waiting for threads already
        // inside the stub to leave. Freeing 16 bytes would buy nothing and cost a write
        // through a dangling pointer at exactly the moment the process is shutting down.
    }
}
