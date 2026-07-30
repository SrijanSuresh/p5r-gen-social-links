using System;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Polls the CMM dialogue line counter discovered via Cheat Engine (2026-08-09).
///
/// Address 0x006FFC28 holds a byte that increments each time the player presses
/// the confirm button to advance a Social Link dialogue line. Detecting this change
/// gives us a true per-line trigger — replacing the once-per-session CMM_EXEC_EVENT
/// dispatch with one LLM call per displayed line.
///
/// NOTE: The address is currently absolute (not module-relative). Verify it is
/// stable across game restarts. If not, replace with a pointer chain resolved
/// from CMM_LINE_COUNTER_STRUCT_BASE via Ghidra analysis.
/// </summary>
internal sealed class LineCounterMonitor
{
    private byte  _lastValue;
    private bool  _active;

    /// <summary>
    /// Call at hang-out start to record the baseline counter value.
    /// Without this, the first real line advance looks like a spurious change.
    /// </summary>
    internal unsafe void Activate()
    {
        _lastValue = ReadCounter();
        _active    = true;
    }

    /// <summary>Call at hang-out end to stop per-line detection.</summary>
    internal void Deactivate() => _active = false;

    /// <summary>
    /// Returns true if the line counter advanced since the last call.
    /// Intended to be called on every poll tick inside the active hang-out.
    /// </summary>
    internal unsafe bool HasAdvanced()
    {
        if (!_active) return false;

        byte current = ReadCounter();
        if (current == _lastValue) return false;

        _lastValue = current;
        return true;
    }

    /// <summary>Current raw counter value — useful for logging.</summary>
    internal unsafe byte CurrentValue() => ReadCounter();

    /// <summary>
    /// Logs the readability and raw value of the counter address.
    /// Call once at session start to confirm the address is live before relying on HasAdvanced().
    /// </summary>
    internal static unsafe void Diagnose(Action<string> log)
    {
        nuint addr = P5ROffsets.CMM_LINE_COUNTER_ADDR;
        bool readable = MemoryGuard.IsReadable(addr, sizeof(byte));
        if (!readable)
        {
            log($"[LineCounter] 0x{addr:X} NOT READABLE — address may have changed this session.");
            return;
        }
        byte val = *(byte*)addr;
        log($"[LineCounter] 0x{addr:X} readable, value={val} — polling active.");
    }

    private static unsafe byte ReadCounter()
    {
        nuint addr = P5ROffsets.CMM_LINE_COUNTER_ADDR;
        if (!MemoryGuard.IsReadable(addr, sizeof(byte))) return 0;
        return *(byte*)addr;
    }
}
