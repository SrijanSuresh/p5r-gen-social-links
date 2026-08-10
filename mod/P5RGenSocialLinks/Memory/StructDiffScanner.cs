using System;
using System.Text;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Captures the first N bytes of the session struct and diffs them on each call.
/// Any byte that changes between calls is a candidate for a per-line counter.
/// Used during Phase 2 offset discovery — remove once live offsets are confirmed.
/// </summary>
internal sealed unsafe class StructDiffScanner
{
    private const int ScanBytes = 512;

    private readonly byte[] _previous = new byte[ScanBytes];
    private bool _hasPrevious;

    /// <summary>
    /// Compares the current struct bytes against the previous snapshot.
    /// Returns a log line describing changed offsets, or null if nothing changed.
    /// Resets baseline when sessionPtr changes (new hang-out).
    /// </summary>
    internal string? Diff(nuint sessionPtr)
    {
        if (!MemoryGuard.IsReadable(sessionPtr, ScanBytes))
        {
            _hasPrevious = false;
            return null;
        }

        byte* p = (byte*)sessionPtr;

        if (!_hasPrevious)
        {
            for (int i = 0; i < ScanBytes; i++) _previous[i] = p[i];
            _hasPrevious = true;
            return null;
        }

        StringBuilder? sb = null;
        for (int i = 0; i < ScanBytes; i++)
        {
            byte cur = p[i];
            byte prev = _previous[i];
            if (cur == prev) continue;

            sb ??= new StringBuilder("[StructDiff]");
            sb.Append($" +0x{i:X2}:{prev:X2}→{cur:X2}");
            _previous[i] = cur;
        }

        return sb?.ToString();
    }

    /// <summary>
    /// Resets baseline so the next call establishes a fresh snapshot.
    /// Call this when the session pointer changes.
    /// </summary>
    internal void Reset() => _hasPrevious = false;
}
