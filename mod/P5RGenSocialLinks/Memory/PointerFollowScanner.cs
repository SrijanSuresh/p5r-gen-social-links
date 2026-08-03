using System;
using System.Collections.Generic;
using System.Text;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Follows the session-struct pointers that appeared at hang-out start (+0xE0, +0xE8, +0xF0)
/// and diffs the first 256 bytes of each pointed-to sub-object on every poll tick.
///
/// Rationale: StructDiffScanner confirmed the line counter is not in the first 512 bytes
/// of the session struct. Those three slots hold pointers to CMM dialogue subsystem objects.
/// One of them contains the per-line counter. When Diff() reports a byte incrementing
/// monotonically with each dialogue advance, that offset becomes the new CMM_LINE_COUNTER_ADDR.
/// </summary>
internal sealed unsafe class PointerFollowScanner
{
    // Session-struct offsets that held pointer-like values (zero → heap addr) at hang-out start.
    // Observed in Takemi Session 40 StructDiff log (2026-08-10).
    private static readonly int[] PtrOffsets = { 0xE0, 0xE8, 0xF0 };

    private const int SubScanBytes = 256;

    private readonly nuint[] _targets;
    private readonly byte[][] _prev;
    private readonly bool[]   _hasSnapshot;

    internal PointerFollowScanner()
    {
        int n = PtrOffsets.Length;
        _targets     = new nuint[n];
        _prev        = new byte[n][];
        _hasSnapshot = new bool[n];
        for (int i = 0; i < n; i++)
            _prev[i] = new byte[SubScanBytes];
    }

    /// <summary>
    /// Reads and caches the pointer values stored in the session struct at the known offsets.
    /// Call once per hang-out, right after the session pointer is resolved.
    /// </summary>
    internal void Capture(nuint sessionPtr)
    {
        for (int i = 0; i < PtrOffsets.Length; i++)
        {
            _hasSnapshot[i] = false;
            nuint slot = sessionPtr + (nuint)PtrOffsets[i];
            if (!MemoryGuard.IsReadable(slot, sizeof(nuint)))
            {
                _targets[i] = 0;
                continue;
            }
            _targets[i] = *(nuint*)slot;
        }
    }

    /// <summary>
    /// Diffs each pointed-to sub-object against the previous snapshot.
    /// Returns one log string per sub-object that has changed bytes this tick.
    /// On first call per sub-object (no snapshot yet), establishes baseline silently.
    /// </summary>
    internal List<string> Diff()
    {
        var results = new List<string>();

        for (int i = 0; i < PtrOffsets.Length; i++)
        {
            nuint target = _targets[i];
            if (target == 0) continue;
            if (!MemoryGuard.IsReadable(target, SubScanBytes)) continue;

            DiffOne(i, target, results);
        }

        return results;
    }

    private unsafe void DiffOne(int i, nuint target, List<string> results)
    {
        byte* p = (byte*)target;

        if (!_hasSnapshot[i])
        {
            for (int j = 0; j < SubScanBytes; j++) _prev[i][j] = p[j];
            _hasSnapshot[i] = true;
            return;
        }

        StringBuilder? sb = null;
        for (int j = 0; j < SubScanBytes; j++)
        {
            byte cur  = p[j];
            byte prev = _prev[i][j];
            if (cur == prev) continue;

            sb ??= new StringBuilder($"[PtrFollow +0x{PtrOffsets[i]:X2}→0x{target:X}]");
            sb.Append($" +0x{j:X2}:{prev:X2}→{cur:X2}");
            _prev[i][j] = cur;
        }

        if (sb is not null)
            results.Add(sb.ToString());
    }

    /// <summary>Clears all captured pointers and snapshots. Call when hang-out ends.</summary>
    internal void Reset()
    {
        for (int i = 0; i < PtrOffsets.Length; i++)
        {
            _targets[i]     = 0;
            _hasSnapshot[i] = false;
        }
    }
}
