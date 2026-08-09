using System;
using System.Collections.Generic;
using System.Text;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// On every poll tick, reads the session-struct pointer slots at +0xE0/+0xE8/+0xF0 and
/// diffs 256 bytes at each pointed-to address. Unlike Capture (one-shot at session start),
/// Update runs every tick so it catches pointers that appear transiently mid-session.
///
/// Once a valid user-mode heap address is captured for a slot, it stays locked in for the
/// rest of the hang-out — we keep scanning even after the slot reverts to garbage, because
/// the sub-object itself persists on the heap.
/// </summary>
internal sealed unsafe class PointerFollowScanner
{
    // Offsets confirmed in Takemi Session 40 StructDiff: pointer-like values flashed here.
    private static readonly int[] PtrOffsets = { 0xE0, 0xE8, 0xF0 };

    // User-mode heap range. DLLs load at 0x7FF8_0000_0000 and above; exclude them
    // so DLL vtable pointers can't evict stable heap targets and reset baselines.
    private const nuint UserModeMin  = 0x10000;
    private static readonly nuint HeapAddressMax =
        unchecked((nuint)0x0000_7F00_0000_0000ul);

    // 512 bytes covers sub-object headers (~200 B) + dialogue line counter region.
    private const int SubScanBytes = 512;

    private readonly nuint[] _targets;
    private readonly byte[][] _prev;
    private readonly byte[][] _baseline;
    private readonly bool[]   _hasSnapshot;
    private readonly bool[]   _hasBaseline;

    internal PointerFollowScanner()
    {
        int n = PtrOffsets.Length;
        _targets     = new nuint[n];
        _prev        = new byte[n][];
        _baseline    = new byte[n][];
        _hasSnapshot = new bool[n];
        _hasBaseline = new bool[n];
        for (int i = 0; i < n; i++)
        {
            _prev[i]     = new byte[SubScanBytes];
            _baseline[i] = new byte[SubScanBytes];
        }
    }

    /// <summary>
    /// Reads the pointer slots in the session struct. If a slot holds a valid user-mode
    /// address that differs from the current target, the target is updated and logged.
    /// Call on every poll tick so transient pointers are not missed.
    /// </summary>
    internal void Update(nuint sessionPtr, Action<string>? log = null)
    {
        for (int i = 0; i < PtrOffsets.Length; i++)
        {
            nuint slot = sessionPtr + (nuint)PtrOffsets[i];
            if (!MemoryGuard.IsReadable(slot, sizeof(nuint))) continue;

            nuint candidate = *(nuint*)slot;

            // Ignore null, kernel-space, and DLL-image addresses.
            // HeapAddressMax excludes the 0x7FF8... DLL range so vtable pointers
            // can't overwrite a stable heap target and reset the cumulative baseline.
            if (candidate < UserModeMin || candidate >= HeapAddressMax) continue;
            if (candidate == _targets[i]) continue;

            _targets[i]     = candidate;
            _hasSnapshot[i] = false;
            log?.Invoke($"[PtrFollow] +0x{PtrOffsets[i]:X2} captured target 0x{candidate:X}");
        }
    }

    /// <summary>
    /// Diffs each captured sub-object against its previous snapshot.
    /// Returns one log line per sub-object that has bytes changed this tick.
    /// On first Diff() after a new target is captured, silently establishes baseline.
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

    private void DiffOne(int i, nuint target, List<string> results)
    {
        byte* p = (byte*)target;

        if (!_hasSnapshot[i])
        {
            for (int j = 0; j < SubScanBytes; j++)
            {
                _prev[i][j]     = p[j];
                _baseline[i][j] = p[j];
            }
            _hasSnapshot[i] = true;
            _hasBaseline[i] = true;
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

    /// <summary>
    /// Compares current sub-object bytes to the capture-time baseline.
    /// Reports every byte that increased since the baseline was established.
    /// Use at mid-session and session-end to detect slowly-incrementing counters.
    /// </summary>
    internal List<string> CumulativeDiff()
    {
        var results = new List<string>();
        for (int i = 0; i < PtrOffsets.Length; i++)
        {
            nuint target = _targets[i];
            if (target == 0 || !_hasBaseline[i]) continue;
            if (!MemoryGuard.IsReadable(target, SubScanBytes)) continue;
            CumulativeDiffOne(i, target, results);
        }
        return results;
    }

    private unsafe void CumulativeDiffOne(int i, nuint target, List<string> results)
    {
        byte* p = (byte*)target;
        StringBuilder? sb = null;

        for (int j = 0; j < SubScanBytes; j++)
        {
            int delta = (int)p[j] - (int)_baseline[i][j];
            if (delta <= 0 || delta > 100) continue;  // only small positive increments

            sb ??= new StringBuilder($"[PtrFollow cumul +0x{PtrOffsets[i]:X2}→0x{target:X}]");
            sb.Append($" +0x{j:X2}:{_baseline[i][j]:X2}→{p[j]:X2}(+{delta})");
        }

        if (sb is not null)
            results.Add(sb.ToString());
    }

    /// <summary>Clears all targets, snapshots, and baselines. Call when hang-out ends.</summary>
    internal void Reset()
    {
        for (int i = 0; i < PtrOffsets.Length; i++)
        {
            _targets[i]     = 0;
            _hasSnapshot[i] = false;
            _hasBaseline[i] = false;
        }
    }
}
