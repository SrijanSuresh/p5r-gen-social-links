using System;
using System.Runtime.InteropServices;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Locates the BF script text pool in P5R's heap at runtime.
///
/// Three-phase strategy anchored to the session struct (not the CE counter address):
///   Phase 1 — One-level pointer traversal: reads 8-byte-aligned words from the session
///              struct, filters to heap VA range, probes each for the pool fingerprint.
///   Phase 2 — Two-level pointer traversal: for each heap pointer found in Phase 1,
///              scans the first 256 bytes of its target for a second tier of heap
///              pointers, probing each.
///   Phase 3 — Bidirectional heap scan: VirtualQuery walk ±128 MB / ±32 MB from the
///              session struct as a fallback when no pointer path leads to the pool.
///
/// Text pool fingerprint: ≥4 consecutive null-terminated strings where each string has
/// ≥3 printable bytes (0x20–0x7E) and ≥50 % of non-null bytes are printable.
/// Control bytes below 0x20 are tolerated as BF escape codes (pause, speaker, color).
///
/// Call Find() inside OnCmmExecEvent (not just at session-detection time) so the pool
/// is queried after the BF interpreter has decompressed the scene script into heap memory.
/// </summary>
internal static class DialogueTextPoolFinder
{
    private static readonly nuint HeapVaLow  = unchecked((nuint)0x0000_0001_0000_0000UL);
    private static readonly nuint HeapVaHigh = unchecked((nuint)0x0007_FFFF_FFFF_0000UL);

    private const int MinPoolStrings      = 4;   // Phases 1 & 2 (targeted pointer follow)
    private const int MinPoolStringsPhase3 = 8;  // Phase 3 (heap scan) — higher bar to avoid false positives
    private const int MinPrintableChars   = 3;   // printable bytes required per string
    private const int MaxStrLen           = 400;
    private const int ProbeBytes          = 16384; // bytes to probe per candidate
    private const int SessionScanBytes    = 512;   // bytes of session struct to walk
    private const int Depth2ScanBytes     = 256;   // bytes of each Phase-1 target to walk

    [DllImport("kernel32.dll")]
    private static extern nuint VirtualQuery(
        nuint lpAddress, out MBI lpBuffer, nuint dwLength);

    [StructLayout(LayoutKind.Sequential)]
    private struct MBI
    {
        public nuint BaseAddress;
        public nuint AllocationBase;
        public uint  AllocationProtect;
        public nuint RegionSize;
        public uint  State;
        public uint  Protect;
        public uint  Type;
    }

    private const uint MEM_COMMIT    = 0x1000;
    private const uint MEM_PRIVATE   = 0x20000;
    private const uint PAGE_NOACCESS = 0x01;
    private const uint PAGE_GUARD    = 0x100;

    /// <summary>
    /// Finds the text pool anchored to the current session struct.
    /// Returns 0 if not found. Safe to call repeatedly on each CmmExecEvent fire.
    /// </summary>
    internal static nuint Find(nuint sessionBase, Action<string>? log = null)
    {
        nuint phase1 = ProbeSessionPointers(sessionBase, depth: 1, log);
        if (phase1 != 0) return phase1;

        nuint phase2 = ProbeSessionPointers(sessionBase, depth: 2, log);
        if (phase2 != 0) return phase2;

        log?.Invoke("[TextPoolFinder] Pointer traversal found nothing — trying heap scan.");
        nuint phase3 = HeapScanBidirectional(sessionBase, log);
        if (phase3 != 0) return phase3;

        // Emit a pointer map so the next game session can be inspected manually.
        DiagnoseSessionPointers(sessionBase, log);
        log?.Invoke("[TextPoolFinder] No pool found — write-back disabled for this session.");
        return 0;
    }

    // ── Phase 1 & 2: pointer traversal from session struct ───────────────

    private static unsafe nuint ProbeSessionPointers(nuint sessionBase, int depth, Action<string>? log)
    {
        if (!MemoryGuard.IsReadable(sessionBase, SessionScanBytes)) return 0;

        byte* p = (byte*)sessionBase;

        for (int offset = 0; offset + 8 <= SessionScanBytes; offset += 8)
        {
            nuint candidate = *(nuint*)(p + offset);
            if (candidate < HeapVaLow || candidate > HeapVaHigh) continue;
            if (!MemoryGuard.IsReadable(candidate, ProbeBytes)) continue;

            if (depth == 1)
            {
                int count = CountPoolStrings(candidate, ProbeBytes);
                if (count >= MinPoolStrings)
                {
                    log?.Invoke(
                        $"[TextPoolFinder] Phase1: session+0x{offset:X3} → 0x{candidate:X} ({count} strings).");
                    return candidate;
                }
            }
            else
            {
                // Depth 2: scan the first Depth2ScanBytes of each Phase-1 target.
                if (!MemoryGuard.IsReadable(candidate, Depth2ScanBytes)) continue;
                byte* q = (byte*)candidate;

                for (int off2 = 0; off2 + 8 <= Depth2ScanBytes; off2 += 8)
                {
                    nuint cand2 = *(nuint*)(q + off2);
                    if (cand2 < HeapVaLow || cand2 > HeapVaHigh) continue;
                    if (!MemoryGuard.IsReadable(cand2, ProbeBytes)) continue;

                    int count = CountPoolStrings(cand2, ProbeBytes);
                    if (count >= MinPoolStrings)
                    {
                        log?.Invoke(
                            $"[TextPoolFinder] Phase2: session+0x{offset:X3}→+0x{off2:X3} → 0x{cand2:X} ({count} strings).");
                        return cand2;
                    }
                }
            }
        }
        return 0;
    }

    // ── Phase 3: bidirectional heap scan ─────────────────────────────────

    private static nuint HeapScanBidirectional(nuint sessionBase, Action<string>? log)
    {
        nuint forward = HeapScanRange(sessionBase, sessionBase + 0x8000000, log);
        if (forward != 0) return forward;

        nuint bwStart = sessionBase > 0x2000000 ? sessionBase - 0x2000000 : 0;
        return HeapScanRange(bwStart, sessionBase, log);
    }

    private static nuint HeapScanRange(nuint start, nuint end, Action<string>? log)
    {
        nuint addr = start;
        while (addr < end)
        {
            nuint r = VirtualQuery(addr, out MBI mbi, (nuint)Marshal.SizeOf<MBI>());
            if (r == 0) break;

            nuint regionEnd = mbi.BaseAddress + mbi.RegionSize;

            if (mbi.State == MEM_COMMIT &&
                mbi.Type  == MEM_PRIVATE &&
                (mbi.Protect & PAGE_NOACCESS) == 0 &&
                (mbi.Protect & PAGE_GUARD)    == 0 &&
                mbi.RegionSize >= 1024)
            {
                int probeLen = (int)Math.Min(mbi.RegionSize, (nuint)ProbeBytes);
                int count = CountPoolStrings(mbi.BaseAddress, probeLen);
                if (count >= MinPoolStringsPhase3)
                {
                    log?.Invoke(
                        $"[TextPoolFinder] Phase3: 0x{mbi.BaseAddress:X} ({count} strings, sz=0x{mbi.RegionSize:X}).");
                    return mbi.BaseAddress;
                }
            }

            addr = regionEnd;
        }
        return 0;
    }

    // ── Fingerprint ───────────────────────────────────────────────────────

    /// <summary>
    /// Counts consecutive valid strings starting at <paramref name="addr"/>.
    /// A string is valid if it is null-terminated, has ≥MinPrintableChars printable bytes
    /// (0x20–0x7E), and at least 50 % of its non-null bytes are printable.
    /// Bytes 0x01–0x1F are tolerated as BF escape codes.
    /// </summary>
    private static unsafe int CountPoolStrings(nuint addr, int maxBytes)
    {
        byte* p     = (byte*)addr;
        int   pos   = 0;
        int   count = 0;

        while (pos < maxBytes - MinPrintableChars)
        {
            int printable = 0;
            int total     = 0;
            int strEnd    = -1;

            int limit = Math.Min(pos + MaxStrLen, maxBytes);
            for (int i = pos; i < limit; i++)
            {
                byte b = p[i];
                if (b == 0) { strEnd = i; break; }
                total++;
                if (b >= 0x20 && b <= 0x7E) printable++;
            }

            if (strEnd < 0)                          break; // no null terminator
            if (printable < MinPrintableChars)        break; // too few printable chars
            if (total > 0 && printable * 2 < total)  break; // below 50 % printable

            count++;
            pos = strEnd + 1;
        }

        return count;
    }

    // ── Diagnostic: pointer map when all phases fail ──────────────────────

    private static unsafe void DiagnoseSessionPointers(nuint sessionBase, Action<string>? log)
    {
        if (log is null) return;
        if (!MemoryGuard.IsReadable(sessionBase, SessionScanBytes)) return;

        log($"[TextPoolFinder] Diag session=0x{sessionBase:X} — heap pointers:");

        byte* p = (byte*)sessionBase;
        for (int offset = 0; offset + 8 <= SessionScanBytes; offset += 8)
        {
            nuint candidate = *(nuint*)(p + offset);
            if (candidate < HeapVaLow || candidate > HeapVaHigh) continue;
            if (!MemoryGuard.IsReadable(candidate, 64)) continue;

            byte* t = (byte*)candidate;
            int printable = 0;
            for (int i = 0; i < 64; i++)
                if (t[i] >= 0x20 && t[i] <= 0x7E) printable++;
            int pct = (printable * 100) / 64;

            var sb = new System.Text.StringBuilder(32);
            for (int i = 0; i < 64 && sb.Length < 32; i++)
            {
                byte b = t[i];
                sb.Append(b >= 0x20 && b <= 0x7E ? (char)b : '.');
            }

            log($"[TextPoolFinder] Diag  +0x{offset:X3} → 0x{candidate:X} printable={pct}% \"{sb}\"");
        }
    }
}
