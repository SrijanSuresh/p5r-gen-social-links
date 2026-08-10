using System;
using System.Runtime.InteropServices;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Locates the BF script text pool in P5R's heap at runtime — no Cheat Engine required.
///
/// Two-phase strategy:
///   Phase 1 — Counter struct probe: reads 256 bytes from CMM_LINE_COUNTER_STRUCT_BASE
///              and extracts 8-byte words in the heap VA range as pointer candidates.
///              Each candidate is probed for the text pool fingerprint.
///   Phase 2 — Bidirectional heap scan: walks VirtualQuery-enumerated committed private
///              pages both forward and backward from the session struct address.
///
/// Text pool fingerprint: ≥5 consecutive null-terminated strings in either UTF-8/ASCII
/// or UTF-16LE encoding, each 10–300 chars. Both encodings are probed because P5R PC
/// may store English dialogue as either ASCII bytes or wide chars.
///
/// Timing: call Find() when the first LineCounterMonitor tick fires — by that point the
/// dialogue box has rendered and the BF engine has definitely loaded the text pool into
/// heap memory. Calling at raw hang-out start may miss the pool if P5R loads it lazily.
/// </summary>
internal static class DialogueTextPoolFinder
{
    // Heap VA range for a 64-bit P5R process (filters out module image sections).
    // static readonly because nuint cannot hold ulong literals as const in C#.
    private static readonly nuint HeapVaLow  = unchecked((nuint)0x0000_0001_0000_0000UL);
    private static readonly nuint HeapVaHigh = unchecked((nuint)0x0007_FFFF_FFFF_0000UL);

    // Pool fingerprint thresholds.
    private const int MinPoolStrings = 5;
    private const int MinStrLen      = 10;   // chars
    private const int MaxStrLen      = 300;  // chars
    private const int ProbeBytes     = 8192; // bytes per region probe

    [DllImport("kernel32.dll")]
    private static extern nuint VirtualQuery(
        nuint lpAddress, out MBI lpBuffer, nuint dwLength);

    [StructLayout(LayoutKind.Sequential)]
    private struct MBI
    {
        public nuint BaseAddress;
        public nuint AllocationBase;
        public uint  AllocationProtect;
        // 4-byte implicit alignment pad (CLR matches Windows 64-bit MEMORY_BASIC_INFORMATION)
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
    /// Finds the text pool anchored to the current session. Returns 0 if not found.
    /// Safe to call repeatedly — logs a one-line result each attempt.
    /// </summary>
    internal static nuint Find(nuint sessionBase, Action<string>? log = null)
    {
        nuint fromCounter = ProbeCounterStruct(log);
        if (fromCounter != 0) return fromCounter;

        log?.Invoke("[TextPoolFinder] Counter struct probe found nothing — trying heap scan.");
        return HeapScanBidirectional(sessionBase, log);
    }

    // ── Phase 1: counter struct probe ────────────────────────────────────

    private static unsafe nuint ProbeCounterStruct(Action<string>? log)
    {
        nuint structBase = P5ROffsets.CMM_LINE_COUNTER_STRUCT_BASE;
        const int ScanBytes = 256;

        if (!MemoryGuard.IsReadable(structBase, ScanBytes))
        {
            log?.Invoke($"[TextPoolFinder] Counter struct 0x{structBase:X} not readable — skipping Phase 1.");
            return 0;
        }

        byte* p = (byte*)structBase;
        for (int offset = 0; offset + 8 <= ScanBytes; offset += 8)
        {
            nuint candidate = *(nuint*)(p + offset);
            if (candidate < HeapVaLow || candidate > HeapVaHigh) continue;
            if (!MemoryGuard.IsReadable(candidate, ProbeBytes)) continue;

            int count = ProbeCandidate(candidate);
            if (count > 0)
                log?.Invoke($"[TextPoolFinder] CounterStruct+0x{offset:X}: 0x{candidate:X} → {count} strings");

            if (count >= MinPoolStrings)
            {
                log?.Invoke($"[TextPoolFinder] Pool found via counter struct at 0x{candidate:X} ({count} strings).");
                return candidate;
            }
        }
        return 0;
    }

    // ── Phase 2: bidirectional heap scan ─────────────────────────────────

    private static nuint HeapScanBidirectional(nuint sessionBase, Action<string>? log)
    {
        // Scan forward up to 128 MB, then backward up to 32 MB.
        // The text pool has historically been ~7.7 MB forward from the session struct.
        nuint forward = HeapScanRange(sessionBase, sessionBase + 0x8000000, log);
        if (forward != 0) return forward;

        // The VA address space can place pool before the session struct too.
        nuint backward = HeapScanRange(
            sessionBase > 0x2000000 ? sessionBase - 0x2000000 : 0,
            sessionBase,
            log);
        if (backward != 0) return backward;

        log?.Invoke("[TextPoolFinder] No pool found — write-back disabled for this session.");
        return 0;
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
                int count = ProbeCandidate(mbi.BaseAddress);
                if (count >= MinPoolStrings)
                {
                    log?.Invoke($"[TextPoolFinder] Pool found at 0x{mbi.BaseAddress:X} ({count} strings, size=0x{mbi.RegionSize:X}).");
                    return mbi.BaseAddress;
                }
            }

            addr = regionEnd;
        }
        return 0;
    }

    // ── Fingerprint: try both encodings ──────────────────────────────────

    private static int ProbeCandidate(nuint addr)
    {
        int ascii = CountPoolStringsAscii(addr, ProbeBytes);
        if (ascii >= MinPoolStrings) return ascii;

        int wide = CountPoolStringsUtf16(addr, ProbeBytes);
        return wide;
    }

    private static unsafe int CountPoolStringsAscii(nuint addr, int maxBytes)
    {
        byte* p     = (byte*)addr;
        int   pos   = 0;
        int   count = 0;

        while (pos + MinStrLen < maxBytes)
        {
            int strEnd = -1;
            bool valid = true;

            int scanLimit = Math.Min(pos + MaxStrLen + 1, maxBytes);
            for (int i = pos; i < scanLimit; i++)
            {
                byte b = p[i];
                if (b == 0) { strEnd = i; break; }
                if (b < 0x20 || b > 0x7E) { valid = false; break; }
            }

            if (!valid || strEnd < 0 || strEnd - pos < MinStrLen) break;

            count++;
            pos = strEnd + 1;
        }
        return count;
    }

    private static unsafe int CountPoolStringsUtf16(nuint addr, int maxBytes)
    {
        // UTF-16LE: each char is [lo, hi]. For ASCII content: hi = 0x00, lo = 0x20–0x7E.
        // Null terminator: [0x00, 0x00].
        char* p     = (char*)addr;
        int   maxChars = maxBytes / 2;
        int   pos   = 0;
        int   count = 0;

        while (pos + MinStrLen < maxChars)
        {
            int strEnd = -1;
            bool valid = true;

            int scanLimit = Math.Min(pos + MaxStrLen + 1, maxChars);
            for (int i = pos; i < scanLimit; i++)
            {
                char c = p[i];
                if (c == '\0') { strEnd = i; break; }
                // Accept printable ASCII range in wide form.
                if (c < 0x20 || c > 0x7E) { valid = false; break; }
            }

            if (!valid || strEnd < 0 || strEnd - pos < MinStrLen) break;

            count++;
            pos = strEnd + 1;
        }
        return count;
    }
}
