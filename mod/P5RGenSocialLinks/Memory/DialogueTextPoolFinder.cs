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
///   Phase 2 — Forward heap scan: walks VirtualQuery-enumerated committed private pages
///              starting from the session struct address. Slower but covers the case where
///              the counter struct holds no direct pointer to the pool.
///
/// Text pool fingerprint: ≥5 consecutive null-terminated ASCII strings, each 10–300 chars.
/// </summary>
internal static class DialogueTextPoolFinder
{
    // Heap virtual-address range for a 64-bit P5R process.
    // Filters out module image sections (.text, .data, .rdata) and the low 4 GB.
    // static readonly because nuint constants can't hold ulong literals in C#.
    private static readonly nuint HeapVaLow  = unchecked((nuint)0x0000_0001_0000_0000UL);
    private static readonly nuint HeapVaHigh = unchecked((nuint)0x0007_FFFF_FFFF_0000UL);

    // A region must have at least this many consecutive valid strings to qualify.
    private const int MinPoolStrings = 5;
    // Valid dialogue string bounds (chars).
    private const int MinStrLen = 10;
    private const int MaxStrLen = 300;
    // How many bytes to probe inside a candidate region.
    private const int ProbeBytes = 8192;

    [DllImport("kernel32.dll")]
    private static extern nuint VirtualQuery(
        nuint lpAddress, out MBI lpBuffer, nuint dwLength);

    [StructLayout(LayoutKind.Sequential)]
    private struct MBI
    {
        public nuint BaseAddress;
        public nuint AllocationBase;
        public uint  AllocationProtect;
        // 4-byte implicit alignment pad (CLR matches Windows 64-bit layout)
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
    /// Call once at hang-out start after <paramref name="sessionBase"/> is confirmed.
    /// </summary>
    internal static nuint Find(nuint sessionBase, Action<string>? log = null)
    {
        nuint fromCounter = ProbeCounterStruct(log);
        if (fromCounter != 0) return fromCounter;

        log?.Invoke("[TextPoolFinder] Counter struct probe found nothing — falling back to heap scan.");
        return HeapScanForward(sessionBase, log);
    }

    // ── Phase 1: counter struct probe ────────────────────────────────────

    private static unsafe nuint ProbeCounterStruct(Action<string>? log)
    {
        nuint structBase = P5ROffsets.CMM_LINE_COUNTER_STRUCT_BASE;
        const int ScanBytes = 256;

        if (!MemoryGuard.IsReadable(structBase, ScanBytes)) return 0;

        byte* p = (byte*)structBase;
        for (int offset = 0; offset + 8 <= ScanBytes; offset += 8)
        {
            nuint candidate = *(nuint*)(p + offset);
            if (candidate < HeapVaLow || candidate > HeapVaHigh) continue;
            if (!MemoryGuard.IsReadable(candidate, ProbeBytes)) continue;

            int count = CountPoolStrings(candidate, ProbeBytes);
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

    // ── Phase 2: forward heap scan ───────────────────────────────────────

    private static nuint HeapScanForward(nuint sessionBase, Action<string>? log)
    {
        nuint addr  = sessionBase;
        nuint limit = sessionBase + 0x4000000; // 64 MB window

        while (addr < limit)
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
                int probe = (int)Math.Min(mbi.RegionSize, (nuint)ProbeBytes);
                int count = CountPoolStrings(mbi.BaseAddress, probe);
                if (count >= MinPoolStrings)
                {
                    log?.Invoke($"[TextPoolFinder] Pool found via heap scan at 0x{mbi.BaseAddress:X} ({count} strings).");
                    return mbi.BaseAddress;
                }
            }

            addr = regionEnd;
        }

        log?.Invoke("[TextPoolFinder] No pool found — write-back disabled for this session.");
        return 0;
    }

    // ── String pool fingerprint ──────────────────────────────────────────

    private static unsafe int CountPoolStrings(nuint addr, int maxBytes)
    {
        // The region is already confirmed readable by the caller; walk bytes directly.
        byte* p   = (byte*)addr;
        int   pos = 0;
        int   count = 0;

        while (pos + MinStrLen < maxBytes)
        {
            int strEnd = -1;
            bool valid = true;

            int scanLimit = Math.Min(pos + MaxStrLen + 1, maxBytes);
            for (int i = pos; i < scanLimit; i++)
            {
                byte b = p[i];
                if (b == 0)
                {
                    strEnd = i;
                    break;
                }
                // Accept printable ASCII only (English P5R PC uses UTF-8 ≡ ASCII for dialogue).
                if (b < 0x20 || b > 0x7E) { valid = false; break; }
            }

            if (!valid || strEnd < 0) break;

            int strLen = strEnd - pos;
            if (strLen < MinStrLen) break; // string too short — not dialogue

            count++;
            pos = strEnd + 1; // advance past null terminator
        }

        return count;
    }
}
