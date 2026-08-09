using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Replaces the manual Cheat Engine workflow for finding the dialogue line counter.
///
/// At hang-out start it snapshots every writable byte near the session struct in the
/// high heap (pivot ± 256 MB). When the hang-out ends it compares the live bytes
/// to the snapshot and reports every address whose value increased by a small amount —
/// consistent with a per-line dialogue counter.
///
/// Once the counter address is confirmed (one or two sessions), hard-code it in
/// P5ROffsets and remove this scanner.
/// </summary>
internal sealed unsafe class HeapCounterScanner
{
    // ±256 MB around the session struct pivot — covers the primary game heap arena.
    private const nuint ScanRadius       = 256 * 1024 * 1024;
    private const nuint ScanFloor        = 0x10000;         // never go below user-mode start
    private const int   MaxBytesPerRegion = 16 * 1024 * 1024;
    private const int   MaxTotalBytes     = 64 * 1024 * 1024;

    [DllImport("kernel32.dll")]
    private static extern nint VirtualQuery(nuint address, out MBI mbi, nuint dwLength);

    [StructLayout(LayoutKind.Sequential)]
    private struct MBI
    {
        public nuint BaseAddress;
        public nuint AllocationBase;
        public uint  AllocationProtect;
        public uint  _pad;
        public nuint RegionSize;
        public uint  State;
        public uint  Protect;
        public uint  Type;
    }

    private const uint MEM_COMMIT     = 0x1000;
    private const uint PAGE_READWRITE = 0x04;
    private const uint PAGE_WRITECOPY = 0x08;
    private const uint PAGE_NOACCESS  = 0x01;
    private const uint PAGE_GUARD     = 0x100;

    private readonly List<(nuint Base, byte[] Data)> _regions = new();

    /// <summary>
    /// Snapshots all writable pages within ±256 MB of the session struct.
    /// Call at hang-out start, passing the resolved session struct address.
    /// </summary>
    internal void TakeSnapshot(nuint pivotAddr, Action<string> log)
    {
        _regions.Clear();
        nuint scanStart = pivotAddr > ScanRadius ? pivotAddr - ScanRadius : ScanFloor;
        nuint scanEnd   = pivotAddr + ScanRadius;
        nuint addr      = scanStart < ScanFloor ? ScanFloor : scanStart;
        int   totalSnap = 0;

        while (addr < scanEnd && totalSnap < MaxTotalBytes)
        {
            if (VirtualQuery(addr, out MBI mbi, (nuint)Marshal.SizeOf<MBI>()) == 0) break;

            nuint regionEnd = mbi.BaseAddress + mbi.RegionSize;

            bool writable = mbi.State == MEM_COMMIT
                         && (mbi.Protect & PAGE_NOACCESS) == 0
                         && (mbi.Protect & PAGE_GUARD)    == 0
                         && ((mbi.Protect & PAGE_READWRITE) != 0
                             || (mbi.Protect & PAGE_WRITECOPY) != 0);

            if (writable)
            {
                nuint start     = mbi.BaseAddress > scanStart ? mbi.BaseAddress : scanStart;
                nuint end       = regionEnd        < scanEnd   ? regionEnd       : scanEnd;
                nuint rawBytes  = end > start ? end - start : 0;
                int   snapBytes = rawBytes > (nuint)MaxBytesPerRegion
                                  ? MaxBytesPerRegion
                                  : (int)rawBytes;

                if (snapBytes > 0)
                {
                    var buf = new byte[snapBytes];
                    Marshal.Copy((nint)start, buf, 0, snapBytes);
                    _regions.Add((start, buf));
                    totalSnap += snapBytes;
                }
            }

            if (regionEnd <= addr) break;
            addr = regionEnd;
        }

        log($"[HeapScan] Snapshot done: {_regions.Count} regions, {totalSnap / 1024} KB (pivot 0x{pivotAddr:X} ±256MB)");
    }

    /// <summary>
    /// Compares current memory to the snapshot.
    /// Returns up to maxResults addresses where the byte value increased.
    /// Sorted by delta descending so the biggest movers appear first.
    /// Call at hang-out end, after the player has advanced many dialogue lines.
    ///
    /// Reads page-by-page (4 KB chunks) and re-checks readability before each
    /// page — pages that were freed since the snapshot are skipped safely.
    /// </summary>
    internal List<string> FindIncreased(int minDelta = 1, int maxDelta = 255, int maxResults = 30)
    {
        var hits = new List<(nuint Addr, int Old, int New, int Delta)>();

        foreach (var (baseAddr, snap) in _regions)
        {
            byte* p   = (byte*)baseAddr;
            int   len = snap.Length;

            for (int pageStart = 0; pageStart < len; pageStart += 4096)
            {
                nuint pageAddr = baseAddr + (nuint)pageStart;
                if (!MemoryGuard.IsReadable(pageAddr, 1)) continue;

                int pageEnd = pageStart + 4096 < len ? pageStart + 4096 : len;
                for (int i = pageStart; i < pageEnd; i++)
                {
                    int delta = p[i] - snap[i];
                    if (delta >= minDelta && delta <= maxDelta)
                        hits.Add((baseAddr + (nuint)i, snap[i], p[i], delta));
                }
            }
        }

        // Remove timer arrays: pairs of hits within 48 bytes at 4-byte aligned stride
        // are co-members of a game timer struct array and are filtered out.
        // The dialogue counter is a single isolated byte, never part of such arrays.
        var arrayMembers = new System.Collections.Generic.HashSet<nuint>();
        for (int a = 0; a < hits.Count; a++)
        {
            for (int b = a + 1; b < hits.Count; b++)
            {
                ulong diff = hits[b].Addr > hits[a].Addr
                             ? (ulong)(hits[b].Addr - hits[a].Addr)
                             : (ulong)(hits[a].Addr - hits[b].Addr);
                if (diff <= 48 && diff % 4 == 0)
                {
                    arrayMembers.Add(hits[a].Addr);
                    arrayMembers.Add(hits[b].Addr);
                }
            }
        }
        int arrayCount = 0;
        if (arrayMembers.Count > 0)
        {
            arrayCount = hits.RemoveAll(h => arrayMembers.Contains(h.Addr));
        }

        hits.Sort((a, b) => b.Delta.CompareTo(a.Delta));

        var lines = new List<string>(Math.Min(hits.Count, maxResults) + 1);
        if (arrayCount > 0)
            lines.Add($"(filtered {arrayCount} timer-array bytes; {hits.Count} isolated left)");
        foreach (var h in hits)
        {
            if (lines.Count >= maxResults + 1) break;
            lines.Add($"0x{h.Addr:X}  {h.Old} → {h.New}  (+{h.Delta})");
        }
        return lines;
    }

    internal void Clear() => _regions.Clear();
}
