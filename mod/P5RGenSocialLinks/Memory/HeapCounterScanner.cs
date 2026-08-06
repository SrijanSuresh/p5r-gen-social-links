using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Replaces the manual Cheat Engine workflow for finding the dialogue line counter.
///
/// At hang-out start it snapshots every writable byte in the low-address heap
/// (0x10000 – 0x20000000). When the hang-out ends it compares the live bytes
/// to the snapshot and reports every address whose value increased. The counter
/// object was at ~0x6FFC10 in the original CE session — a low-address global
/// allocation. This scanner finds whatever address it lives at in the current session.
///
/// Once the counter address is confirmed (one or two sessions), hard-code it in
/// P5ROffsets and remove this scanner.
/// </summary>
internal sealed unsafe class HeapCounterScanner
{
    private const nuint ScanLow          = 0x10000;
    private const nuint ScanHigh         = 0x20000000;   // 512 MB ceiling
    private const int   MaxBytesPerRegion = 16 * 1024 * 1024;  // 16 MB cap per region
    private const int   MaxTotalBytes     = 64 * 1024 * 1024;  // 64 MB total snapshot cap

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
    /// Snapshots all writable pages in the low heap. Call at hang-out start.
    /// Safe to call from the poll thread.
    /// </summary>
    internal void TakeSnapshot(Action<string> log)
    {
        _regions.Clear();
        nuint addr      = ScanLow;
        int   totalSnap = 0;

        while (addr < ScanHigh && totalSnap < MaxTotalBytes)
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
                nuint start     = mbi.BaseAddress > ScanLow  ? mbi.BaseAddress : ScanLow;
                nuint end       = regionEnd        < ScanHigh ? regionEnd       : ScanHigh;
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

        log($"[HeapScan] Snapshot done: {_regions.Count} regions, {totalSnap / 1024} KB");
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
                // Re-check readability per page: pages may have been freed since snapshot.
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

        hits.Sort((a, b) => b.Delta.CompareTo(a.Delta));

        var lines = new List<string>(Math.Min(hits.Count, maxResults));
        foreach (var h in hits)
        {
            if (lines.Count >= maxResults) break;
            lines.Add($"0x{h.Addr:X}  {h.Old} → {h.New}  (+{h.Delta})");
        }
        return lines;
    }

    internal void Clear() => _regions.Clear();
}
