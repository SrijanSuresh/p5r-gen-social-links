using System.Runtime.InteropServices;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Wraps VirtualQuery so unsafe pointer reads can be guarded without
/// catching corrupted-state exceptions (which .NET 8 does not allow).
/// </summary>
internal static class MemoryGuard
{
    [DllImport("kernel32.dll")]
    private static extern nint VirtualQuery(
        nuint lpAddress,
        out MEMORY_BASIC_INFORMATION lpBuffer,
        nuint dwLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    internal static extern bool VirtualProtect(
        nuint lpAddress,
        nuint dwSize,
        uint  flNewProtect,
        out uint lpflOldProtect);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool ReadProcessMemory(
        nint  hProcess,
        nuint lpBaseAddress,
        byte[] lpBuffer,
        nuint nSize,
        out nuint lpNumberOfBytesRead);

    [DllImport("kernel32.dll")]
    private static extern nint GetCurrentProcess();

    /// <summary>
    /// Copies game memory into a managed buffer, returning false instead of faulting.
    ///
    /// IsReadable followed by a raw pointer walk is a time-of-check/time-of-use race: the
    /// game frees and remaps heap on its own threads, so a region validated at the start
    /// of a multi-megabyte scan can vanish partway through. That raises an
    /// AccessViolationException, which .NET 8 treats as a corrupted-state exception —
    /// uncatchable, and fatal to the whole process. Routing bulk scans through
    /// ReadProcessMemory turns that crash into a false return.
    /// </summary>
    internal static bool TryRead(nuint addr, byte[] buffer, int len)
    {
        if (addr == 0 || len <= 0 || len > buffer.Length) return false;
        return ReadProcessMemory(GetCurrentProcess(), addr, buffer, (nuint)len, out nuint got)
               && (int)got == len;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MEMORY_BASIC_INFORMATION
    {
        public nuint BaseAddress;
        public nuint AllocationBase;
        public uint  AllocationProtect;
        public nuint RegionSize;
        public uint  State;
        public uint  Protect;
        public uint  Type;
    }

    private const uint MEM_COMMIT              = 0x1000;
    private const uint PAGE_NOACCESS           = 0x01;
    private const uint PAGE_READWRITE          = 0x04;
    private const uint PAGE_WRITECOPY          = 0x08;
    private const uint PAGE_EXECUTE_READWRITE  = 0x40;
    private const uint PAGE_EXECUTE_WRITECOPY  = 0x80;
    private const uint PAGE_GUARD              = 0x100;

    private static readonly uint WritableMask =
        PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;

    /// <summary>
    /// Returns the VirtualQuery descriptor for the region that contains
    /// <paramref name="addr"/>. Used by callers that need to walk the
    /// address space region-by-region rather than check one range.
    /// </summary>
    internal static (bool ok, nuint regionBase, nuint regionSize, uint state, uint protect)
        QueryRegion(nuint addr)
    {
        nint result = VirtualQuery(addr, out var mbi,
                          (nuint)Marshal.SizeOf<MEMORY_BASIC_INFORMATION>());
        if (result == 0) return (false, 0, 0, 0, 0);
        return (true, mbi.BaseAddress, mbi.RegionSize, mbi.State, mbi.Protect);
    }

    /// <summary>
    /// Returns true only if [addr, addr+size) is fully within a committed,
    /// readable page — safe to dereference without an AV.
    /// </summary>
    internal static bool IsReadable(nuint addr, int size)
    {
        if (addr == 0) return false;
        nint queryResult = VirtualQuery(addr, out var mbi, (nuint)Marshal.SizeOf<MEMORY_BASIC_INFORMATION>());
        if (queryResult == 0) return false;
        if (mbi.State != MEM_COMMIT) return false;
        if ((mbi.Protect & PAGE_NOACCESS) != 0) return false;
        if ((mbi.Protect & PAGE_GUARD) != 0) return false;
        return (addr - mbi.BaseAddress) + (nuint)size <= mbi.RegionSize;
    }

    /// <summary>
    /// Returns true only if [addr, addr+size) is fully within a committed,
    /// writable page — safe to write without an AV.
    /// </summary>
    internal static bool IsWritable(nuint addr, int size)
    {
        if (addr == 0) return false;
        nint queryResult = VirtualQuery(addr, out var mbi, (nuint)Marshal.SizeOf<MEMORY_BASIC_INFORMATION>());
        if (queryResult == 0) return false;
        if (mbi.State != MEM_COMMIT) return false;
        if ((mbi.Protect & PAGE_NOACCESS) != 0) return false;
        if ((mbi.Protect & PAGE_GUARD) != 0) return false;
        if ((mbi.Protect & WritableMask) == 0) return false;
        return (addr - mbi.BaseAddress) + (nuint)size <= mbi.RegionSize;
    }
}
