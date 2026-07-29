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
