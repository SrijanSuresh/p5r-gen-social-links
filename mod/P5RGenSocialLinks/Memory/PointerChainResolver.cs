using System;
using System.Diagnostics;
using Reloaded.Memory.Sources;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Resolves the multi-level pointer chain that leads to the active
/// SocialLinkSession struct in P5R's heap. All offsets are relative to
/// the p5r.exe module base so ASLR is handled transparently.
/// </summary>
internal sealed class PointerChainResolver
{
    private readonly IMemory _memory;
    private readonly nuint   _moduleBase;

    // Pointer chain: [moduleBase + SL_STATIC_PTR] -> +0x18 -> +0x08 -> SocialLinkSession*
    // These offsets are PLACEHOLDERS — verify with Cheat Engine + Ghidra against your build.
    private static readonly int[] Chain = { 0x18, 0x08 };

    internal PointerChainResolver(IMemory memory, nuint moduleBase)
    {
        _memory     = memory;
        _moduleBase = moduleBase;
    }

    /// <summary>
    /// Attempts to walk the pointer chain. Returns false (and sets result = 0)
    /// if any intermediate pointer is null, keeping the process stable.
    /// </summary>
    internal bool TryResolve(out nuint result)
    {
        result = 0;

        // Step 1: read the root static pointer (module-relative)
        nuint address = _moduleBase + (nuint)P5ROffsets.SL_STATIC_PTR;
        if (!_memory.Read(address, out nuint current) || current == 0)
            return false;

        // Step 2: walk each offset in the chain
        foreach (int offset in Chain)
        {
            nuint next = current + (nuint)offset;
            if (!_memory.Read(next, out current) || current == 0)
                return false;
        }

        result = current;
        return true;
    }
}