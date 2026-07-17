using System;
using Reloaded.Memory;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Resolves the multi-level pointer chain that leads to the active
/// SocialLinkSession struct in P5R''s heap. All offsets are relative to
/// the p5r.exe module base so ASLR is handled transparently.
/// </summary>
internal sealed class PointerChainResolver
{
    private readonly nuint _moduleBase;

    // Pointer chain: [moduleBase + SL_STATIC_PTR] -> +0x18 -> +0x08 -> SocialLinkSession*
    // These offsets are PLACEHOLDERS — verify with Cheat Engine + Ghidra against your build.
    private static readonly int[] Chain = { 0x18, 0x08 };

    internal PointerChainResolver(nuint moduleBase)
    {
        _moduleBase = moduleBase;
    }

    /// <summary>
    /// Walks the pointer chain. Returns false if any intermediate pointer is null or zero,
    /// keeping the process stable when no Social Link session is active.
    /// </summary>
    internal unsafe bool TryResolve(out nuint result)
    {
        result = 0;

        // Step 1: read the root static pointer (module-relative, always a valid .data address)
        nuint address = _moduleBase + (nuint)P5ROffsets.SL_STATIC_PTR;
        nuint current = *(nuint*)address;
        if (current == 0)
            return false;

        // Step 2: walk each heap-level offset, null-guarding before every dereference
        foreach (int offset in Chain)
        {
            current = *(nuint*)(current + (nuint)offset);
            if (current == 0)
                return false;
        }

        result = current;
        return true;
    }
}