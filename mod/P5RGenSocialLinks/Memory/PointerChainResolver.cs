using System;
using Reloaded.Memory;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Resolves the multi-level pointer chain that leads to the active
/// SocialLinkSession struct in P5R's heap. All offsets are relative to
/// the p5r.exe module base so ASLR is handled transparently.
/// </summary>
internal sealed class PointerChainResolver
{
    private readonly nuint           _moduleBase;
    private readonly bool            _verbose;
    private readonly Action<string>? _log;

    // Chain: [moduleBase + SL_STATIC_PTR] -> +CMM_SESSION_OFFSET -> CmmSession*
    // Derived from CMM_EXEC_EVENT: MOV RAX,[CMM_global]; MOV RCX,[RAX+0x48]
    private static readonly int[] Chain = { P5ROffsets.CMM_SESSION_OFFSET };

    internal PointerChainResolver(nuint moduleBase, bool verbose = false, Action<string>? log = null)
    {
        _moduleBase = moduleBase;
        _verbose    = verbose;
        _log        = log;
    }

    /// <summary>
    /// Walks the pointer chain. Returns false if any intermediate pointer is null or unreadable.
    /// When verbose=true (from GenConfig.VerboseChain), logs each step with address and value.
    /// </summary>
    internal unsafe bool TryResolve(out nuint result)
    {
        result = 0;

        // Step 0: read the root static pointer (module-relative)
        nuint address = _moduleBase + (nuint)P5ROffsets.SL_STATIC_PTR;
        if (!MemoryGuard.IsReadable(address, sizeof(nuint)))
        {
            if (_verbose)
                _log?.Invoke($"[P5RGenSocialLinks] ChainStep 0: addr=0x{address:X} UNREADABLE");
            return false;
        }

        nuint current = *(nuint*)address;
        if (_verbose)
            _log?.Invoke($"[P5RGenSocialLinks] ChainStep 0: addr=0x{address:X} → 0x{current:X}");

        if (current == 0)
        {
            if (_verbose)
                _log?.Invoke("[P5RGenSocialLinks] ChainStep 0: null root — CMM not yet initialised");
            return false;
        }

        // Step 1+: walk heap-level offsets, VirtualQuery-guarding before every dereference.
        for (int i = 0; i < Chain.Length; i++)
        {
            nuint next = current + (nuint)Chain[i];
            if (!MemoryGuard.IsReadable(next, sizeof(nuint)))
            {
                if (_verbose)
                    _log?.Invoke($"[P5RGenSocialLinks] ChainStep {i + 1}: addr=0x{next:X} UNREADABLE (offset +0x{Chain[i]:X})");
                return false;
            }

            nuint value = *(nuint*)next;
            if (_verbose)
            {
                bool isLast = i == Chain.Length - 1;
                string suffix = value == 0
                    ? (isLast ? " (null — no active session)" : " (UNEXPECTED NULL)")
                    : string.Empty;
                _log?.Invoke($"[P5RGenSocialLinks] ChainStep {i + 1}: addr=0x{next:X} → 0x{value:X}{suffix}");
            }

            if (value == 0) return false;
            current = value;
        }

        result = current;
        return true;
    }
}
