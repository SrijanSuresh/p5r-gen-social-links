using System;
using Reloaded.Memory.Sources;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// High-level reader: resolves the pointer chain then reads individual
/// fields from the SocialLinkSession struct safely.
/// </summary>
internal sealed class SocialLinkReader
{
    private readonly IMemory              _memory;
    private readonly PointerChainResolver _resolver;

    internal SocialLinkReader(IMemory memory, nuint moduleBase)
    {
        _memory   = memory;
        _resolver = new PointerChainResolver(memory, moduleBase);
    }

    /// <summary>
    /// Returns null when no Social Link session is active (pointer chain unresolved).
    /// </summary>
    internal SocialLinkSnapshot? TryReadSnapshot()
    {
        if (!_resolver.TryResolve(out nuint sessionBase))
            return null;

        // Each field read is guarded — a bad pointer returns false rather than crashing.
        if (!_memory.Read(sessionBase + (nuint)P5ROffsets.CONFIDANT_ID,   out int confidantId))   return null;
        if (!_memory.Read(sessionBase + (nuint)P5ROffsets.RANK_LEVEL,      out int rankLevel))     return null;
        if (!_memory.Read(sessionBase + (nuint)P5ROffsets.DIALOGUE_INDEX,  out int dialogueIndex)) return null;

        return new SocialLinkSnapshot(confidantId, rankLevel, dialogueIndex, sessionBase);
    }
}

/// <summary>Immutable snapshot of one Social Link conversation moment.</summary>
internal sealed record SocialLinkSnapshot(
    int   ConfidantId,
    int   RankLevel,
    int   DialogueIndex,
    nuint SessionBase   // kept so the dialogue injector can write back to the same location
);