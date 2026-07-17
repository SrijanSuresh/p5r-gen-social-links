using Reloaded.Memory;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// High-level reader: resolves the pointer chain then reads individual
/// fields from the SocialLinkSession struct safely.
/// </summary>
internal sealed class SocialLinkReader
{
    private readonly PointerChainResolver _resolver;

    internal SocialLinkReader(nuint moduleBase)
    {
        _resolver = new PointerChainResolver(moduleBase);
    }

    /// <summary>
    /// Returns null when no Social Link session is active (pointer chain unresolved).
    /// </summary>
    internal unsafe SocialLinkSnapshot? TryReadSnapshot()
    {
        if (!_resolver.TryResolve(out nuint sessionBase))
            return null;

        int confidantId   = *(int*)(sessionBase + (nuint)P5ROffsets.CONFIDANT_ID);
        int rankLevel     = *(int*)(sessionBase + (nuint)P5ROffsets.RANK_LEVEL);
        int dialogueIndex = *(int*)(sessionBase + (nuint)P5ROffsets.DIALOGUE_INDEX);

        return new SocialLinkSnapshot(confidantId, rankLevel, dialogueIndex, sessionBase);
    }
}

/// <summary>Immutable snapshot of one Social Link conversation moment.</summary>
internal sealed record SocialLinkSnapshot(
    int   ConfidantId,
    int   RankLevel,
    int   DialogueIndex,
    nuint SessionBase
);