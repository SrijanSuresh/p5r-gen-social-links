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
    /// Reads a snapshot directly from a pointer received via the hook argument.
    /// Bypasses the pointer chain entirely — use this inside OnConversationInit
    /// where sessionPtr is already in hand.
    /// </summary>
    internal static unsafe SocialLinkSnapshot? TryReadFromPtr(nuint sessionPtr)
    {
        if (!MemoryGuard.IsReadable(sessionPtr, 0x10)) return null;

        int confidantId   = *(int*)(sessionPtr + (nuint)P5ROffsets.CONFIDANT_ID);
        int rankLevel     = *(int*)(sessionPtr + (nuint)P5ROffsets.RANK_LEVEL);
        int dialogueIndex = *(int*)(sessionPtr + (nuint)P5ROffsets.DIALOGUE_INDEX);

        return new SocialLinkSnapshot(confidantId, rankLevel, dialogueIndex, sessionPtr);
    }

    /// <summary>
    /// Dumps 128 bytes around sessionPtr as hex so we can identify real struct offsets.
    /// Only used during offset discovery — remove once offsets are confirmed.
    /// </summary>
    internal static unsafe string HexDump(nuint ptr, int bytes = 128)
    {
        if (!MemoryGuard.IsReadable(ptr, bytes))
            return $" [unreadable @ 0x{ptr:X}]";

        var sb = new System.Text.StringBuilder();
        byte* p = (byte*)ptr;
        for (int i = 0; i < bytes; i++)
        {
            if (i % 16 == 0) sb.Append($"\n+{i:X3}: ");
            sb.Append($"{p[i]:X2} ");
        }
        return sb.ToString();
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