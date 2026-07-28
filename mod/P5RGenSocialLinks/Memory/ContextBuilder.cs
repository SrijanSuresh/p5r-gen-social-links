namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Builds an LLM context string from a SocialLinkSnapshot.
///
/// The CMM session struct is a controller object — it tracks which hang-out
/// is happening (confidantId, rank, sceneNumber) but not individual dialogue
/// lines. We fire once per new hang-out session and pass scene metadata as
/// the LLM context.
/// </summary>
internal static class ContextBuilder
{
    internal static string Build(SocialLinkSnapshot snap)
    {
        string name = ConfidantNames.Resolve(snap.ConfidantId);
        return $"[Scene {snap.SceneNumber}] Hang-out with {name} (rank {snap.RankLevel}/10). " +
               $"This is a Social Link conversation where {name} is spending time with the protagonist.";
    }
}
