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
        string name  = ConfidantNames.Resolve(snap.ConfidantId);
        string scene = SceneHints.Describe(snap.ConfidantId, snap.SceneNumber);
        return $"Hang-out with {name} (rank {snap.RankLevel}/10) at {scene}. " +
               $"This is a Social Link conversation — {name} and the protagonist are spending time together.";
    }
}
