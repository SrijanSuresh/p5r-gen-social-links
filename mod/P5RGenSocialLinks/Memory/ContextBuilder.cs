namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Builds an LLM context string from a SocialLinkSnapshot.
///
/// The CMM session struct holds no pointer to the actual dialogue text —
/// that lives in P5R's separate script-engine memory. We instead derive
/// context from the four confirmed struct fields (confidantId, rank,
/// sceneNumber, dialogueIndex), which uniquely identify the scripted line
/// in P5R's flowscript database.
/// </summary>
internal static class ContextBuilder
{
    internal static string Build(SocialLinkSnapshot snap) =>
        $"[Scene {snap.SceneNumber}, Line {snap.DialogueIndex}] " +
        $"Social Link conversation — Confidant #{snap.ConfidantId}, rank {snap.RankLevel}";
}
