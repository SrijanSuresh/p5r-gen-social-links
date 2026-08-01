using System.Collections.Generic;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Maps known scene numbers (from the session struct +0x0C) to human-readable
/// descriptions for richer LLM context. Populated incrementally from in-game
/// observations. Unknown scenes fall back to "Social Link hang-out scene N".
/// </summary>
internal static class SceneHints
{
    // Key: (confidantId, sceneNumber) — scene numbers are not globally unique.
    // Value: brief setting description for the LLM prompt.
    private static readonly Dictionary<(int, int), string> _hints = new()
    {
        // Ryuji Sakamoto (ID=8) — confirmed from gym hang-out log
        { (8, 51), "gym training session at Tae-ken Sports Club" },
    };

    internal static string Describe(int confidantId, int sceneNumber)
    {
        if (_hints.TryGetValue((confidantId, sceneNumber), out string? hint))
            return hint;
        return $"scene {sceneNumber}";
    }
}
