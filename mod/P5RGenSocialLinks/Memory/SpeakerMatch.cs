using System;
using System.Collections.Generic;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Decides whether a speaker label from a scene's BMD names a given confidant.
///
/// The two strings are almost never equal. The game labels a bubble <c>Takemi</c>,
/// <c>Dr. Takemi</c> or <c>???</c>; the mod calls her <c>Tae Takemi</c>. Substring
/// matching handles those and then reintroduces the bug it was meant to fix, because
/// <c>Sakura</c> is inside both <c>Futaba Sakura</c> and <c>Sojiro Sakura</c> — so in a
/// Futaba scene, Sojiro's lines would be rewritten in Futaba's voice.
///
/// The rule instead is: compare name tokens, and only trust tokens that identify exactly
/// one confidant. Which tokens those are is derived from the confidant table rather than
/// written down, so adding a character with a colliding surname moves that surname into
/// the ambiguous set instead of silently creating a mis-attribution.
///
/// See learning.md Ch. 77.
/// </summary>
internal static class SpeakerMatch
{
    /// <summary>
    /// Words that appear in a speaker label without naming anybody.
    ///
    /// Honorifics arrive both as English titles and as Japanese suffixes the localisation
    /// keeps, and a label like "Dr. Takemi" has to reduce to the same token set as
    /// "Takemi" or the match fails on politeness.
    /// </summary>
    private static readonly HashSet<string> Honorifics = new(StringComparer.Ordinal)
    {
        "dr", "mr", "mrs", "ms", "miss", "prof", "professor", "doctor",
        "san", "kun", "chan", "sama", "sensei", "senpai", "sempai", "the",
    };

    /// <summary>
    /// Shortest token that can identify anyone. Nothing distinguishing survives at two
    /// letters, and the format's inline control codes routinely leave one behind.
    /// </summary>
    private const int MinTokenLength = 3;

    /// <summary>
    /// Split a name into comparable tokens: lowercase, letters only, no honorifics.
    ///
    /// Dropping non-letters rather than splitting on whitespace is what makes "Dr." lose
    /// its full stop and "???" reduce to nothing — an unlabelled bubble produces an empty
    /// token set, which matches no one, which is the right answer.
    /// </summary>
    internal static string[] Tokenize(string? name)
    {
        if (string.IsNullOrEmpty(name)) return Array.Empty<string>();

        var tokens  = new List<string>();
        var current = new System.Text.StringBuilder(name!.Length);

        for (int i = 0; i <= name.Length; i++)
        {
            char c = i < name.Length ? name[i] : ' ';
            if (char.IsLetter(c))
            {
                current.Append(char.ToLowerInvariant(c));
                continue;
            }

            if (current.Length >= MinTokenLength)
            {
                string token = current.ToString();
                if (!Honorifics.Contains(token)) tokens.Add(token);
            }
            current.Clear();
        }
        return tokens.ToArray();
    }

    /// <summary>
    /// Tokens that occur in more than one confidant's name, and therefore identify none
    /// of them. Computed from the table so it cannot fall out of date with it.
    /// </summary>
    internal static HashSet<string> AmbiguousTokens(IEnumerable<string> confidantNames)
    {
        var seen      = new Dictionary<string, int>(StringComparer.Ordinal);
        var ambiguous = new HashSet<string>(StringComparer.Ordinal);

        foreach (string name in confidantNames)
        {
            // A token repeated inside one name is still one confidant, so each name
            // contributes each token at most once.
            var contributed = new HashSet<string>(StringComparer.Ordinal);
            foreach (string token in Tokenize(name))
            {
                if (!contributed.Add(token)) continue;
                seen.TryGetValue(token, out int count);
                seen[token] = count + 1;
                if (count + 1 > 1) ambiguous.Add(token);
            }
        }
        return ambiguous;
    }

    /// <summary>
    /// True when <paramref name="speakerLabel"/> names <paramref name="confidantName"/>.
    ///
    /// Requires a shared token that identifies exactly one confidant. An empty label, an
    /// unknown name or a purely ambiguous one all return false, which the caller reads as
    /// "not the confidant" and therefore "leave this line alone".
    /// </summary>
    internal static bool Matches(
        string? speakerLabel, string? confidantName, HashSet<string> ambiguous)
    {
        string[] spoken = Tokenize(speakerLabel);
        if (spoken.Length == 0) return false;

        string[] target = Tokenize(confidantName);
        if (target.Length == 0) return false;

        foreach (string token in spoken)
        {
            if (ambiguous.Contains(token)) continue;
            foreach (string other in target)
                if (string.Equals(token, other, StringComparison.Ordinal)) return true;
        }
        return false;
    }
}
