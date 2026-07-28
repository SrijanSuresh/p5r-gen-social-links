using System.Collections.Generic;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Maps P5R internal CMM IDs to confidant display names.
/// IDs are the game's 1-indexed arcana numbers (confirmed via Amicitia wiki).
/// Kept in sync with server/social_link/arcana.py.
/// </summary>
internal static class ConfidantNames
{
    private static readonly Dictionary<int, string> _names = new()
    {
        {  1, "Igor"           },
        {  2, "Morgana"        },
        {  3, "Makoto Niijima" },
        {  4, "Haru Okumura"   },
        {  5, "Yusuke Kitagawa"},
        {  6, "Sojiro Sakura"  },
        {  7, "Ann Takamaki"   },
        {  8, "Ryuji Sakamoto" },
        {  9, "Goro Akechi"    },
        { 10, "Futaba Sakura"  },
        { 18, "Yuuki Mishima"  },
    };

    internal static string Resolve(int confidantId) =>
        _names.TryGetValue(confidantId, out string? name) ? name : $"Confidant #{confidantId}";
}
