using System.Collections.Generic;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Maps P5R internal CMM IDs to confidant display names.
///
/// This table is a mirror of <c>server/social_link/arcana.py</c>, and the two are pinned
/// together by <c>server/tests/test_confidant_tables.py</c>. They had drifted: this side
/// was missing Justine and Caroline (12) and Lavenza (22), which shifted every id above
/// them by one and made the mod call Tae Takemi "Sadayo Kawakami" in the context it sends
/// with every request.
///
/// The drift was harmless-looking while the name was only ever printed, because the server
/// keys off <c>confidant_id</c> and had the right character all along. It stops being
/// harmless once the name is compared against the speaker table in a scene's BMD — a
/// mismatch there does not print the wrong label, it declines to rewrite the confidant's
/// own lines.
/// </summary>
internal static class ConfidantNames
{
    private static readonly Dictionary<int, string> _names = new()
    {
        {  1, "Igor"                 },
        {  2, "Morgana"              },
        {  3, "Makoto Niijima"       },
        {  4, "Haru Okumura"         },
        {  5, "Yusuke Kitagawa"      },
        {  6, "Sojiro Sakura"        },
        {  7, "Ann Takamaki"         },
        {  8, "Ryuji Sakamoto"       },
        {  9, "Goro Akechi"          },
        { 10, "Futaba Sakura"        },
        { 11, "Chihaya Mifune"       },
        { 12, "Justine and Caroline" },
        { 13, "Munehisa Iwai"        },
        { 14, "Tae Takemi"           },
        { 15, "Sadayo Kawakami"      },
        { 16, "Ichiko Ohya"          },
        { 17, "Shinya Oda"           },
        { 18, "Hifumi Togo"          },
        { 19, "Yuuki Mishima"        },
        { 20, "Toranosuke Yoshida"   },
        { 21, "Sae Niijima"          },
        { 22, "Lavenza"              },
        // P5R exclusives — ids provisional, not yet confirmed from game memory.
        { 25, "Kasumi Yoshizawa"     },
        { 26, "Takuto Maruki"        },
    };

    internal static string Resolve(int confidantId) =>
        _names.TryGetValue(confidantId, out string? name) ? name : $"Confidant #{confidantId}";
}
