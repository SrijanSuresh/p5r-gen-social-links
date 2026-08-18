using System.Collections.Generic;
using P5RGenSocialLinks.Memory;
using Xunit;

namespace P5RGenSocialLinks.Tests;

/// <summary>
/// Matching a scene's speaker label to a confidant (learning.md Ch. 77).
///
/// The cases that matter are the collisions. Anything can match "Takemi" to "Tae Takemi";
/// the job is to *not* match "Sakura" to Futaba while Sojiro is in the room.
/// </summary>
public class SpeakerMatchTests
{
    private static readonly string[] Everyone =
    {
        "Igor", "Morgana", "Makoto Niijima", "Haru Okumura", "Yusuke Kitagawa",
        "Sojiro Sakura", "Ann Takamaki", "Ryuji Sakamoto", "Goro Akechi",
        "Futaba Sakura", "Chihaya Mifune", "Justine and Caroline", "Munehisa Iwai",
        "Tae Takemi", "Sadayo Kawakami", "Ichiko Ohya", "Shinya Oda", "Hifumi Togo",
        "Yuuki Mishima", "Toranosuke Yoshida", "Sae Niijima", "Lavenza",
        "Kasumi Yoshizawa", "Takuto Maruki",
    };

    private static HashSet<string> Ambiguous() => SpeakerMatch.AmbiguousTokens(Everyone);

    // --- tokenizing -----------------------------------------------------------------

    [Fact]
    public void Lowercases_and_splits_on_anything_that_is_not_a_letter()
    {
        Assert.Equal(new[] { "tae", "takemi" }, SpeakerMatch.Tokenize("Tae Takemi"));
        Assert.Equal(new[] { "takemi" },        SpeakerMatch.Tokenize("Dr. Takemi"));
        Assert.Equal(new[] { "sakamoto" },      SpeakerMatch.Tokenize("SAKAMOTO"));
    }

    [Fact]
    public void Drops_honorifics_and_very_short_words()
    {
        Assert.Empty(SpeakerMatch.Tokenize("Dr."));
        Assert.Empty(SpeakerMatch.Tokenize("-sensei"));
        Assert.Equal(new[] { "kawakami" }, SpeakerMatch.Tokenize("Kawakami-sensei"));
        Assert.Equal(new[] { "ryuji" },    SpeakerMatch.Tokenize("Ryuji-kun"));
    }

    [Fact]
    public void An_unlabelled_speaker_produces_no_tokens()
    {
        // "???" is how the game labels someone before an introduction, and it must match
        // nobody rather than everybody.
        Assert.Empty(SpeakerMatch.Tokenize("???"));
        Assert.Empty(SpeakerMatch.Tokenize(""));
        Assert.Empty(SpeakerMatch.Tokenize(null));
        Assert.Empty(SpeakerMatch.Tokenize("   "));
    }

    // --- ambiguity ------------------------------------------------------------------

    [Fact]
    public void Surnames_shared_by_two_confidants_are_ambiguous()
    {
        HashSet<string> ambiguous = Ambiguous();

        Assert.Contains("sakura",  ambiguous);   // Futaba and Sojiro
        Assert.Contains("niijima", ambiguous);   // Makoto and Sae
    }

    [Fact]
    public void Distinctive_tokens_are_not_ambiguous()
    {
        HashSet<string> ambiguous = Ambiguous();

        Assert.DoesNotContain("takemi", ambiguous);
        Assert.DoesNotContain("futaba", ambiguous);
        Assert.DoesNotContain("sojiro", ambiguous);
        Assert.DoesNotContain("makoto", ambiguous);
    }

    [Fact]
    public void A_token_repeated_inside_one_name_is_still_one_confidant()
    {
        Assert.Empty(SpeakerMatch.AmbiguousTokens(new[] { "Ann Ann Takamaki" }));
    }

    // --- matching -------------------------------------------------------------------

    [Theory]
    [InlineData("Takemi",     "Tae Takemi")]
    [InlineData("Dr. Takemi", "Tae Takemi")]
    [InlineData("Tae",        "Tae Takemi")]
    [InlineData("Ryuji",      "Ryuji Sakamoto")]
    [InlineData("Sakamoto",   "Ryuji Sakamoto")]
    [InlineData("Futaba",     "Futaba Sakura")]
    [InlineData("Sojiro",     "Sojiro Sakura")]
    [InlineData("Morgana",    "Morgana")]
    public void Matches_a_label_to_its_confidant(string label, string confidant)
    {
        Assert.True(SpeakerMatch.Matches(label, confidant, Ambiguous()));
    }

    [Fact]
    public void An_ambiguous_surname_matches_nobody()
    {
        // The whole point. A bubble labelled "Sakura" genuinely does not say which one,
        // and answering "Futaba" would rewrite Sojiro's lines in her voice.
        HashSet<string> ambiguous = Ambiguous();

        Assert.False(SpeakerMatch.Matches("Sakura", "Futaba Sakura", ambiguous));
        Assert.False(SpeakerMatch.Matches("Sakura", "Sojiro Sakura", ambiguous));
        Assert.False(SpeakerMatch.Matches("Niijima", "Makoto Niijima", ambiguous));
    }

    [Fact]
    public void A_full_name_still_matches_past_its_ambiguous_half()
    {
        // "Futaba Sakura" carries one ambiguous token and one distinctive one. The
        // distinctive one decides.
        HashSet<string> ambiguous = Ambiguous();

        Assert.True(SpeakerMatch.Matches("Futaba Sakura", "Futaba Sakura", ambiguous));
        Assert.False(SpeakerMatch.Matches("Futaba Sakura", "Sojiro Sakura", ambiguous));
    }

    [Theory]
    [InlineData("Ann",     "Tae Takemi")]
    [InlineData("Nurse",   "Tae Takemi")]
    [InlineData("???",     "Tae Takemi")]
    [InlineData("",        "Tae Takemi")]
    [InlineData("Takemi",  "")]
    public void Does_not_match_someone_else(string label, string confidant)
    {
        Assert.False(SpeakerMatch.Matches(label, confidant, Ambiguous()));
    }

    // --- through the real table -----------------------------------------------------

    [Fact]
    public void Resolves_against_the_shipped_confidant_table()
    {
        Assert.True(ConfidantNames.IsSpokenBy(14, "Takemi"));       // Tae Takemi
        Assert.True(ConfidantNames.IsSpokenBy(14, "Dr. Takemi"));
        Assert.True(ConfidantNames.IsSpokenBy(8,  "Ryuji"));        // Ryuji Sakamoto

        Assert.False(ConfidantNames.IsSpokenBy(14, "Ryuji"));
        Assert.False(ConfidantNames.IsSpokenBy(10, "Sakura"));      // Futaba vs Sojiro
        Assert.False(ConfidantNames.IsSpokenBy(14, "???"));
        Assert.False(ConfidantNames.IsSpokenBy(999, "Takemi"));     // unknown id
        Assert.False(ConfidantNames.IsSpokenBy(14, null));
    }
}
