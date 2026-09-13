using P5RGenSocialLinks.Memory;
using Xunit;

namespace P5RGenSocialLinks.Tests;

/// <summary>
/// Inferring who a line is spoken to from the scene's turn order (learning.md Ch. 81).
///
/// The scenes worth encoding are the two real ones: Takemi's clinic, where three people
/// share a room and the addressee changes hands, and Ryuji's gym, where one person talks
/// for twenty-one records and the answer must never stop being Joker.
/// </summary>
public class AddresseeTests
{
    // Speaker-table indices as the clinic file reports them:
    //   [0] Takemi  [1] Girl's Father  [2] Sick Girl
    private const int Takemi = 0;
    private const int Father = 1;
    private const int Girl   = 2;

    private static Addressee.Turn Spoken(int id, string name) => new(id, name);

    private static Addressee.Turn Narration() =>
        new(BmdMessage.NoSpeaker, string.Empty);

    private static Addressee.Turn Selection() =>
        new(BmdMessage.NoSpeaker, string.Empty, isSelection: true);

    // --- the solo hang-out, which is most of them -----------------------------------

    [Fact]
    public void A_scene_with_one_speaker_addresses_the_protagonist_throughout()
    {
        var turns = new[]
        {
            Spoken(Takemi, "Ryuji"), Spoken(Takemi, "Ryuji"), Spoken(Takemi, "Ryuji"),
        };

        Assert.Equal("Joker", Addressee.Resolve(turns, 0));
        Assert.Equal("Joker", Addressee.Resolve(turns, 1));
        Assert.Equal("Joker", Addressee.Resolve(turns, 2));
    }

    [Fact]
    public void The_opening_line_of_any_scene_addresses_the_protagonist()
    {
        var turns = new[] { Spoken(Father, "Girl's Father"), Spoken(Takemi, "Takemi") };

        Assert.Equal("Joker", Addressee.Resolve(turns, 0));
    }

    // --- the clinic, where the turn changes hands -----------------------------------

    [Fact]
    public void A_line_after_somebody_elses_answers_them()
    {
        var turns = new[]
        {
            Spoken(Takemi, "Takemi"),
            Spoken(Father, "Girl's Father"),
            Spoken(Takemi, "Takemi"),
        };

        Assert.Equal("Girl's Father", Addressee.Resolve(turns, 2));
    }

    [Fact]
    public void A_speaker_running_several_bubbles_keeps_the_same_addressee()
    {
        var turns = new[]
        {
            Spoken(Father, "Girl's Father"),
            Spoken(Takemi, "Takemi"),
            Spoken(Takemi, "Takemi"),
            Spoken(Takemi, "Takemi"),
        };

        Assert.Equal("Girl's Father", Addressee.Resolve(turns, 1));
        Assert.Equal("Girl's Father", Addressee.Resolve(turns, 2));
        Assert.Equal("Girl's Father", Addressee.Resolve(turns, 3));
    }

    [Fact]
    public void The_addressee_follows_whoever_spoke_most_recently()
    {
        var turns = new[]
        {
            Spoken(Father, "Girl's Father"),
            Spoken(Takemi, "Takemi"),
            Spoken(Girl,   "Sick Girl"),
            Spoken(Takemi, "Takemi"),
        };

        Assert.Equal("Girl's Father", Addressee.Resolve(turns, 1));
        Assert.Equal("Sick Girl",     Addressee.Resolve(turns, 3));
    }

    // --- narration is not a participant ---------------------------------------------

    [Fact]
    public void Narration_between_two_turns_does_not_become_the_addressee()
    {
        var turns = new[]
        {
            Spoken(Father, "Girl's Father"),
            Narration(),
            Spoken(Takemi, "Takemi"),
        };

        Assert.Equal("Girl's Father", Addressee.Resolve(turns, 2));
    }

    [Fact]
    public void A_scene_of_pure_narration_addresses_the_protagonist()
    {
        var turns = new[] { Narration(), Narration(), Spoken(Takemi, "Takemi") };

        Assert.Equal("Joker", Addressee.Resolve(turns, 2));
    }

    // --- the protagonist's own turn is a menu selection -----------------------------

    [Fact]
    public void A_line_after_a_selection_answers_the_protagonist()
    {
        // The player picked a reply, so the next line is answering him — not whichever
        // NPC spoke before the choice was offered.
        var turns = new[]
        {
            Spoken(Father, "Girl's Father"),
            Selection(),
            Spoken(Takemi, "Takemi"),
        };

        Assert.Equal("Joker", Addressee.Resolve(turns, 2));
    }

    // --- an unnamed speaker is not silently replaced with Joker ----------------------

    [Fact]
    public void A_turn_that_changed_hands_to_an_unnamed_speaker_resolves_to_nothing()
    {
        // The table knows somebody else spoke but not what to call them. Answering
        // "Joker" here would assert something we know to be wrong, so the caller gets an
        // empty string and leaves the addressee out of the prompt entirely.
        var turns = new[]
        {
            Spoken(Father, string.Empty),
            Spoken(Takemi, "Takemi"),
        };

        Assert.Equal(string.Empty, Addressee.Resolve(turns, 1));
    }

    // --- boundaries -----------------------------------------------------------------

    [Fact]
    public void An_index_outside_the_scene_addresses_the_protagonist()
    {
        var turns = new[] { Spoken(Takemi, "Takemi") };

        Assert.Equal("Joker", Addressee.Resolve(turns, -1));
        Assert.Equal("Joker", Addressee.Resolve(turns, 5));
        Assert.Equal("Joker", Addressee.Resolve(null!, 0));
    }

    [Fact]
    public void IsProtagonist_distinguishes_Joker_from_the_scenes_cast()
    {
        Assert.True(Addressee.IsProtagonist("Joker"));
        Assert.False(Addressee.IsProtagonist("Girl's Father"));
        Assert.False(Addressee.IsProtagonist(""));
        Assert.False(Addressee.IsProtagonist(null));
    }
}
