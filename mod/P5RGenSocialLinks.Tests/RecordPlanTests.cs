using P5RGenSocialLinks.Memory;
using Xunit;

namespace P5RGenSocialLinks.Tests;

/// <summary>
/// The writability invariant — what the mod is allowed to overwrite, and when.
///
/// Every rule here was a bug first. The record life cycle exists because Ch. 71 turned
/// out to be two separate faults behind one cursor, and each terminal state below marks
/// a different thing that went wrong on screen.
/// </summary>
public class RecordPlanTests
{
    private static RecordPlan Record(
        RecordState state = RecordState.Pending, bool selection = false) =>
        new() { Index = 0, Capacity = 40, Original = "Oh yeah! You bring your stuff?",
                State = state, IsSelection = selection };

    // --- the ordinary life cycle ----------------------------------------------------

    [Fact]
    public void A_record_the_player_has_not_reached_is_writable()
    {
        // Looped rather than a [Theory]: RecordState is internal, and a public test
        // method cannot take one as a parameter.
        foreach (RecordState state in new[]
                 {
                     RecordState.Pending, RecordState.InFlight,
                     RecordState.Ready,   RecordState.Written,
                 })
            Assert.True(Record(state).IsWritable, $"{state} should be writable");
    }

    [Fact]
    public void Written_stays_writable_so_a_better_line_can_replace_it()
    {
        // A pre-generated line is a floor, not a final answer: it may be upgraded right
        // up until the interpreter reads the record.
        Assert.True(Record(RecordState.Written).IsWritable);
    }

    [Fact]
    public void Rendered_is_frozen_because_the_player_is_looking_at_it()
    {
        // Overwriting here is what the player described as "the text keeps switching".
        Assert.False(Record(RecordState.Rendered).IsWritable);
    }

    [Fact]
    public void Foreign_is_frozen_because_the_line_belongs_to_somebody_else()
    {
        Assert.False(Record(RecordState.Foreign).IsWritable);
    }

    // --- the rule Ch. 80 states as "never, regardless" -------------------------------

    [Fact]
    public void A_selection_is_never_writable_in_any_state()
    {
        // Changing what the buttons say without changing what they do is a worse bug
        // than a mis-attributed line, so this cannot depend on reaching a state, on the
        // speaker filter being switched on, or on the scene's archive having resolved.
        //
        // Enumerated from the enum itself rather than listed, so a state added later is
        // covered without anyone remembering to add it here.
        foreach (RecordState state in System.Enum.GetValues<RecordState>())
            Assert.False(Record(state, selection: true).IsWritable,
                         $"a selection in {state} must never be writable");
    }

    // --- defaults ------------------------------------------------------------------

    [Fact]
    public void A_fresh_record_is_unattributed_and_unaddressed()
    {
        var plan = new RecordPlan();

        // Narration and "header not found" share NoSpeaker deliberately: both mean "not
        // attributable to the confidant", and the safe answer to either is to leave the
        // line alone.
        Assert.Equal(BmdMessage.NoSpeaker, plan.SpeakerId);
        Assert.Equal(string.Empty, plan.Speaker);

        // Empty rather than "Joker" — an addressee nobody has worked out yet must not
        // arrive at the prompt as a claim.
        Assert.Equal(string.Empty, plan.Addressee);
        Assert.False(plan.IsSelection);
        Assert.False(plan.WasWritten);
        Assert.Equal(RecordState.Pending, plan.State);
    }

    [Fact]
    public void ToString_names_the_speaker_when_there_is_one()
    {
        var plan = new RecordPlan
        {
            Index = 4, Capacity = 30, Original = "Not quite famous enough, anyway.",
            Speaker = "Takemi", SpeakerId = 0,
        };

        string text = plan.ToString();
        Assert.Contains("#4", text);
        Assert.Contains("Takemi", text);
    }

    [Fact]
    public void ToString_falls_back_to_the_raw_id_then_to_a_dash()
    {
        // The log is where a mis-grouped record gets noticed, so an unnamed speaker has
        // to still be distinguishable from no speaker at all.
        Assert.Contains("spk3", new RecordPlan { SpeakerId = 3 }.ToString());
        Assert.Contains("-",    new RecordPlan().ToString());
    }
}
