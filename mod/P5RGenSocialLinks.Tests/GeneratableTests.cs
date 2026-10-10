using System.Collections.Generic;
using P5RGenSocialLinks.Memory;
using Xunit;

namespace P5RGenSocialLinks.Tests;

/// <summary>
/// Choosing the record that answers a line the player typed (learning.md Ch. 88).
///
/// The interesting cases are the ones where the obvious answer is wrong: the target is
/// usually <em>not</em> a Pending record, because a player who pauses to type is a player
/// whose whole lookahead has just finished filling.
/// </summary>
public class GeneratableTests
{
    private static RecordPlan Record(
        int index,
        RecordState state = RecordState.Pending,
        int capacity = 40,
        string original = "Oh yeah! You bring your stuff?",
        bool selection = false) =>
        new()
        {
            Index = index, Capacity = capacity, Original = original,
            State = state, IsSelection = selection,
        };

    // --- what is worth asking about -------------------------------------------------

    [Fact]
    public void An_ordinary_record_is_worth_generating_for()
    {
        Assert.True(Generatable.IsWorthGenerating(Record(0)));
    }

    [Fact]
    public void A_selection_is_never_worth_generating_for()
    {
        Assert.False(Generatable.IsWorthGenerating(Record(0, selection: true)));
    }

    [Fact]
    public void A_scripted_line_too_short_to_be_dialogue_is_skipped()
    {
        Assert.False(Generatable.IsWorthGenerating(Record(0, original: "Huh?")));
        // Exactly at the threshold is in.
        Assert.True(Generatable.IsWorthGenerating(Record(0, original: new string('x', 8))));
    }

    [Fact]
    public void A_record_too_small_for_a_sentence_is_skipped()
    {
        Assert.False(Generatable.IsWorthGenerating(Record(0, capacity: 23)));
        Assert.True(Generatable.IsWorthGenerating(Record(0, capacity: 24)));
    }

    [Fact]
    public void A_null_record_is_not_worth_generating_for()
    {
        Assert.False(Generatable.IsWorthGenerating(null!));
    }

    // --- picking the record that answers a typed line -------------------------------

    [Fact]
    public void The_target_is_the_first_unread_record()
    {
        var plan = new List<RecordPlan>
        {
            Record(0, RecordState.Rendered),
            Record(1, RecordState.Rendered),
            Record(2),
            Record(3),
        };

        Assert.Equal(2, Generatable.FindTypedLineTarget(plan, from: 2));
    }

    [Fact]
    public void A_record_the_queue_already_filled_is_still_a_valid_target()
    {
        // This is the whole difficulty. The player paused to type, the lookahead caught
        // up, and the record they are about to read is Written with something else.
        var plan = new List<RecordPlan> { Record(0, RecordState.Written) };

        Assert.Equal(0, Generatable.FindTypedLineTarget(plan, from: 0));
    }

    [Fact]
    public void A_record_waiting_to_be_written_is_also_a_valid_target()
    {
        var plan = new List<RecordPlan> { Record(0, RecordState.Ready) };

        Assert.Equal(0, Generatable.FindTypedLineTarget(plan, from: 0));
    }

    [Fact]
    public void Records_behind_the_cursor_are_never_targets()
    {
        // Everything before the cursor has been read. Answering there is the bug the
        // player described as the text switching.
        var plan = new List<RecordPlan> { Record(0), Record(1), Record(2) };

        Assert.Equal(2, Generatable.FindTypedLineTarget(plan, from: 2));
    }

    [Fact]
    public void A_rendered_record_is_skipped_even_at_the_cursor()
    {
        var plan = new List<RecordPlan>
        {
            Record(0, RecordState.Rendered),
            Record(1),
        };

        Assert.Equal(1, Generatable.FindTypedLineTarget(plan, from: 0));
    }

    [Fact]
    public void Another_speakers_line_is_skipped()
    {
        // Answering the player in somebody else's bubble is worse than not answering.
        var plan = new List<RecordPlan>
        {
            Record(0, RecordState.Foreign),
            Record(1, RecordState.Foreign),
            Record(2),
        };

        Assert.Equal(2, Generatable.FindTypedLineTarget(plan, from: 0));
    }

    [Fact]
    public void A_selection_is_skipped_as_a_target()
    {
        var plan = new List<RecordPlan>
        {
            Record(0, selection: true),
            Record(1),
        };

        Assert.Equal(1, Generatable.FindTypedLineTarget(plan, from: 0));
    }

    [Fact]
    public void Records_too_short_to_hold_a_reply_are_skipped()
    {
        var plan = new List<RecordPlan>
        {
            Record(0, capacity: 12),
            Record(1, original: "Hm."),
            Record(2),
        };

        Assert.Equal(2, Generatable.FindTypedLineTarget(plan, from: 0));
    }

    // --- nothing to answer with -----------------------------------------------------

    [Fact]
    public void A_scene_with_nothing_writable_ahead_has_no_target()
    {
        // The caller must leave the typed line in the store rather than discard it, so
        // this has to be distinguishable from "record 0".
        var plan = new List<RecordPlan>
        {
            Record(0, RecordState.Rendered),
            Record(1, RecordState.Foreign),
        };

        Assert.Equal(-1, Generatable.FindTypedLineTarget(plan, from: 0));
    }

    [Fact]
    public void A_cursor_past_the_end_of_the_plan_has_no_target()
    {
        var plan = new List<RecordPlan> { Record(0) };

        Assert.Equal(-1, Generatable.FindTypedLineTarget(plan, from: 5));
    }

    [Fact]
    public void An_empty_or_missing_plan_has_no_target()
    {
        Assert.Equal(-1, Generatable.FindTypedLineTarget(new List<RecordPlan>(), from: 0));
        Assert.Equal(-1, Generatable.FindTypedLineTarget(null!, from: 0));
    }

    [Fact]
    public void A_negative_cursor_is_treated_as_the_start()
    {
        var plan = new List<RecordPlan> { Record(0) };

        Assert.Equal(0, Generatable.FindTypedLineTarget(plan, from: -3));
    }
}
