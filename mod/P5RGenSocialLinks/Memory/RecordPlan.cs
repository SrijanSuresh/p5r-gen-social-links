namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Where a single message record is in its life cycle.
///
/// Written as an explicit state rather than inferred from "has text" or "index &lt; cursor",
/// because Ch. 71 was two separate bugs caused by a cursor that different events moved for
/// different reasons. Every transition here is made deliberately by one place.
/// </summary>
internal enum RecordState
{
    /// Nothing requested yet.
    Pending,

    /// A generation is in flight. Distinct from Pending so the scheduler cannot ask twice.
    InFlight,

    /// Text has arrived but has not been written into game memory. Separate from Written
    /// because generation completing and bytes landing are different moments that fail
    /// for different reasons — a 503 loses the text, a freed region loses the write.
    Ready,

    /// The bytes are in the pool. Still overwritable: the player has not seen it yet, so
    /// a reactive generation may still upgrade it.
    Written,

    /// The interpreter has read it. Frozen — overwriting now is the bug the player
    /// described as "the text keeps switching".
    Rendered,

    /// <summary>
    /// The header attributes this line to somebody other than the confidant.
    ///
    /// Terminal, and decided at plan time rather than reached through the life cycle. A
    /// Takemi rank-up contains a patient and her father, and every one of their lines used
    /// to be rewritten in Takemi's voice because the only thing the mod knew about a
    /// record was that it was a record (Ch. 77).
    ///
    /// Distinct from Rendered because the two mean opposite things about the same record:
    /// Rendered is "too late", Foreign is "not ours". Collapsing them would report a scene
    /// full of other people's dialogue as a scene the queue failed to keep up with.
    /// </summary>
    Foreign,
}

/// <summary>
/// One record of a scene, and everything needed to replace its line without the player
/// having reached it yet.
///
/// The point of holding this per record rather than per event is that all of it is known
/// at arm time. Waiting for a dispatch to discover the capacity or the original text was
/// only ever necessary while the buffer was being found by heuristic.
/// </summary>
internal sealed class RecordPlan
{
    /// Index within the region's record list. Also the order the scene plays in, since
    /// records are laid out in script order and read monotonically upward.
    internal int Index { get; init; }

    /// Total characters the bubble can display across all its rows, including the word
    /// break each row join contributes. This is the generation budget.
    internal int Capacity { get; init; }

    /// The scripted line being replaced. Captured before any write, because the write
    /// destroys it and it is the best context the model can be given about this specific
    /// moment (Ch. 64).
    internal string Original { get; init; } = string.Empty;

    /// <summary>
    /// The BMD symbol for this message, e.g. "MSG_001_5_0", or empty when the header
    /// could not be located. Diagnostic rather than load-bearing: it is what makes a
    /// mis-grouped record obvious in the log, because names run in script order.
    /// </summary>
    internal string Name { get; set; } = string.Empty;

    /// <summary>
    /// Index into the scene's speaker table, or <see cref="BmdMessage.NoSpeaker"/> for
    /// narration and for records whose header was not found.
    ///
    /// Unknown and narration deliberately share a value. Both mean "not attributable to
    /// the confidant", and the safe response to either is the same: leave the line alone.
    /// </summary>
    internal int SpeakerId { get; set; } = BmdMessage.NoSpeaker;

    /// <summary>Resolved display name for <see cref="SpeakerId"/>, when the table is known.</summary>
    internal string Speaker { get; set; } = string.Empty;

    internal RecordState State { get; set; } = RecordState.Pending;

    /// The generated replacement, once it exists.
    internal string? Generated { get; set; }

    /// <summary>
    /// How many times generation has been asked for and come back empty.
    ///
    /// Every failure returns the record to Pending so a busy server or a timeout is
    /// simply retried — which is right for a transient fault and wrong for a permanent
    /// one. A short record whose line cannot fit a complete sentence fails identically
    /// every time, and without a cap it would occupy the queue for the whole scene while
    /// records that could succeed waited behind it.
    /// </summary>
    internal int Attempts { get; set; }

    /// <summary>
    /// True once the bytes actually reached the pool.
    ///
    /// Separate from State because a record can be generated and then frozen by the
    /// interpreter before the flush runs, which leaves Generated set and State at
    /// Rendered without a single byte having been written. Coverage counted those as
    /// successes and reported 10 replaced against 7 writes in the log beneath it.
    /// </summary>
    internal bool WasWritten { get; set; }

    /// True while the record may still be overwritten. Written is included on purpose:
    /// a pre-generated line is a floor that a reactive generation is allowed to raise,
    /// right up until the interpreter reads it.
    internal bool IsWritable =>
        State != RecordState.Rendered && State != RecordState.Foreign;

    public override string ToString()
    {
        string who = Speaker.Length > 0 ? Speaker
                   : SpeakerId != BmdMessage.NoSpeaker ? $"spk{SpeakerId}"
                   : "-";
        return $"#{Index} {State} cap={Capacity} {who} " +
               $"\"{Original[..System.Math.Min(Original.Length, 32)]}\"";
    }
}
