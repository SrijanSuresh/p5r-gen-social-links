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

    /// <summary>
    /// Who this line is spoken to, or empty when the scene changed turn to somebody it
    /// did not name.
    ///
    /// Derived from the turn order rather than read from the file, because the format has
    /// no addressee field and could not have one — the renderer never draws the listener,
    /// so it was never stored (Ch. 81). Resolved once per plan: it depends only on the
    /// speaker ids, which are fixed at arm time.
    ///
    /// Empty and <see cref="Addressee.Protagonist"/> are different answers. Empty means
    /// "somebody, but we cannot say who", and the prompt omits the line instead of
    /// claiming Joker.
    /// </summary>
    internal string Addressee { get; set; } = string.Empty;

    /// <summary>
    /// True when this record is a <c>SEL</c> — the list of replies the player picks from,
    /// not a line anybody says.
    ///
    /// Two separate things need it. It is the one record kind that must never be
    /// rewritten, because changing what the buttons say without changing what they do is
    /// worse than a mis-attributed line (Ch. 80). And it is the only place the
    /// protagonist speaks, so it is his turn as far as the addressee model is concerned
    /// (Ch. 81) — a confidant answering a menu choice is answering Joker.
    /// </summary>
    internal bool IsSelection { get; set; }

    /// <summary>
    /// True when this record was reopened to answer a line the player typed.
    ///
    /// Carried so the write path can tell two outcomes apart that take the same code
    /// path. A pre-generated line discarded because the player advanced past it is
    /// routine and worth one line of log; a *reply to something the player typed*
    /// discarded for the same reason is the feature not working, and deserves to say so
    /// (Ch. 88).
    ///
    /// Cleared when the record is handed back to the ordinary queue, so a later
    /// generation for the same record is not mislabelled.
    /// </summary>
    internal bool AnsweringPlayer { get; set; }

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

    /// <summary>
    /// True while the record may still be overwritten. Written is included on purpose:
    /// a pre-generated line is a floor that a reactive generation is allowed to raise,
    /// right up until the interpreter reads it.
    ///
    /// <see cref="IsSelection"/> is checked here rather than in the speaker filter because
    /// Ch. 80's rule is "never, regardless" and the filter is none of those things: it can
    /// be switched off in <c>GenDialogue.json</c>, it goes idle when a scene's archive does
    /// not resolve, and it declines to act at all when nothing attributes to the confidant.
    /// A selection survived every one of those paths — protected only incidentally, by
    /// having no speaker to match. Protection that depends on a config flag is not the
    /// rule the chapter states.
    /// </summary>
    internal bool IsWritable =>
        !IsSelection && State != RecordState.Rendered && State != RecordState.Foreign;

    public override string ToString()
    {
        string who = Speaker.Length > 0 ? Speaker
                   : SpeakerId != BmdMessage.NoSpeaker ? $"spk{SpeakerId}"
                   : "-";
        return $"#{Index} {State} cap={Capacity} {who} " +
               $"\"{Original[..System.Math.Min(Original.Length, 32)]}\"";
    }
}
