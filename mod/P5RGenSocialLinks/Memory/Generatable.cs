using System.Collections.Generic;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Which records are worth asking the server about, and which one should answer a line
/// the player typed.
///
/// Both questions were answered inline in the pre-generation pump, with the thresholds
/// written as local constants. The typed-input path needs the identical test — a record
/// too short for the queue is too short for a reply as well — and a second copy of a
/// threshold is a threshold that drifts.
///
/// Pure functions over the plan, deliberately. This is the only part of the typed-input
/// path that can be tested without a game attached, so it is where the decisions live.
///
/// See learning.md Ch. 88.
/// </summary>
internal static class Generatable
{
    /// <summary>
    /// Shortest scripted line that counts as dialogue.
    ///
    /// Below this a record is a fragment of UI rather than something a character says, and
    /// replacing it spends a slow inference on text nobody reads as a line.
    /// </summary>
    internal const int MinOriginalChars = 8;

    /// <summary>
    /// Smallest record a complete sentence fits into.
    ///
    /// Measured: records of 15-21 characters produced "You can grab a" and "Guess we good
    /// to" before fragments started being rejected outright. The server now correctly
    /// returns nothing for these, so asking is a wasted round trip.
    /// </summary>
    internal const int MinCapacityChars = 24;

    /// <summary>
    /// True when a record is one the server could usefully write a line for.
    ///
    /// Says nothing about whether it is the right record *now* — that is the caller's
    /// question, and the two paths answer it differently: the queue wants the next
    /// <see cref="RecordState.Pending"/> one, and a typed line wants the next unread one
    /// whatever state it is in.
    /// </summary>
    internal static bool IsWorthGenerating(RecordPlan record) =>
        record is not null
        && !record.IsSelection
        && record.Original.Length >= MinOriginalChars
        && record.Capacity >= MinCapacityChars;

    /// <summary>
    /// Index of the record that should answer a typed line, or -1 if there is none.
    ///
    /// <paramref name="from"/> is the cursor: <c>_poolNextRecord[0]</c>, the index of the
    /// next record the interpreter will read. Everything at or after it is unread and so
    /// still writable; everything before it is frozen by the render freeze, which is the
    /// same mechanism that stopped the text switching under the player (Ch. 71).
    ///
    /// Note what is *not* required: the record does not have to be
    /// <see cref="RecordState.Pending"/>. The whole difficulty of typed input is that the
    /// queue has usually already filled this record — the player paused to type, and a
    /// pause is exactly when the lookahead catches up. A <see cref="RecordState.Written"/>
    /// record is a legitimate target precisely because the player has not read it yet.
    ///
    /// Returns -1 when the scene has nothing writable left ahead, which the caller must
    /// handle by leaving the typed line in the store rather than discarding it.
    /// </summary>
    internal static int FindTypedLineTarget(IReadOnlyList<RecordPlan> plan, int from)
    {
        if (plan is null) return -1;

        for (int i = from < 0 ? 0 : from; i < plan.Count; i++)
        {
            RecordPlan record = plan[i];

            // Frozen or foreign. Rendered means the interpreter already read it, and
            // Foreign means the line belongs to another speaker — answering the player in
            // somebody else's bubble is a worse bug than not answering at all.
            if (!record.IsWritable) continue;

            if (!IsWorthGenerating(record)) continue;

            return i;
        }
        return -1;
    }
}
