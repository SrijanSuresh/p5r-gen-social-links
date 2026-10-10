using System;
using System.Collections.Generic;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Who a line is spoken <em>to</em>, inferred from the order the scene's speakers take
/// their turns.
///
/// The BMD has no addressee field and could not have one: the renderer draws the speaker's
/// name above the bubble and never draws the listener, so the listener was never stored. A
/// format holds what its consumer reads (Ch. 81). Attribution could be parsed; this has to
/// be derived.
///
/// What the archive does give us is every speaker in script order for the whole scene,
/// including records the player has not reached. One rule covers it:
///
/// <para><b>The addressee of a line is the most recent earlier speaker who is not the
/// line's own speaker — and the protagonist when there is none.</b></para>
///
/// The three cases that rule was written from all fall out of it:
///
/// <list type="bullet">
/// <item>The previous line was somebody else's, so they are the addressee. This is the
/// case that fixes Takemi answering the protagonist when a patient's father asked her a
/// question.</item>
/// <item>The previous lines were the speaker's own, so they are skipped and the addressee
/// carries forward. A character running three bubbles long has not changed who they are
/// talking to.</item>
/// <item>Nothing earlier names anybody, so it is the protagonist. This is also the
/// whole-scene answer for a solo hang-out, which is most of them — 21 records of Ryuji at
/// the gym never find a different speaker and never stop addressing Joker.</item>
/// </list>
///
/// See learning.md Ch. 81.
/// </summary>
internal static class Addressee
{
    /// <summary>
    /// The protagonist's codename, and the addressee of any line with no earlier speaker
    /// to answer.
    ///
    /// The prompt has always called him this, and the player's chosen name is not
    /// something the mod reads.
    /// </summary>
    internal const string Protagonist = "Joker";

    /// <summary>
    /// Identity of a turn taken by the protagonist.
    ///
    /// Distinct from every table index and from <see cref="BmdMessage.NoSpeaker"/>,
    /// because the protagonist is a party to every scene and a speaker in none: he is
    /// silent by design and his lines are menu selections, which is why he never appears
    /// in a speaker table. The set of people who can be addressed is the speaker table
    /// plus him, and leaving that asymmetry implicit is how the silent party gets lost.
    /// </summary>
    internal const int ProtagonistId = -1;

    /// <summary>A turn whose speaker could not be identified — narration, or an unparsed header.</summary>
    internal const int UnknownId = -2;

    /// <summary>
    /// One record reduced to what the turn model needs: who spoke, and what to call them.
    ///
    /// Deliberately not <see cref="RecordPlan"/>. The parsers in this folder are compiled
    /// into the test assembly rather than referenced through it, so anything tested here
    /// has to take plain data and return a value.
    /// </summary>
    internal readonly struct Turn
    {
        /// <summary>Index into the scene's speaker table, or <see cref="BmdMessage.NoSpeaker"/>.</summary>
        internal int SpeakerId { get; }

        /// <summary>Resolved display name, or empty when the table did not name this speaker.</summary>
        internal string Name { get; }

        /// <summary>
        /// True for a <c>SEL</c> record — the list of replies the player picks from.
        ///
        /// A selection is the one place the protagonist speaks, so it counts as his turn.
        /// That matters for the line after it: a confidant answering a menu choice is
        /// answering Joker, and without this the rule would carry forward whichever NPC
        /// spoke before the choice was offered.
        /// </summary>
        internal bool IsSelection { get; }

        internal Turn(int speakerId, string name, bool isSelection = false)
        {
            SpeakerId   = speakerId;
            Name        = name ?? string.Empty;
            IsSelection = isSelection;
        }

        /// <summary>
        /// Which participant took this turn, as a comparable id.
        ///
        /// Narration and an unparsed header share <see cref="UnknownId"/> for the same
        /// reason they share <see cref="BmdMessage.NoSpeaker"/> in the plan: neither names
        /// a participant, and a line cannot be a reply to nobody.
        /// </summary>
        internal int Identity =>
            IsSelection                     ? ProtagonistId
            : SpeakerId == BmdMessage.NoSpeaker ? UnknownId
            : SpeakerId;
    }

    /// <summary>
    /// The addressee of <paramref name="index"/> within <paramref name="turns"/>, or an
    /// empty string when the turn changed hands to somebody the scene did not name.
    ///
    /// Empty is not the same as <see cref="Protagonist"/>, and the difference is the
    /// point: we know the line is answering someone other than Joker, so saying "Joker"
    /// would assert something false. The caller leaves the addressee out of the prompt
    /// instead, which is where it was before this existed.
    /// </summary>
    internal static string Resolve(IReadOnlyList<Turn> turns, int index)
    {
        if (turns is null || index < 0 || index >= turns.Count) return Protagonist;

        int me = turns[index].Identity;

        for (int j = index - 1; j >= 0; j--)
        {
            int other = turns[j].Identity;

            // Narration is transparent. It sits between two turns without taking one, and
            // treating it as a participant would have every line after a stage direction
            // addressed to the narrator.
            if (other == UnknownId) continue;

            // Still the same person talking: their earlier bubbles are not someone for
            // them to answer, so keep walking back for the turn before theirs.
            if (other == me) continue;

            if (other == ProtagonistId) return Protagonist;

            return turns[j].Name.Length > 0 ? turns[j].Name : string.Empty;
        }

        // Nobody spoke earlier, so there is nothing to be answering. A hang-out happens
        // because the player started it, and its opening line is aimed at them.
        return Protagonist;
    }

    /// <summary>
    /// True when <paramref name="name"/> is the protagonist rather than a member of the
    /// scene's cast.
    ///
    /// The prompt wording differs for the two — "you trust him completely" is a thing to
    /// say about Joker and not about a patient's father — so the caller has to be able to
    /// tell them apart without string-comparing against a constant itself.
    /// </summary>
    internal static bool IsProtagonist(string? name) =>
        string.Equals(name, Protagonist, StringComparison.Ordinal);
}
