using System;
using System.Threading;
using System.Threading.Tasks;
using P5RGenSocialLinks.Memory;
using P5RGenSocialLinks.Server;

namespace P5RGenSocialLinks;

/// <summary>
/// Phase 5: the player types a line, and the confidant answers it.
///
/// The channel is the page the server hosts on loopback. What lives here is the half that
/// is actually difficult — getting the reply onto the right bubble.
///
/// Pre-generation keeps eight records ahead, so a player who pauses to type for ten
/// seconds is a player whose whole lookahead has just finished filling, including the
/// record they are about to read. That record has to be taken back: returned to the
/// queue, regenerated with the typed line, and rewritten before they advance.
///
/// Which is safe for a reason that predates the feature. Ch. 72's invariant — a record is
/// writable until the interpreter reads it, frozen afterwards — means everything at or
/// after <c>_poolNextRecord[0]</c> is fair game, and everything before it is protected by
/// the same mechanism that stopped the text switching under the player in Ch. 71.
///
/// See learning.md Ch. 88.
/// </summary>
public partial class Mod
{
    /// <summary>
    /// True when the server says a typed line is waiting.
    ///
    /// Written by the poll task, read by the pump on the poll thread. Volatile rather
    /// than locked: it is a single bool whose worst case is acting one tick late, and the
    /// authoritative answer is the collection attempt itself — which can still come back
    /// empty if the line expired in between.
    /// </summary>
    private volatile bool _playerLineWaiting;

    /// <summary>Guards against a slow poll overlapping the next one.</summary>
    private int _playerPollBusy;

    /// <summary>Ticks since the last peek, against <see cref="GenConfig.PlayerPollTicks"/>.</summary>
    private int _playerPollTick;

    /// <summary>
    /// True once "nowhere to put it" has been reported for the current scene.
    ///
    /// The poll keeps finding the same waiting line every couple of seconds, so without
    /// this the end of a scene with a stranded line prints the same warning for as long
    /// as the player stands there.
    /// </summary>
    private bool _playerNoTargetLogged;

    /// <summary>
    /// Ask the server whether anything has been typed.
    ///
    /// Fire-and-forget on purpose: this runs from the poll loop, which must not block on
    /// a loopback round trip, and a missed answer costs one tick of latency.
    /// </summary>
    private void PollPlayerInput()
    {
        if (!_cfg.PlayerInputEnabled || _llmClient is null) return;

        int every = Math.Max(1, _cfg.PlayerPollTicks);
        if (++_playerPollTick < every) return;
        _playerPollTick = 0;

        // Already asking. A server mid-generation can take seconds to answer anything, and
        // stacking polls behind it would turn one slow reply into a queue of them.
        if (Interlocked.CompareExchange(ref _playerPollBusy, 1, 0) != 0) return;

        _ = Task.Run(async () =>
        {
            try
            {
                _playerLineWaiting = await _llmClient.HasPlayerLineAsync();
            }
            finally
            {
                Volatile.Write(ref _playerPollBusy, 0);
            }
        });
    }

    /// <summary>
    /// Reopen the record the player is about to read, so it can answer what they typed.
    ///
    /// Returns true when a request was issued, which tells the pump to spend this tick
    /// here rather than on the lookahead. That ordering is the point: a tick that could
    /// do either should do the typed one, or the queue would be generating record
    /// <c>cursor + 7</c> while the line the player is waiting on sits in the store.
    /// </summary>
    private bool TryAnswerPlayerLine(SocialLinkSnapshot snap)
    {
        if (!_cfg.PlayerInputEnabled || !_playerLineWaiting) return false;

        int cursor = _poolNextRecord.Count > 0 ? Math.Max(0, _poolNextRecord[0]) : 0;
        int index  = Generatable.FindTypedLineTarget(_plan, cursor);

        if (index < 0)
        {
            // Nothing writable left ahead — the scene is over, or everything remaining
            // belongs to another speaker. The line stays in the store rather than being
            // collected and dropped; it will expire on its own if the next scene does not
            // want it.
            //
            // Said once. The poll finds the same line again every couple of seconds, and
            // this is a standing condition rather than an event.
            if (!_playerNoTargetLogged)
            {
                _modLog!.Info("[PLAYER] a line is waiting but no record ahead can answer it");
                _playerNoTargetLogged = true;
            }
            _playerLineWaiting = false;
            return false;
        }

        _playerNoTargetLogged = false;
        RecordPlan target = _plan[index];

        // One attempt per detected line. The flag is set again by the next poll if the
        // line is still there, which is also how a failed collection gets retried.
        _playerLineWaiting = false;

        string had = target.State == RecordState.Pending ? "unfilled"
                   : target.Generated is null ? target.State.ToString().ToLowerInvariant()
                   : $"holding \"{Trim(target.Generated, 28)}\"";

        _modLog!.Info($"[PLAYER] reopening #{target.Index} ({had}) to answer a typed line");

        target.State           = RecordState.InFlight;
        target.AnsweringPlayer = true;
        target.Attempts        = 0;   // a new question, not a retry of the old one

        RequestPlayerReply(snap, target);
        return true;
    }

    /// <summary>
    /// Collect the typed line and generate a reply to it, off the poll thread.
    ///
    /// The collection happens here rather than in the pump because it can fail: the line
    /// may have expired while a target was being chosen, or something else may have taken
    /// it. Reopening a record and only then discovering there is nothing to answer would
    /// leave it InFlight for the rest of the scene.
    /// </summary>
    private void RequestPlayerReply(SocialLinkSnapshot snap, RecordPlan record)
    {
        _ = Task.Run(async () =>
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(_cfg.TimeoutSeconds));
            try
            {
                string? said = await _llmClient!.TakePlayerLineAsync(cts.Token);
                if (said is null)
                {
                    // Gone between the peek and now. Hand the record back to the queue
                    // with whatever it already had: if it was Written, those bytes are
                    // still in the pool and still correct.
                    ReleaseToQueue(record, "the typed line was gone by the time we asked");
                    return;
                }

                _modLog!.Info($"[PLAYER] #{record.Index} answering: \"{Trim(said, 60)}\"");

                var req = new Server.GenerateRequest
                {
                    ConfidantId   = snap.ConfidantId,
                    Rank          = snap.RankLevel,
                    Context       = BuildRecordContext(snap, record),
                    CharacterName = Memory.ConfidantNames.Resolve(snap.ConfidantId),
                    MaxChars      = record.Capacity,

                    // Joker, not whatever the turn model worked out for this record.
                    //
                    // That is not an override of Ch. 81's rule, it is the rule: the
                    // addressee is the most recent earlier speaker, and the player has
                    // just become that by typing. A record whose planned addressee was
                    // the Girl's Father is still answering Joker if Joker is the one who
                    // spoke last.
                    Addressee     = _cfg.AddresseeEnabled ? Addressee.Protagonist : string.Empty,

                    // The one request that ever carries this. Everything else in the
                    // scene is pre-generation for records the player has not reached.
                    PlayerLine    = said,
                };

                string text = await _llmClient!.GenerateAsync(req, cts.Token);
                if (string.IsNullOrWhiteSpace(text))
                {
                    // The player's line is spent either way — it was collected, and the
                    // store no longer has it. Saying so matters: silence here looks
                    // identical to the mod never having noticed they typed.
                    ReleaseToQueue(record, "the server had no reply that fit the record");
                    return;
                }

                // Deliberately not run through RepeatsPreviousLine. That guard exists
                // because the model restates its own last line when it has nothing to
                // add; answering a direct question is the one case where echoing the
                // subject back is correct, and rejecting it would throw away the reply
                // and leave the player's line unanswered.
                record.Generated = text;
                record.State     = RecordState.Ready;
                _modLog!.Info($"[PLAYER] #{record.Index} reply ready " +
                              $"({text.Length}/{record.Capacity}): \"{text}\"");

                FlushReadyRecords();
            }
            catch (Server.InferenceInFlightException)
            {
                // The line is already collected, so it cannot be put back. Leaving the
                // record Pending means the ordinary queue fills it with a line that does
                // not answer anything, which is the honest outcome: the reply is lost and
                // the scene carries on.
                ReleaseToQueue(record, "server busy — the reply was lost");
            }
            catch (OperationCanceledException)
            {
                ReleaseToQueue(record, "timed out");
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                ReleaseToQueue(record, ex.Message);
            }
        });
    }

    /// <summary>
    /// Give a reopened record back to the ordinary queue.
    ///
    /// Pending rather than Rendered: the record is still unread, so the queue filling it
    /// with a normal line is better than leaving it on its scripted one. The reply is
    /// what was lost, not the record.
    /// </summary>
    private void ReleaseToQueue(RecordPlan record, string why)
    {
        record.AnsweringPlayer = false;
        record.State           = RecordState.Pending;
        _modLog!.Warn($"[PLAYER] #{record.Index} not answered: {why}");
    }

    /// <summary>Drop any line left in the store when a hang-out ends.</summary>
    private void ClearPlayerInput()
    {
        _playerLineWaiting    = false;
        _playerPollTick       = 0;
        _playerNoTargetLogged = false;

        if (!_cfg.PlayerInputEnabled || _llmClient is null) return;

        // Fire-and-forget: the session is already over and nothing waits on this. A line
        // typed into the end of one hang-out must not open the next one.
        _ = _llmClient.ClearPlayerLineAsync();
    }

    private static string Trim(string text, int max) =>
        text.Length <= max ? text : text[..max] + "...";
}
