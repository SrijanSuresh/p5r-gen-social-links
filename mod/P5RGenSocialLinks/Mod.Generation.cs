using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Reloaded.Hooks.Definitions;
using Reloaded.Hooks.Definitions.X64;
using Reloaded.Mod.Interfaces;
using Reloaded.Mod.Interfaces.Internal;
using P5RGenSocialLinks.Memory;
using P5RGenSocialLinks.Server;

namespace P5RGenSocialLinks;

/// <summary>
/// Asking the server for a line, and deciding which line to ask for next.
///
/// Generation is not triggered by the player arriving. Inference sustains about 0.7 lines
/// a second and a held fast-forward button advances at up to 1.7, so reacting cannot win;
/// the queue keeps several records ahead and writes seconds before the player gets there
/// (Ch. 72).
///
/// Each request carries the line it is replacing and the four before it, attributed to
/// whoever said them, which is what turns separate remarks into a conversation. The rules
/// here with teeth are the ones the prompt cannot enforce on its own: a restatement is
/// rejected by word overlap, and a record that fails the same way twice is given up on
/// rather than left occupying the queue for the whole scene.
/// </summary>
public partial class Mod
{
    /// <summary>
    /// Captures (offset, length) slots across a region using buffered reads. Strings that
    /// straddle a chunk boundary are dropped rather than stitched — at 256 KB chunks that
    /// is a negligible fraction, and it keeps every read bounds-checked.
    /// </summary>
    /// <summary>
    /// Renders the captured script as a context fragment, or empty if nothing was caught.
    /// </summary>
    /// <remarks>
    /// Framed as what the scene is about rather than as lines to imitate. Handing the
    /// model verbatim dialogue invites it to parrot a line back, which reads worse than
    /// generic filler — the goal is a reply that belongs in this conversation, not a
    /// paraphrase of one already in it.
    /// </remarks>
    private string ScriptContext()
    {
        if (_sceneScript.Count == 0) return "";
        return " The scene's original dialogue, for subject matter only — do not repeat " +
               "or paraphrase these lines: \"" +
               string.Join("\" \"", _sceneScript) + "\"";
    }

    /// <summary>
    /// Reads the scripted lines out of a pool region, in address order.
    /// </summary>
    /// <remarks>
    /// Must run before the first write to this region. The pool is the only place the
    /// scene's script exists in memory, and overwriting it is destructive — there is no
    /// second chance to read it, which is why this is called at arm time rather than
    /// lazily when the context is first needed.
    ///
    /// Lines are capped in count and total length: the context field is limited to 1024
    /// characters server-side, and a 36-slot scene at ~40 characters each would exceed it
    /// on its own and crowd out the rolling session history.
    /// </remarks>
    /// <summary>
    /// True when <paramref name="line"/> contains text this scene already generated.
    ///
    /// Containment rather than equality: a captured slot run is a fragment of a record's
    /// text, so a generated line shows up inside it rather than as the whole of it.
    /// </summary>
    private bool WasGenerated(string line)
    {
        foreach (RecordPlan record in _plan)
        {
            string? made = record.Generated;
            if (made is not null && made.Length > 0 && line.Contains(made)) return true;
        }
        return false;
    }

    private string[] CaptureSceneScript(nuint poolBase, int scanLen, int maxLines, int maxChars)
    {
        var lines = new System.Collections.Generic.List<string>();
        var seen  = new System.Collections.Generic.HashSet<string>();
        int used  = 0;

        for (int chunkOff = 0; chunkOff < scanLen && lines.Count < maxLines; chunkOff += ScanChunk)
        {
            int want = Math.Min(ScanChunk, scanLen - chunkOff);
            if (!MemoryGuard.TryRead(poolBase + (nuint)chunkOff, _scanBuf, want)) continue;

            foreach (var (_, text) in FindAsciiEnglishBuf(_scanBuf, want, poolBase + (nuint)chunkOff, 20000))
            {
                string line = text.Trim();
                if (line.Length < 8) continue;

                // Consecutive slots repeat the same line often enough that the raw list
                // wastes most of the budget on duplicates.
                if (!seen.Add(line)) continue;

                // Refuse anything we generated. The pool is the only copy of the script
                // and a write is destructive, so a capture that happens after the hook
                // path has already written would otherwise hand the model its own output
                // back as "the scene's original dialogue" and compound drift.
                //
                // Checked against the plan rather than a single cached string. The field
                // this used to read was only ever written by the reactive dispatch path,
                // which pre-generation replaced and Ch. 83 deleted - leaving a guard that
                // compiled, read as a safeguard, and could never fire.
                if (WasGenerated(line)) continue;

                lines.Add(line);
                used += line.Length + 1;
                if (lines.Count >= maxLines || used >= maxChars) return lines.ToArray();
            }
        }
        return lines.ToArray();
    }


    /// <summary>
    /// Context for one specific record: the scene, the line being replaced, and what has
    /// already been said.
    ///
    /// The line being replaced is the part the reactive path never had. It was sending
    /// "hang-out with Ryuji at Protein Lovers gym" for every line of the scene, so the
    /// model wrote ambience and repeated itself — two consecutive generations came back
    /// as "You comin' back here every week like me now?" and "You comin' here more often
    /// now?", which are the same sentence.
    /// </summary>
    private string BuildRecordContext(SocialLinkSnapshot snap, RecordPlan record)
    {
        var ctx = new System.Text.StringBuilder(ContextBuilder.Build(snap));

        ctx.Append(" The line you are replacing is: \"").Append(record.Original).Append('"');

        // "Same purpose" is right for pre-generation and wrong for a reply. The scripted
        // line was written for a conversation the player has just changed by typing into
        // it, so telling the model to preserve its purpose sets that instruction against
        // the typed line the prompt rules say outranks it — and a model given two
        // conflicting instructions obeys the nearer one (Ch. 88).
        ctx.Append(record.AnsweringPlayer
            ? " Joker has just said something to you directly. Answer him. The line above " +
              "is only there to show where the conversation had got to — you do not have " +
              "to keep its purpose."
            : " Say something with the same purpose, in your own words.");

        // What the player has actually heard, most recent last. Four lines rather than
        // two: with two, consecutive generations came back as "Yeah, it was a sweet gym"
        // followed by "You think you can just crash our gym session for free, huh?" —
        // each fine alone, and plainly not the same conversation.
        //
        // Generated text where it exists, the script where it does not. A record the
        // queue missed still played on screen, so leaving it out would describe a
        // conversation with a hole in it.
        // Drop the oldest lines until it fits, rather than truncating the finished string.
        // The server rejects anything over 1024 characters, and a tail cut would remove
        // "Continue from the last one." while keeping the least relevant history —
        // exactly the wrong end to lose.
        const int Budget = 1000;
        int from = Math.Max(0, record.Index - 4);

        // Whether this scene resolved speaker names at all. Without them every earlier
        // line has to be presented as the confidant's own, which is what the unattributed
        // path below does; with them, the history can name who said what.
        bool attributed = false;
        foreach (RecordPlan planned in _plan)
            if (planned.Speaker.Length > 0) { attributed = true; break; }

        while (true)
        {
            var history = new System.Text.StringBuilder();
            if (from < record.Index)
            {
                if (attributed) AppendCast(history, snap, record, from);
                else            AppendMonologue(history, record, from);
            }

            if (ctx.Length + history.Length <= Budget || from >= record.Index)
            {
                string full = ctx.Append(history).ToString();
                return full.Length > Budget ? full[..Budget] : full;
            }

            from++;   // give up the oldest line and try again
        }
    }

    /// <summary>
    /// Earlier lines with nobody's name on them, presented as the confidant's own.
    ///
    /// This is the shape the context had before speaker attribution existed, and it is
    /// still the right one when a scene's archive does not resolve: every line has to be
    /// attributed to somebody, and the confidant is the only name available.
    ///
    /// The wording is load-bearing. Calling this block "the conversation so far" made the
    /// model read the lines as the other person's and answer them — replacing Ryuji
    /// explaining the gym's pricing, it came back with "Fair enough, bro, that's a small
    /// price," which is a reply to himself.
    /// </summary>
    private void AppendMonologue(
        System.Text.StringBuilder history, RecordPlan record, int from)
    {
        history.Append(" You have just said, oldest first:");
        for (int i = from; i < record.Index; i++)
        {
            string said = _plan[i].Generated ?? _plan[i].Original;
            if (said.Length > 0) history.Append(" \"").Append(said).Append('"');
        }
        history.Append(" Carry on from your own last line — you are still talking, " +
                       "not replying to someone.");
    }

    /// <summary>
    /// Earlier lines with their speakers named — the scene as a script rather than a
    /// monologue.
    ///
    /// This is the half of attribution that improves lines rather than protecting them.
    /// The records the filter declines to rewrite are not obstacles; they are the other
    /// half of the conversation, and until now the mod had no way to say so. A Takemi
    /// line that follows her patient's father asking a question can now answer the
    /// question, instead of being a plausible sentence dropped into the gap.
    ///
    /// The closing instruction changes with the last speaker, because "carry on from your
    /// own last line" is exactly wrong when the last line was somebody else's.
    /// </summary>
    private void AppendCast(
        System.Text.StringBuilder history, SocialLinkSnapshot snap, RecordPlan record, int from)
    {
        history.Append(" The conversation so far, oldest first:");
        for (int i = from; i < record.Index; i++)
        {
            string said = _plan[i].Generated ?? _plan[i].Original;
            if (said.Length == 0) continue;

            history.Append(' ')
                   .Append(IsConfidantLine(snap, _plan[i]) ? "You" : SpeakerLabel(_plan[i]))
                   .Append(": \"").Append(said).Append('"');
        }

        RecordPlan previous = _plan[record.Index - 1];
        history.Append(IsConfidantLine(snap, previous)
            ? " Carry on from your own last line — you are still talking, not replying to someone."
            : $" Reply to what {SpeakerLabel(previous)} just said.");
    }

    private bool IsConfidantLine(SocialLinkSnapshot snap, RecordPlan record) =>
        ConfidantNames.IsSpokenBy(snap.ConfidantId, record.Speaker);

    /// A name for a line the prompt has to attribute to somebody. Unlabelled bubbles are
    /// common — narration, and anyone the scene has not introduced yet.
    private static string SpeakerLabel(RecordPlan record) =>
        record.Speaker.Length > 0 ? record.Speaker : "Someone else";

    /// <summary>
    /// True when a candidate line says what the previous one already said.
    ///
    /// The prompt forbids restating and the model does it anyway — observed consecutive:
    /// "Now I'm crushin' my reps like Makoto's got nothin' on me!" then "Man, I'm
    /// crushin' my reps like Makoto's got nothin' on me, damn!". A rule the model can
    /// ignore is not a guarantee, so this is the part with teeth.
    ///
    /// Word overlap rather than string distance, because the failure is semantic
    /// repetition dressed in different punctuation. Two thirds shared is the threshold: a
    /// genuine follow-up reuses the subject and little else, while a restatement reuses
    /// nearly everything. Short lines are exempt — "Hell yeah, dude!" twice in a scene is
    /// how the character talks, not a defect.
    /// </summary>
    private bool RepeatsPreviousLine(RecordPlan record, string candidate)
    {
        if (record.Index == 0 || candidate.Length < 20) return false;

        string? previous = _plan[record.Index - 1].Generated;
        if (previous is null || previous.Length < 20) return false;

        var earlier = new System.Collections.Generic.HashSet<string>(
            previous.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries));
        if (earlier.Count == 0) return false;

        string[] words = candidate.ToLowerInvariant()
                                  .Split(' ', StringSplitOptions.RemoveEmptyEntries);
        string[] prior = previous.ToLowerInvariant()
                                 .Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (words.Length == 0) return false;

        // Signal one: most of the words were already used.
        int shared = 0;
        foreach (string w in words) if (earlier.Contains(w)) shared++;
        if (shared * 3 >= words.Length * 2) return true;

        // Signal two: it opens the same way. Overlap alone missed "Gonna be seein' me
        // sweat stains all over 'em" followed by "Gonna be seein' those stains all over
        // my damn gym shorts too" — only half the words match, and the repetition is
        // obvious on screen because the first three are identical.
        return words.Length >= 3 && prior.Length >= 3
            && prior[0] == words[0] && prior[1] == words[1] && prior[2] == words[2];
    }

    /// <summary>
    /// Ask the server for one record's replacement line, off the poll thread.
    ///
    /// The record is passed rather than looked up on completion: by the time the answer
    /// arrives the player may have moved on, the plan may have been rebuilt, or the
    /// session may have ended, and resolving an index against a list that has since
    /// changed is how a generated line lands on the wrong bubble.
    /// </summary>
    private void RequestForRecord(SocialLinkSnapshot snap, RecordPlan record)
    {
        _ = Task.Run(async () =>
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(_cfg.TimeoutSeconds));
            try
            {
                var req = new Server.GenerateRequest
                {
                    ConfidantId   = snap.ConfidantId,
                    Rank          = snap.RankLevel,
                    Context       = BuildRecordContext(snap, record),
                    CharacterName = Memory.ConfidantNames.Resolve(snap.ConfidantId),
                    MaxChars      = record.Capacity,
                    Addressee     = _cfg.AddresseeEnabled ? record.Addressee : string.Empty,
                };

                string text = await _llmClient!.GenerateAsync(req, cts.Token);
                if (string.IsNullOrWhiteSpace(text))
                {
                    // Empty means the server had nothing that fit — usually a short record
                    // no complete sentence will go into. Retrying that produces the same
                    // answer while holding a queue slot the whole scene, so it gets a
                    // couple of chances and then keeps its scripted line.
                    GiveUpOrRetry(record, "no line fit the record");
                    return;
                }

                if (RepeatsPreviousLine(record, text))
                {
                    _modLog!.Info($"[PREGEN] #{record.Index} rejected as a restatement: \"{text}\"");
                    GiveUpOrRetry(record, "restated the previous line");
                    return;
                }

                record.Generated = text;
                record.State     = RecordState.Ready;
                _modLog!.Info($"[PREGEN] #{record.Index} ready ({text.Length}/{record.Capacity}): \"{text}\"");

                // Write it now rather than on the next poll tick. Waiting added up to a
                // full 500ms between a line existing and it reaching memory, on top of
                // ~1.5s of inference — and that delay lands exactly where it hurts, at the
                // front of a scene where the queue has not banked any buffer yet.
                FlushReadyRecords();
            }
            catch (Server.InferenceInFlightException)
            {
                // Put it back for the next tick. Logged rather than swallowed: this was
                // silent, and a scene where 57 of 78 requests were dropped looked from
                // the mod's side like generation simply not happening.
                record.State = RecordState.Pending;
                _modLog!.Warn($"[PREGEN] #{record.Index} dropped — server busy.");
            }
            catch (OperationCanceledException)
            {
                GiveUpOrRetry(record, "timed out");
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                // Logged here, not only when the retries run out. A backend that was down
                // for an entire scene produced no line and no explanation: the queue looked
                // idle, and the only trace was a thirty-second gap between two requests.
                // Same defect as the 429 storm, same handler (Ch. 80).
                _modLog!.Warn($"[PREGEN] #{record.Index} failed: {ex.Message}");
                GiveUpOrRetry(record, ex.Message);
            }
        });
    }

    /// <summary>
    /// Put a failed record back in the queue, or stop trying.
    ///
    /// A busy server and a record no sentence will fit fail the same way here, and only
    /// one of them is worth retrying. Two attempts covers the transient case; beyond that
    /// the record keeps its scripted line, which is a good line rather than a missing one.
    /// </summary>
    private void GiveUpOrRetry(RecordPlan record, string why)
    {
        const int MaxAttempts = 2;

        if (++record.Attempts >= MaxAttempts)
        {
            record.State = RecordState.Rendered;   // out of the queue for good
            _modLog!.Info($"[PREGEN] #{record.Index} keeping the script after " +
                          $"{record.Attempts} attempts ({why})");
            return;
        }
        record.State = RecordState.Pending;
    }

    /// <summary>
    /// Keep the next few records generated, without waiting for the player to reach them.
    ///
    /// Runs on the poll tick. Everything it needs was known at arm time, so this asks a
    /// question the reactive path could not: not "what should Ryuji say now" but "what
    /// should he say at record 14", answerable long before record 14 is on screen.
    ///
    /// One request is issued per tick at most. The server processes one at a time and
    /// answers 429 to anything overlapping, so firing the whole window at once would
    /// convert a queue into a pile of rejections.
    /// </summary>
    private void PumpPregen(SocialLinkSnapshot snap)
    {
        int lookahead = _cfg.PregenLookahead;
        if (lookahead <= 0 || _plan.Count == 0) return;

        ApplySpeakerFilter(snap);

        // One request at a time, because the server runs one at a time. It answers 429 to
        // anything overlapping, and the poll tick is 500ms against 2-3s of inference — so
        // four requests in five were born to be dropped. The server counted 78 requests,
        // 57 drops and 14 completions across one scene, and the mod never saw a single
        // failure because a 429 was handled silently.
        //
        // Issuing one and waiting is not slower. The rejected four were never going to
        // produce anything; they only churned state and hid the real throughput.
        foreach (RecordPlan inFlight in _plan)
            if (inFlight.State == RecordState.InFlight) return;

        // A line the player typed outranks the lookahead. Served before the loop below so
        // a tick that could do either does this one — otherwise the queue spends it on
        // record cursor+7 while the reply the player is waiting for sits in the store
        // (Ch. 88).
        //
        // It waits for any in-flight request like everything else. That costs up to about
        // two seconds, which is worth not plumbing a cancellation token out of
        // RequestForRecord for: the player has just spent five to fifteen seconds typing.
        if (TryAnswerPlayerLine(snap)) return;

        int from = _poolNextRecord.Count > 0 ? Math.Max(0, _poolNextRecord[0]) : 0;
        int to   = Math.Min(_plan.Count, from + lookahead);

        for (int i = from; i < to; i++)
        {
            RecordPlan record = _plan[i];
            if (record.State != RecordState.Pending) continue;

            // Selections and records too short to hold a sentence are taken out of the
            // queue for good. A selection must never be rewritten at all (Ch. 80), and a
            // three-character record is a fragment of UI — replacing it spends a slow
            // inference on text nobody reads as dialogue.
            //
            // The thresholds live in Generatable because the typed-input path applies the
            // identical test, and a second copy of a threshold is one that drifts.
            if (!Generatable.IsWorthGenerating(record))
            {
                record.State = RecordState.Rendered;   // permanently out of the way
                continue;
            }

            record.State = RecordState.InFlight;

            // Logged at issue, not only on arrival. Without both timestamps a slow line
            // is indistinguishable from a line that was never asked for, and a 19-second
            // gap between arming and the first result could have been either.
            _modLog!.Info($"[PREGEN] #{record.Index} requested (cap={record.Capacity})");
            RequestForRecord(snap, record);
            return;   // one per tick
        }
    }

    /// <summary>
    /// Start the queue immediately after arming, without waiting for the next poll tick.
    ///
    /// The front of a scene is where coverage is always lost — #3 through #6 in the last
    /// three sessions — because the player is reading at full speed while the queue has
    /// banked nothing. Half a second of idling before the first request is spent at the
    /// exact moment it can least be afforded.
    /// </summary>
    private void KickPregen()
    {
        if (_cfg.PregenLookahead <= 0 || _plan.Count == 0) return;
        if (!_reader!.TryResolve(out nuint session)) return;

        SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(session);
        if (snap is not null) PumpPregen(snap);
    }
}
