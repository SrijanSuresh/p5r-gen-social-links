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
/// What a scene contains: every record, its capacity, its scripted line, who speaks it
/// and who it is spoken to.
///
/// All of it is known when the scene is armed, which is the point. Waiting for a dispatch
/// to discover a record's capacity was only ever necessary while the buffer was being
/// found by heuristic, and pre-generation needs the records the player has *not* reached
/// (Ch. 72).
///
/// The archive is parsed rather than guessed at: the dialogue table locates every message,
/// each header carries a speaker id, the speaker table resolves it to a name (Ch. 79), and
/// the order those speakers take their turns gives each line an addressee (Ch. 81).
/// </summary>
public partial class Mod
{
    /// <summary>
    /// Report how much of a scene was actually replaced, and which records were not.
    ///
    /// Coverage is the number that says whether this works, and it had to be counted by
    /// hand from the log until now — 16 of 22, with the gaps at 3-6 and 14-15. Both gaps
    /// were bursts where the player advanced faster than inference could keep up, which is
    /// a fact about pace rather than a fault, and it is invisible without this line.
    /// </summary>
    private void LogSceneCoverage()
    {
        if (_plan.Count == 0) return;

        int written = 0, pending = 0, tooSmall = 0, foreign = 0, menus = 0, answered = 0;
        var gaps = new System.Text.StringBuilder();
        foreach (RecordPlan record in _plan)
        {
            bool replaced = record.WasWritten;
            if (replaced)
            {
                written++;
                if (record.AnsweringPlayer) answered++;
                continue;
            }
            if (record.Original.Length < 8) continue;   // never a candidate

            // A menu is not a line we failed to replace. "Yes, I'm sure!" is fourteen
            // characters in a roomy slot, so without this it cleared both the length and
            // capacity floors and was counted as a miss.
            if (record.IsSelection) { menus++; continue; }

            // Another character's line is not a miss. Counting it as one would report a
            // Takemi scene shared with a patient and her father as half-failed, when
            // leaving those lines alone is the whole point of the filter.
            if (record.State == RecordState.Foreign) { foreign++; continue; }

            // Records we deliberately never queue are not failures, and counting them as
            // misses understated the result: the first report read 17/22 when five of the
            // gaps were short interjections the queue was told to leave alone.
            if (record.Capacity < 24) { tooSmall++; continue; }

            pending++;
            if (gaps.Length < 60) gaps.Append(gaps.Length > 0 ? ", " : "").Append('#').Append(record.Index);
        }

        int candidates = written + pending;
        int percent    = candidates == 0 ? 0 : written * 100 / candidates;
        _modLog!.Info($"[SCENE] replaced {written}/{candidates} records ({percent}%)" +
                      (foreign  > 0 ? $", {foreign} left to other speakers" : "") +
                      (answered > 0 ? $", {answered} answering the player" : "") +
                      (menus    > 0 ? $", {menus} selections left alone" : "") +
                      (tooSmall > 0 ? $", {tooSmall} too short to try" : "") +
                      (gaps.Length > 0 ? $" — missed {gaps}" : ""));
    }

    /// Index the player is currently at, as far as the interpreter has told us.
    private int CurrentRecord() =>
        _poolNextRecord.Count > 0 ? Math.Max(0, _poolNextRecord[0] - 1) : 0;

    /// <summary>
    /// Take the records this scene attributes to somebody else out of the queue.
    ///
    /// Runs on every pump but does work once: only Pending records can move to Foreign,
    /// so the second pass finds nothing left to do.
    ///
    /// The audit line is printed whether or not the filter is on, because deciding to turn
    /// it on needs the same numbers the filter acts on. A scene that reports "9 own, 7
    /// other, 0 unattributed" is one this is ready for; "0 own, 0 other, 16 unattributed"
    /// means the archive did not resolve and turning the switch on would silence the mod
    /// completely.
    /// </summary>
    private void ApplySpeakerFilter(SocialLinkSnapshot snap)
    {
        // Once per plan. Both halves are decided by data fixed at arm time, so repeating
        // them on a 500 ms tick would re-tokenize every name in the scene twice a second
        // to reach the same answer — the kind of per-tick cost that has been felt as
        // stutter before.
        if (_speakerAudited || _plan.Count == 0) return;
        _speakerAudited = true;

        int mine = 0, others = 0, unattributed = 0;
        foreach (RecordPlan record in _plan)
        {
            if (record.Speaker.Length == 0) unattributed++;
            else if (ConfidantNames.IsSpokenBy(snap.ConfidantId, record.Speaker)) mine++;
            else others++;
        }

        _modLog!.Info($"[SPEAKER] {ConfidantNames.Resolve(snap.ConfidantId)}: " +
                      $"{mine} own, {others} other, {unattributed} unattributed" +
                      (_cfg.SpeakerFilterEnabled ? "" : " (filter off — nothing skipped)"));

        if (!_cfg.SpeakerFilterEnabled) return;

        // Refuse to filter a scene the confidant appears to say nothing in.
        //
        // That is not a scene — it is a broken chain somewhere between the header and the
        // name table, and acting on it would replace "every line in the wrong voice" with
        // "no lines at all". Ch. 77 calls for failing closed on a single record; failing
        // closed on all of them is a different decision and it should not be silent.
        if (mine == 0)
        {
            _modLog!.Info("[SPEAKER] filter idle — no record attributed to the confidant");
            return;
        }

        int skipped = 0;
        foreach (RecordPlan record in _plan)
        {
            if (record.State != RecordState.Pending) continue;
            if (ConfidantNames.IsSpokenBy(snap.ConfidantId, record.Speaker)) continue;

            record.State = RecordState.Foreign;
            skipped++;
        }

        if (skipped > 0)
            _modLog!.Info($"[SPEAKER] left {skipped} records to their own speakers");
    }

    /// <summary>
    /// Build the scene plan from region 0: one entry per record, with its capacity and
    /// the scripted line it currently holds.
    ///
    /// Ordering is load-bearing in the same way as Ch. 64's script capture. The original
    /// text only exists until the first write, and it is the best context the model can
    /// be given about this specific moment — not the scene in general, but the line it is
    /// replacing. Reading it later means reading whatever we ourselves wrote.
    /// </summary>
    private void BuildRecordPlan()
    {
        _plan.Clear();
        _speakerAudited = false;
        if (_heapPools.Count == 0 || _poolRecords.Count == 0) return;

        (nuint poolBase, _, (int Off, int Len)[] slots) = _heapPools[0];
        var records = _poolRecords[0];

        // Before the loop, because every record is attributed through it. Declared up
        // front rather than inline: the short circuit on an empty pool would otherwise
        // leave the outputs unassigned.
        BmdArchive archive  = default;
        nuint      fileAddr = 0;
        bool haveArchive = records.Count > 0 &&
            TryResolveArchive(poolBase, slots[records[0].Start].Off, out archive, out fileAddr);

        for (int i = 0; i < records.Count; i++)
        {
            (int start, int count) = records[i];

            int capacity = 0;
            var line     = new System.Text.StringBuilder();
            for (int k = 0; k < count; k++)
            {
                (int off, int len) = slots[start + k];
                capacity += len;

                // Rows of one bubble join with a space, not a newline: the model is being
                // shown a sentence, and the row split is a rendering detail it has no use
                // for. Wrapping puts the breaks back on the way out.
                if (line.Length > 0) line.Append(' ');
                line.Append(AsciiPreview(poolBase + (nuint)off, len));
            }

            // Who says it. The archive's dialogue table already lists every message and
            // where its text starts, so this is containment rather than a search: the run
            // the scanner found belongs to the last message beginning at or before it
            // (Ch. 79).
            string name      = string.Empty;
            string speaker   = string.Empty;
            int    speakerId = BmdMessage.NoSpeaker;
            bool   selection = false;

            if (haveArchive)
            {
                int fileOffset = (int)(poolBase + (nuint)slots[start].Off - fileAddr);
                if (archive.TryFindByTextOffset(fileOffset, out BmdArchive.Entry entry))
                {
                    name      = entry.Name;
                    speakerId = entry.SpeakerId;
                    selection = entry.IsSelection;
                    archive.TryGetSpeakerName(speakerId, out speaker);
                }
            }

            _plan.Add(new RecordPlan
            {
                Index       = i,
                Capacity    = capacity + (count - 1),
                Original    = line.ToString().Trim(),
                Name        = name,
                SpeakerId   = speakerId,
                Speaker     = speaker,
                IsSelection = selection,
            });
        }

        ResolveAddressees();

        LogSlotSeparators(poolBase, slots);
        LogSpeakerCensus();

        _modLog!.Info($"[PLAN] {_plan.Count} records; " +
                      $"capacity {MinCapacity()}-{MaxCapacity()} chars");
        for (int i = 0; i < Math.Min(4, _plan.Count); i++)
            _modLog!.Info($"[PLAN]   {_plan[i]}");
    }

    /// <summary>
    /// Locate and parse the MSG1 archive the armed pool's records belong to.
    ///
    /// The backwards walk is deliberately not bounded by the pool region. VirtualQuery
    /// hands back the region containing an address, not the allocation, and two arms gave
    /// up after "no MSG1 header within 116B of the first record" because the file started
    /// before the region did. The span halves on a failed read instead, so the search runs
    /// as far back as memory allows and no further.
    ///
    /// Every step may fail, and failing costs nothing: no archive means no names, which
    /// leaves the plan exactly as it was before any of this existed.
    /// </summary>
    private bool TryResolveArchive(
        nuint poolBase, int firstTextOff, out BmdArchive archive, out nuint fileAddr)
    {
        const int LookBack = 0x10000;

        archive  = default;
        fileAddr = 0;

        nuint   textAddr = poolBase + (nuint)firstTextOff;
        byte[]? window   = null;
        int     span     = LookBack;

        while (span >= BmdMessage.HeaderBytes)
        {
            if ((ulong)textAddr > (ulong)span)
            {
                var buf = new byte[span];
                if (MemoryGuard.TryRead(textAddr - (nuint)span, buf, span)) { window = buf; break; }
            }
            span /= 2;
        }
        if (window is null) return false;

        if (!BmdArchive.TryFindMagicBefore(window, span - 1, out int magicIndex))
        {
            _modLog!.Info($"[SPEAKER] no MSG1 header within {span}B of the first record");
            return false;
        }
        if (!BmdArchive.TryReadFileSize(window, magicIndex, out int fileSize))
        {
            _modLog!.Info("[SPEAKER] MSG1 header found but its declared size is implausible");
            return false;
        }

        nuint addr = textAddr - (nuint)span + (nuint)magicIndex - 8;
        var   file = new byte[fileSize];
        if (!MemoryGuard.TryRead(addr, file, fileSize))
        {
            _modLog!.Info($"[SPEAKER] MSG1 at 0x{addr:X} declared {fileSize}B, unreadable");
            return false;
        }

        LogArchiveDump(addr, file, (int)(textAddr - addr));

        if (!BmdArchive.TryParse(file, out archive))
        {
            _modLog!.Info($"[SPEAKER] MSG1 at 0x{addr:X} ({fileSize}B) did not resolve");
            return false;
        }

        fileAddr = addr;
        _modLog!.Info($"[SPEAKER] MSG1 at 0x{addr:X}: {archive.DialogCount} messages, " +
                      $"{archive.SpeakerCount} speakers, dialogue ends at file+0x" +
                      $"{archive.DialogueEnd:X}" +
                      (archive.HoleCount > 0 ? $", {archive.HoleCount} unreadable" : "") +
                      (BmdArchive.IsRelocated(file) ? "" : ", NOT relocated"));

        // An entry that still does not parse is a hypothesis that has not survived, and the
        // bytes are the only thing that will say which one. Printing them costs three lines
        // and saves a play session.
        if (archive.HoleCount > 0 && archive.FirstHoleOffset >= 0)
        {
            _modLog!.Info($"[SPEAKER] first unreadable entry #{archive.FirstHole} " +
                          $"at file+0x{archive.FirstHoleOffset:X}");
            LogHexBlock("hole", file, archive.FirstHoleOffset, 0x30);
        }

        for (int i = 0; i < Math.Min(16, archive.SpeakerCount); i++)
            if (archive.TryGetSpeakerName(i, out string name))
                _modLog!.Info($"[SPEAKER]   [{i}] {name}");

        return true;
    }

    /// <summary>
    /// Dump the three parts of the archive that decide what its interior looks like.
    ///
    /// Two hypotheses were offered for this layout before one hex dump settled it (Ch. 79).
    /// The dump stays: it costs a dozen log lines once per armed scene, and it is what any
    /// future disagreement between the parser and the game will be settled with.
    /// </summary>
    private void LogArchiveDump(nuint fileAddr, byte[] file, int firstTextOffset)
    {
        if (!_cfg.BmdDumpEnabled) return;

        _modLog!.Info($"[BMDHEX] file at 0x{fileAddr:X}, {file.Length}B, " +
                      $"first text at file+0x{firstTextOffset:X}");

        LogHexBlock("head", file, 0, 0x60);

        int dialogCount = file[0x18] | (file[0x19] << 8) | (file[0x1A] << 16) | (file[0x1B] << 24);
        _modLog!.Info($"[BMDHEX] dialogCount@0x18 = {dialogCount} " +
                      $"-> speaker table at file+0x{0x20 + 8 * dialogCount:X}");
        if (dialogCount > 0 && 0x20 + 8 * dialogCount + 0x30 < file.Length)
            LogHexBlock("spk ", file, 0x20 + 8 * dialogCount, 0x30);

        int before = Math.Max(0, firstTextOffset - 0x40);
        LogHexBlock("pre ", file, before, Math.Min(0x50, file.Length - before));

        int tail = Math.Max(0, file.Length - 0x140);
        LogHexBlock("tail", file, tail, file.Length - tail);
    }

    /// <summary>Sixteen bytes per line, hex then ASCII, offsets relative to the file.</summary>
    private void LogHexBlock(string tag, byte[] file, int start, int length)
    {
        for (int off = start; off < start + length && off < file.Length; off += 16)
        {
            var hex   = new System.Text.StringBuilder(48);
            var ascii = new System.Text.StringBuilder(16);

            for (int i = 0; i < 16; i++)
            {
                if (off + i >= file.Length) { hex.Append("   "); continue; }
                byte b = file[off + i];
                hex.Append(b.ToString("X2")).Append(' ');
                ascii.Append(b >= 0x20 && b <= 0x7E ? (char)b : '.');
            }
            _modLog!.Info($"[BMDHEX] {tag} +{off:X4}  {hex} |{ascii}|");
        }
    }

    /// <summary>
    /// Work out who each line is spoken to, for the whole scene, at arm time.
    ///
    /// Once per plan rather than once per request. The answer depends only on the speaker
    /// ids and the record kinds, both fixed when the scene is armed, so recomputing it per
    /// generation would walk the plan backwards on every line to reach the same string.
    ///
    /// The log line is the evidence. A solo hang-out should print one addressee for the
    /// whole scene and a rank-up should print the cast changing hands; a scatter means the
    /// turn model is reading the speaker table wrong, and that is visible here before it
    /// is visible in a generated line.
    /// </summary>
    private void ResolveAddressees()
    {
        if (_plan.Count == 0) return;

        var turns = new Addressee.Turn[_plan.Count];
        for (int i = 0; i < _plan.Count; i++)
            turns[i] = new Addressee.Turn(_plan[i].SpeakerId, _plan[i].Speaker,
                                          _plan[i].IsSelection);

        var counts = new System.Collections.Generic.Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < _plan.Count; i++)
        {
            string to = Addressee.Resolve(turns, i);
            _plan[i].Addressee = to;

            string key = to.Length > 0 ? to : "(unnamed)";
            counts.TryGetValue(key, out int n);
            counts[key] = n + 1;
        }

        var summary = new System.Text.StringBuilder();
        foreach (var pair in counts)
        {
            if (summary.Length > 0) summary.Append(", ");
            summary.Append(pair.Key).Append(" x").Append(pair.Value);
        }
        _modLog!.Info($"[ADDRESSEE] {summary}");
    }

    /// <summary>
    /// Report which speakers the scene contains, and how much of it we could attribute.
    ///
    /// The census is the evidence, not the feature. If the layout in Ch. 75 is right this
    /// prints a handful of ids in script order with names beside them; if it is wrong it
    /// prints a scatter of one-offs, or nothing at all, and no behaviour has changed yet
    /// either way. Nothing keys off SpeakerId until a real scene has produced this line.
    /// </summary>
    private void LogSpeakerCensus()
    {
        var counts  = new System.Collections.Generic.Dictionary<int, int>();
        int unknown = 0;

        foreach (RecordPlan record in _plan)
        {
            if (record.Name.Length == 0) { unknown++; continue; }
            counts.TryGetValue(record.SpeakerId, out int n);
            counts[record.SpeakerId] = n + 1;
        }

        var parts = new System.Collections.Generic.List<string>();
        foreach (var pair in counts)
            parts.Add($"{(pair.Key == BmdMessage.NoSpeaker ? "narration" : $"spk{pair.Key}")}" +
                      $"x{pair.Value}");

        _modLog!.Info($"[SPEAKER] {_plan.Count - unknown}/{_plan.Count} records attributed; " +
                      $"{(parts.Count > 0 ? string.Join(", ", parts) : "none")}");

        for (int i = 0; i < Math.Min(6, _plan.Count); i++)
            _modLog!.Info($"[SPEAKER]   #{i} {(_plan[i].Name.Length > 0 ? _plan[i].Name : "?")} " +
                          $"spk={_plan[i].SpeakerId}");
    }

    private int MinCapacity()
    {
        int min = int.MaxValue;
        foreach (var p in _plan) min = Math.Min(min, p.Capacity);
        return min == int.MaxValue ? 0 : min;
    }

    private int MaxCapacity()
    {
        int max = 0;
        foreach (var p in _plan) max = Math.Max(max, p.Capacity);
        return max;
    }

}
