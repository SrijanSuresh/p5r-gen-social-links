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
/// Arming a region, following the player through it, and writing bytes into it.
///
/// This is the part that touches the game's memory, so every rule in it was learned the
/// expensive way. A region is armed from an address the interpreter just read rather than
/// from a scan (Ch. 65). Each line is held in two copies at a fixed offset, and the mirror
/// write verifies its target byte-for-byte first, so a wrong twin delta writes nothing
/// (Ch. 66). A record is writable until the interpreter reads it and frozen afterwards,
/// which began as the bug the player described as "the text keeps switching" (Ch. 71).
/// </summary>
public partial class Mod
{
    private (int Off, int Len)[] CapturePoolSlotsSafe(nuint poolBase, int scanLen)
    {
        var slots = new System.Collections.Generic.List<(int, int)>();

        for (int chunkOff = 0; chunkOff < scanLen && slots.Count < 20000; chunkOff += ScanChunk)
        {
            int want = Math.Min(ScanChunk, scanLen - chunkOff);
            if (!MemoryGuard.TryRead(poolBase + (nuint)chunkOff, _scanBuf, want)) continue;

            foreach (var (addr, text) in FindAsciiEnglishBuf(_scanBuf, want, poolBase + (nuint)chunkOff, 20000))
            {
                slots.Add(((int)(addr - poolBase), text.Length));
                if (slots.Count >= 20000) break;
            }
        }
        return slots.ToArray();
    }

    /// <summary>
    /// Print absolute addresses of the anchor's first slots, for setting a hardware
    /// read watchpoint on them in a debugger.
    ///
    /// The ranking tells us which region looks most like dialogue; it cannot tell us
    /// which slot the renderer actually reads, which is why every slot gets written and
    /// both rows of the bubble show the same line. That question is answerable only by
    /// observing the consumer (learning.md Ch. 65), and observing it starts with an
    /// address to watch.
    ///
    /// Hunting that address by string-scanning the process is possible but slow and
    /// ambiguous — the same line exists in several copies. The mod already holds the
    /// exact offsets, so printing them turns a search into a paste.
    ///
    /// The list has to cover the whole region, not just the head. A watchpoint catches
    /// only reads that happen after it is armed, and the renderer reads a slot once, on
    /// the transition into that line — so the useful target is always a line that has
    /// not been displayed yet. Logging the first 8 slots meant the player had to hand-read
    /// memory as soon as the scene got past them.
    /// </summary>
    private void LogWatchpointTargets(nuint poolBase, (int Off, int Len)[] slots, int maxSlots)
    {
        int n = Math.Min(maxSlots, slots.Length);
        _modLog!.Info($"[SLOTS] anchor 0x{poolBase:X} — first {n} of {slots.Length} slots:");
        for (int i = 0; i < n; i++)
        {
            (int off, int len) = slots[i];
            nuint addr = poolBase + (nuint)off;

            // Read the text back rather than reusing the capture-time string: this runs
            // before any write, so it is the scripted line, and it must match what is on
            // screen for the address to be worth watching.
            string text = AsciiPreview(addr, Math.Min(len, 64));
            _modLog!.Info($"[SLOTS]   0x{addr:X} len={len} \"{text}\"");
        }
    }

    /// <summary>
    /// Writes <paramref name="text"/> into every armed heap region. Same in-place rules as
    /// the single-pool writer: never exceed a slot's original run length, and pad with
    /// spaces rather than a NUL so the surrounding function codes stay intact.
    /// </summary>
    /// <summary>
    /// Returns how many bytes of <paramref name="enc"/> to write into a slot of
    /// <paramref name="slotLen"/> bytes, breaking on a word boundary rather than
    /// mid-word.
    /// </summary>
    /// <remarks>
    /// The renderer reads the slot in place and the write cannot exceed the original
    /// line's length, so a long generation is clipped rather than wrapped. A raw
    /// Math.Min cut produced "…you're seriously the onl" on screen — text that reads
    /// as a bug rather than as a shortened line.
    ///
    /// Backing up to the last space costs a word but always ends cleanly. The 60%
    /// floor guards the degenerate case: if the only space sits near the start, a
    /// word-boundary cut would throw away most of the slot, and a hard cut carries
    /// more of the sentence.
    /// </remarks>
    /// <summary>
    /// Group slots that are one byte apart into a single record.
    ///
    /// The ASCII scanner reports printable runs, and it splits on any non-printable byte
    /// — including 0x0A. A two-line speech bubble therefore arrives as two "slots" that
    /// were never two strings: they are one message with a line break in the middle,
    /// stored contiguously. Measured live:
    ///
    ///   0x4250B2FBC7 len=33  "The equipment's kinda crappy, but"
    ///   0x4250B2FBE9 len=25  "they got tons of variety."
    ///
    /// 0xFBC7 + 33 = 0xFBE8, and the next run starts at 0xFBE9. Exactly one byte between
    /// them, and that byte is the newline. Where a genuinely new record starts the gap is
    /// tens of bytes of header instead.
    ///
    /// This is the fix for the oldest cosmetic bug in the mod: writing each fragment
    /// independently put the whole generated line in row one and the same line again in
    /// row two. Rows are not independent, so they cannot be written independently.
    /// </summary>
    private System.Collections.Generic.List<(int Start, int Count)>
        GroupSlotsIntoRecords(nuint poolBase, (int Off, int Len)[] slots)
    {
        var records = new System.Collections.Generic.List<(int, int)>();
        var gap     = new byte[MaxJoinGap];
        int i = 0;
        while (i < slots.Length)
        {
            int start = i++;
            while (i < slots.Length && SameRecord(poolBase, slots[i - 1], slots[i], gap)) i++;
            records.Add((start, i - start));
        }
        return records;
    }

    /// Add printable runs of three or more characters found between two fragments.
    private void AddGapRuns(nuint poolBase, (int Off, int Len) prev, (int Off, int Len) next,
                            byte[] buf, System.Collections.Generic.List<(int, int)> slots)
    {
        int gapStart = prev.Off + prev.Len;
        int gap      = next.Off - gapStart;
        if (gap <= 1 || gap > MaxJoinGap) return;
        if (!MemoryGuard.TryRead(poolBase + (nuint)gapStart, buf, gap)) return;

        int run = 0;
        for (int b = 0; b <= gap; b++)
        {
            bool printable = b < gap && IsPrintable(buf[b]);
            if (printable) { run++; continue; }
            if (run >= 3) slots.Add((gapStart + b - run, run));
            run = 0;
        }
    }

    /// <summary>
    /// Whether two adjacent text runs belong to the same speech bubble.
    ///
    /// Decided by what is in the gap, not by how wide it is. Measured on a live scene:
    ///
    ///   27B  0A F2 23 00 00 F1 21 F2 05 FF FF F1 41 F7 61 09 ...   message boundary
    ///   12B  0A 79 6F 75 20 67 6F 74 74 61 83 D2                   same bubble
    ///
    /// The second is a newline, the word "you gotta", and a two-byte Shift-JIS dash — no
    /// control codes at all. The first is a block of them: F2 23, F1 21, F2 05 FF FF,
    /// F1 41, F7.
    ///
    /// The previous rule joined runs exactly one byte apart, so it split that bubble in
    /// two, wrote the generated line into the first half, and left "you gotta— Wait, that
    /// ain't it!" showing underneath. Widening it to twelve would have worked on this
    /// scene and been luck: 12 against 27 is a property of these two samples, while the
    /// presence of an F1/F2 control block is a property of the format.
    /// </summary>
    private bool SameRecord(nuint poolBase, (int Off, int Len) prev, (int Off, int Len) next,
                            byte[] buf)
    {
        int gapStart = prev.Off + prev.Len;
        int gap      = next.Off - gapStart;

        if (gap < 1) return false;
        if (gap == 1) return true;              // the bare newline between two rows
        if (gap > MaxJoinGap) return false;     // far enough apart to be padding

        if (!MemoryGuard.TryRead(poolBase + (nuint)gapStart, buf, gap)) return false;
        foreach (byte b in new System.ReadOnlySpan<byte>(buf, 0, gap))
            if (b is 0xF1 or 0xF2 or 0xF7) return false;   // BMD function code: new message
        return true;
    }

    /// <summary>
    /// Distribute <paramref name="text"/> across fragments of the given widths, breaking
    /// on word boundaries.
    ///
    /// Each fragment is a fixed-length slot the renderer reads in place, so this is not
    /// free-form wrapping: fragment k may hold at most widths[k] bytes, and any fragment
    /// the text does not reach must come back empty so the caller can blank it. A bubble
    /// whose second row still held the original dialogue would be worse than one that is
    /// simply short.
    /// </summary>
    private static string[] WrapAcrossFragments(string text, int[] widths)
    {
        var lines = new string[widths.Length];
        string[] words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int w = 0;

        for (int f = 0; f < widths.Length; f++)
        {
            var line = new System.Text.StringBuilder();
            while (w < words.Length)
            {
                int need = line.Length == 0 ? words[w].Length : line.Length + 1 + words[w].Length;
                if (need > widths[f]) break;
                if (line.Length > 0) line.Append(' ');
                line.Append(words[w]);
                w++;
            }

            // A single word wider than the fragment would never fit, so the loop above
            // would leave this fragment empty and every later one empty too — the whole
            // line lost to one long word. Hard-cut it instead.
            if (line.Length == 0 && w < words.Length)
            {
                line.Append(words[w][..Math.Min(words[w].Length, widths[f])]);
                w++;
            }

            lines[f] = line.ToString();
        }
        return lines;
    }

    /// <summary>
    /// The reactive write: put <paramref name="text"/> in whatever record comes next and
    /// step past it.
    ///
    /// Kept for <c>pregen_lookahead = 0</c>, and used by nothing else. When
    /// pre-generation is on, both paths writing would have them competing for the same
    /// records with different text — so the reactive path stands down rather than being
    /// deleted, because it is the fallback if pre-generated lines ever read as
    /// disconnected from the scene.
    /// </summary>
    private int WriteNextRecordReactive(string text)
    {
        if (_cfg.PregenLookahead > 0) return 0;
        if (_poolNextRecord.Count == 0) return 0;

        int target  = _poolNextRecord[0];
        int written = WriteRecord(target, text);
        if (written > 0)
            for (int r = 0; r < _poolNextRecord.Count; r++) _poolNextRecord[r] = target + 1;
        return written;
    }

    /// <summary>
    /// Write one record, by index, into every armed region.
    ///
    /// The index is the same in all regions because they are copies of one script with
    /// identical record widths. Letting each region track its own cursor produced exactly
    /// the drift you would expect — region 0 writing record 8 while region 1 wrote record
    /// 7 — which puts the same sentence on two different lines of the same scene.
    ///
    /// Write per record, not per slot: the unit the player sees is the bubble, and a
    /// bubble spans every fragment up to the next header gap. And one record, not all of
    /// them — that breadth is what filled the text log with a single sentence repeated at
    /// every width, and it was also the entire blast radius.
    /// </summary>
    private unsafe int WriteRecord(int index, string text)
    {
        if (!_cfg.PoolWriteEnabled)
        {
            _modLog!.Info($"[POOL] write disabled by config ← \"{text[..Math.Min(text.Length, 50)]}\"");
            return 0;
        }
        if (_heapPools.Count == 0) return 0;

        int totalSlots = 0, regions = 0;

        for (int r = 0; r < _heapPools.Count; r++)
        {
            var (poolBase, poolLen, slots) = _heapPools[r];

            // Validate the whole range, not just the first bytes. The region was captured
            // on an earlier tick and the game may have freed it since; writing through a
            // stale base would fault fatally the same way the scan did.
            if (!MemoryGuard.IsWritable(poolBase, poolLen)) continue;

            byte* p = (byte*)poolBase;
            int wrote = 0;

            var records = _poolRecords[r];
            int target  = index;
            if (target < 0 || target >= records.Count) continue;

            {
                (int start, int count) = records[target];

                var widths = new int[count];
                for (int k = 0; k < count; k++) widths[k] = slots[start + k].Len;

                string[] lines = WrapAcrossFragments(text, widths);

                int mirrored = 0;
                for (int k = 0; k < count; k++)
                {
                    (int off, int len) = slots[start + k];
                    byte[] enc = System.Text.Encoding.ASCII.GetBytes(lines[k]);
                    int    wl  = Math.Min(enc.Length, len);

                    // Snapshot before overwriting. The twin is identified by still holding
                    // these exact bytes, so this has to be captured while they exist —
                    // the same ordering constraint as the scene-script capture in Ch. 64.
                    byte[] original = new byte[len];
                    bool   haveOriginal = len <= _mirrorBuf.Length &&
                                          MemoryGuard.TryRead(poolBase + (nuint)off, original, len);

                    if (wl > 0)
                        fixed (byte* src = enc)
                            System.Buffer.MemoryCopy(src, p + off, len, wl);

                    // Blank the tail. Anything left unwritten is the original dialogue,
                    // and a row of scripted text under a row of generated text reads far
                    // worse than a short line. The newline between fragments is outside
                    // [off, off+len) and is never touched.
                    for (int i = wl; i < len; i++) p[off + i] = (byte)' ';

                    if (haveOriginal && MirrorToTwin(poolBase + (nuint)off, original, len, p + off))
                        mirrored++;
                }

                if (mirrored > 0)
                    _modLog!.Info($"[TWIN] mirrored {mirrored}/{count} rows at delta " +
                                  $"0x{Math.Abs(_twinDelta):X}");

                // The cursor no longer moves here. With a plan, which record to write is
                // chosen by index rather than taken from the cursor, so writing must not
                // also mean "the player advanced" — that conflation is what made three
                // generations land on the record the player was reading. The cursor now
                // means one thing only: how far the interpreter has actually got.
                _modLog!.Info($"[POOL] region {r} record {target}/{records.Count} " +
                              $"@0x{poolBase + (nuint)slots[start].Off:X} rows={count}");
                wrote++;
            }

            if (wrote > 0) { totalSlots += wrote; regions++; }
        }

        _modLog!.Info($"[POOL] wrote {totalSlots} records across {regions} regions ← " +
                      $"\"{text[..Math.Min(text.Length, 50)]}\"");
        return totalSlots;
    }

    // Records (offset, original length) for every English entry in the pool.
    // Captured once, before any write, so every later write measures against the
    // ORIGINAL entry lengths. Without this, each pass would re-measure the shortened
    // string it wrote last time and the usable space would ratchet down to nothing.
    private static unsafe (int Off, int Len)[] CapturePoolSlots(nuint poolBase, int scanLen)
    {
        var slots = new System.Collections.Generic.List<(int, int)>();
        byte* p = (byte*)poolBase;
        int pos = 0;

        // 20000, not 512. Slots are captured from the region base outward, so a low cap
        // covered only the first sliver of a multi-megabyte region — text deeper in was
        // scored but never made writable, which defeated the whole point of scanning it.
        while (pos < scanLen && slots.Count < 20000)
        {
            while (pos < scanLen && !IsPrintable(p[pos])) pos++;
            int begin = pos;
            while (pos < scanLen && IsPrintable(p[pos])) pos++;
            if (pos > begin && IsEnglishSentence(p + begin, pos - begin))
                slots.Add((begin, pos - begin));
        }
        return slots.ToArray();
    }

    /// <summary>
    /// Log the record the interpreter is reading, whenever it changes.
    ///
    /// Sampling on the poll tick rather than logging from the stub is not a shortcut: the
    /// hook fires once per character of every message in the game, so anything managed on
    /// that path would run tens of thousands of times a second. The stub writes two words
    /// to a fixed buffer and nothing else; this reads the buffer at 500 ms and reports
    /// only transitions.
    ///
    /// The consequence is that short-lived records can be missed entirely. That is fine
    /// for verification — what needs proving is that the pointer is real, in the heap, and
    /// changes line to line — and it is the wrong mechanism for actually driving writes,
    /// which will key off the record changing rather than a timer.
    /// </summary>
    private void ReportWatchedRecord()
    {
        if (_msgWatch is null) return;

        // Opening the backlog re-reads every past line at once, which is a legitimate
        // burst rather than a bug — but logging all of it drowns the console and puts
        // dozens of writes on a tick the game is waiting behind. Every entry is still
        // processed for the cursor and the twin delta; only the log is capped.
        const int MaxLogged = 6;
        int logged = 0, suppressed = 0;

        // Outside a hang-out none of this is useful: no pool to arm, no plan to advance,
        // no twin worth learning. The interpreter is busiest exactly then — the title
        // screen, save loading and menus push far more text than a conversation does — and
        // previewing every record through ReadProcessMemory and logging it was enough to
        // be felt as stutter during boot. Draining without inspecting keeps the sampler's
        // queue from filling while costing nothing.
        if (!_sessionActive)
        {
            _msgWatch.DrainSeen();
            return;
        }

        foreach ((nuint record, int cursor) in _msgWatch.DrainSeen())
        {
            if (record == _lastWatchedRecord) continue;
            _lastWatchedRecord = record;

            // Preview from the record base, not from record+cursor. The two are captured
            // at slightly different points and are not a coherent pair: the cursor has
            // usually run on to wherever that message ended, which is trailing control
            // bytes rather than text. The base is the stable half.
            string text = ReadRecordPreview(record);

            // Whether the record falls inside a region the pool heuristic armed is the
            // whole question this hook exists to answer. "in-pool" means the two agree; a
            // scene line reported as "elsewhere" means the ranking missed the live buffer.
            string where = AdvancePoolCursor(record) ? "in-pool  " : "elsewhere";

            // The struct in front of the text, read but not yet acted on. Ch. 75 derives
            // the layout from the MSG1 format and one live measurement; two scenes of
            // this log decide whether it survives contact with a real pool.
            string header = TryReadHeader(record, out BmdMessage msg) ? msg.ToString() : "unparsed";

            // Arm from the read itself. Gated on an active hang-out because the
            // interpreter serves every string in the game — menus, item names, the
            // newspaper — and arming on the first record seen anywhere would point the
            // writer at a UI buffer.
            //
            // Requiring a dialogue-length preview is the second gate: a scene line is a
            // sentence, and a twelve-character floor rejects labels without pretending to
            // be a classifier.
            if (_sessionActive && text.Length >= 12)
            {
                TryArmFromRecord(record);
            }

            LearnTwinDelta(record, text);

            if (logged++ < MaxLogged)
                _modLog!.Info($"[WATCH] {where} record=0x{record:X} cursor=0x{cursor:X} " +
                              $"[{header}] \"{text}\"");
            else
                suppressed++;
        }

        if (suppressed > 0)
            _modLog!.Info($"[WATCH] +{suppressed} more records this tick (backlog scroll)");
    }

    /// <summary>
    /// Write every record whose text has arrived but has not reached game memory yet.
    ///
    /// Split from generation because the two fail differently and at different times: a
    /// 503 loses the text, a freed region loses the write. Keeping them separate means a
    /// write that fails on one tick is simply retried on the next, with the generated
    /// line still in hand.
    ///
    /// Records already read by the interpreter are skipped. That is the Ch. 71 bug
    /// restated as a rule — overwriting what the player is looking at is what made the
    /// sentence change under them.
    /// </summary>
    private void FlushReadyRecords()
    {
        lock (_writeLock)
        foreach (RecordPlan record in _plan.ToArray())
        {
            if (record.State != RecordState.Ready || record.Generated is null) continue;
            if (!record.IsWritable) continue;

            // Behind the player is not a write, it is vandalism of the backlog. Observed
            // as "#0 written, -32 ahead of the player": the request was issued while the
            // cursor was at 0, the player advanced 32 records during the round trip, and
            // the answer arrived to overwrite a line spoken half a minute earlier.
            //
            // The freeze in AdvancePoolCursor catches this only for records the
            // interpreter reported; a scene skipped fast enough leaves gaps it never saw.
            if (record.Index < CurrentRecord())
            {
                record.State = RecordState.Rendered;   // out of the way for good

                // Two outcomes, same code path, very different meanings. A pre-generated
                // line the player outran is routine. A reply to something they *typed*,
                // discarded for the same reason, is the feature not working — and it was
                // indistinguishable in the log until this said so (Ch. 88).
                if (record.AnsweringPlayer)
                    _modLog!.Warn($"[PLAYER] #{record.Index} reply arrived too late — " +
                                  $"player is at {CurrentRecord()}; the typed line is lost");
                else
                    _modLog!.Info($"[PREGEN] #{record.Index} discarded — player is at " +
                                  $"{CurrentRecord()}");
                continue;
            }

            if (WriteRecord(record.Index, record.Generated) > 0)
            {
                record.State      = RecordState.Written;
                record.WasWritten = true;
                string tag = record.AnsweringPlayer ? "PLAYER" : "PREGEN";
                _modLog!.Info($"[{tag}] #{record.Index} written, " +
                              $"{record.Index - CurrentRecord()} ahead of the player");
            }
        }
    }

    /// <summary>
    /// Dump the bytes between consecutive text runs, to find out what actually separates
    /// the rows of one bubble from the start of the next message.
    ///
    /// Grouping currently joins runs exactly one byte apart, because the one separator
    /// ever inspected in a hex editor was a bare 0x0A. That is not the whole story: a
    /// bubble rendered as
    ///
    ///     "Let's pump it up then, Joker. you gotta— Wait, that ain't it!"
    ///
    /// had its rows twelve bytes apart, so grouping saw two records, wrote the generated
    /// line into the first, and left the rest of the script showing underneath.
    ///
    /// Widening the threshold by guesswork is how the 211-slot write happened. Twelve is
    /// close to the ~27 that separates genuine messages, and the difference has to be read
    /// out of the bytes rather than assumed from the distance.
    /// </summary>
    private void LogSlotSeparators(nuint poolBase, (int Off, int Len)[] slots)
    {
        var buf = new byte[32];
        for (int i = 1, shown = 0; i < slots.Length && shown < 10; i++)
        {
            int gapStart = slots[i - 1].Off + slots[i - 1].Len;
            int gap      = slots[i].Off - gapStart;
            if (gap <= 1 || gap > 32) continue;   // 1 is the known newline; huge gaps are padding

            if (!MemoryGuard.TryRead(poolBase + (nuint)gapStart, buf, gap)) continue;

            var hex = new System.Text.StringBuilder(gap * 3);
            for (int b = 0; b < gap; b++) hex.Append(buf[b].ToString("X2")).Append(' ');

            _modLog!.Info($"[GAP] {gap,2}B between #{i - 1} and #{i}: {hex}");
            shown++;
        }
    }

    /// True when the address lies inside a region already armed for writing.
    private bool InArmedPool(nuint addr)
    {
        foreach ((nuint poolBase, int poolLen, _) in _heapPools)
            if (addr >= poolBase && addr < poolBase + (nuint)poolLen) return true;
        return false;
    }

    /// <summary>
    /// Arm the region containing a record the interpreter just read.
    ///
    /// This replaces the heap scan, and the difference is not an optimisation — it is a
    /// different question. The scan asked "which of 4030 regions looks most like
    /// dialogue?", walked 1.4 GB scoring English, and took 33 seconds, during which the
    /// first six lines of the scene played untouched. This asks "what region is that
    /// address in?", which VirtualQuery answers in microseconds, about an address the
    /// game supplied by reading it.
    ///
    /// Validation still happens, because the interpreter serves every string in the game
    /// and the first record seen could be a menu label. But validating one candidate the
    /// game pointed at is a different problem from searching for it.
    /// </summary>
    private void TryArmFromRecord(nuint record)
    {
        const int MinSlots  = 6;
        const int MinRegion = 0x4000;         // 16 KB
        const int MaxRegion = 0x400000 * 8;   // 32 MB; scene pools measured 45 KB - 1.4 MB

        // A scene's script is small. Measured across five sessions the gym pool held 36
        // text runs every time, while the pool the game reads alongside it — menu prompts,
        // "AHang out with him", "ACheck bond with Ryuji" — held 340 to 409.
        //
        // Without this the two take turns arming each other. Reads interleave between them
        // all through a scene, so consecutive outside reads happen constantly, and every
        // flip rebuilt the plan and discarded everything generated so far. Coverage fell to
        // 50%, with records 0-6 written and then wiped.
        const int MaxSceneSlots = 120;

        if (InArmedPool(record) || IsTwinOfArmed(record))
        {
            _outsideReads = 0;
            return;
        }

        // A hang-out is not one pool. The first scene of a session was a conversation on
        // the street and armed 0x41DBE73000; the gym pool appeared fourteen seconds later
        // at 0x4250C8B13C and never got a look, because arming had already happened.
        //
        // Reads landing outside the armed region are how a scene change announces itself.
        // Two in a row, because one stray read is a menu or a name plate, while a scene
        // that has genuinely moved keeps reading from its new pool.
        if (_heapPools.Count > 0 && ++_outsideReads < 2) return;

        (bool ok, nuint regionBase, nuint regionSize, uint state, uint _) =
            MemoryGuard.QueryRegion(record);
        const uint MEM_COMMIT = 0x1000;
        if (!ok || state != MEM_COMMIT) return;
        if (regionSize < MinRegion || regionSize > MaxRegion) return;
        if (!MemoryGuard.IsWritable(regionBase, (int)Math.Min(regionSize, 0x1000))) return;

        var slots = CapturePoolSlotsSafe(regionBase, (int)regionSize);
        if (slots.Length < MinSlots || slots.Length > MaxSceneSlots)
        {
            _modLog!.Info($"[ARM] 0x{regionBase:X} rejected — {slots.Length} text runs, " +
                          $"outside {MinSlots}-{MaxSceneSlots}. Not a scene script.");
            return;
        }

        // Re-arming the region already armed would rebuild the plan and throw away every
        // line generated for it — the same damage as a flip, from a no-op.
        if (_heapPools.Count > 0 && _heapPools[0].Base == regionBase)
        {
            _outsideReads = 0;
            return;
        }

        // Exactly one region is armed, and it is replaced rather than added to. Arming two
        // and writing the same record index into both assumed they were copies. They were
        // not — 207 records against 180, two unrelated pools, so index N named a different
        // line in each.
        //
        // The second copy is still written, by MirrorToTwin, which finds it at a learned
        // offset and verifies byte-for-byte before touching anything. That is the
        // mechanism that actually knows two addresses hold the same line.
        // Report the outgoing plan before replacing it. A scene with sub-scenes re-arms
        // as it moves between them, and coverage measured only at session end described
        // the last plan alone — 1 of 11, while 67 records had been written across the
        // session. A number that ignores most of its subject is worse than none.
        LogSceneCoverage();

        lock (_writeLock)
        {
        _outsideReads = 0;
        _twinDelta    = 0;   // learned for the pool being replaced; meaningless for this one
        _heapPools.Clear();
        _poolRecords.Clear();
        _poolNextRecord.Clear();

        var (layout, records) = BuildRecordLayout(regionBase, slots);
        _heapPools.Add((regionBase, (int)regionSize, layout));
        _poolRecords.Add(records);
        _poolNextRecord.Add(0);
        _poolIsHeap = true;

        _modLog!.Info($"[ARM] 0x{regionBase:X} len={regionSize} slots={slots.Length} " +
                      "— from a live read, no scan");

        BuildRecordPlan();
        }

        // Outside the lock: the request goes to the thread pool and its continuation takes
        // the same lock to flush.
        KickPregen();
    }

    /// <summary>
    /// True when the address is the armed pool's second copy, at the learned offset.
    ///
    /// Without this every twin read looks like a read outside the armed region and would
    /// trigger a re-arm onto the copy — the two pools then take turns evicting each other
    /// for the length of the scene.
    /// </summary>
    private bool IsTwinOfArmed(nuint addr)
    {
        if (_twinDelta == 0) return false;
        return InArmedPool((nuint)((long)addr + _twinDelta))
            || InArmedPool((nuint)((long)addr - _twinDelta));
    }

    /// <summary>
    /// Learn where the second copy of a record lives, from two reads of the same text.
    ///
    /// The interpreter reads both copies of a line within about a millisecond, and the
    /// sampler catches the pair as two consecutive distinct records with identical
    /// preview text. Their difference is the delta, and it has held constant across every
    /// pair within a run.
    ///
    /// This replaces content-matched twin arming, which searched a ranked list for a
    /// region with the same sample and missed entirely in one run — leaving a single
    /// region armed, every write landing in the text log, and the bubble showing the
    /// original script. Ranking cannot see a relationship the game never expresses as
    /// similarity of score; the hook observes it directly.
    /// </summary>
    private void LearnTwinDelta(nuint record, string text)
    {
        // Short or empty previews are not evidence: control-code runs and blank records
        // collide constantly, and a delta learned from one would aim writes at nothing.
        if (text.Length >= 12 && text == _lastWatched.Text && record != _lastWatched.Addr)
        {
            long delta = (long)record - (long)_lastWatched.Addr;
            if (delta != _twinDelta)
            {
                _twinDelta = delta;
                _modLog!.Info($"[TWIN] delta={(delta < 0 ? "-" : "")}0x{Math.Abs(delta):X} " +
                              $"from 0x{_lastWatched.Addr:X} / 0x{record:X} \"{text}\"");
            }
        }
        _lastWatched = (record, text);
    }

    /// <summary>
    /// Copy a just-written span into the twin, but only where the twin still holds what
    /// the target held before the write.
    ///
    /// The delta comes from observation and observation can be wrong, so this never
    /// writes on the strength of arithmetic alone. The caller passes the original bytes;
    /// if the mirror does not match them byte for byte, the address is not the same line
    /// and nothing is written. That check is what keeps a bad delta from turning into the
    /// 211-slot data-table write of Ch. 63 by another route.
    /// </summary>
    private unsafe bool MirrorToTwin(nuint addr, byte[] original, int len, byte* written)
    {
        if (_twinDelta == 0 || len <= 0 || len > _mirrorBuf.Length) return false;

        // Both directions, because the sign of the delta is an accident of which copy the
        // interpreter happened to read first. LearnTwinDelta computes (this - previous),
        // and the game alternates, so the log shows the same offset as +0x33D83600 and
        // -0x33D83600 within one scene. Trying one direction meant the mirror silently
        // refused about half the time — it fired zero times in a whole hang-out, and the
        // text log showed the original script throughout.
        //
        // Trying both is safe precisely because the guard below does the deciding: an
        // address only gets written if it still holds the bytes the target held.
        return TryMirrorAt((nuint)((long)addr - _twinDelta), original, len, written)
            || TryMirrorAt((nuint)((long)addr + _twinDelta), original, len, written);
    }

    private unsafe bool TryMirrorAt(nuint mirror, byte[] original, int len, byte* written)
    {
        if (!MemoryGuard.TryRead(mirror, _mirrorBuf, len)) return false;

        for (int i = 0; i < len; i++)
            if (_mirrorBuf[i] != original[i]) return false;

        if (!MemoryGuard.IsWritable(mirror, len)) return false;

        byte* dst = (byte*)mirror;
        for (int i = 0; i < len; i++) dst[i] = written[i];
        return true;
    }

    /// <summary>
    /// Point a pool at the record after the one just rendered. Returns whether
    /// <paramref name="addr"/> landed in an armed pool at all.
    ///
    /// The hook reports the record's base, which sits a header ahead of the text — the
    /// one measured live had its first character at +0x28. So the match is not
    /// containment but "the first record whose text begins at or shortly after this
    /// address", within a window wide enough to clear any header seen so far.
    ///
    /// Advancing on render rather than on write is deliberate. A write that never gets
    /// displayed (the player skipped, the scene branched) must not consume a record, or
    /// the mod would drift one line ahead of the game and stay there for the rest of the
    /// scene.
    /// </summary>
    private bool AdvancePoolCursor(nuint addr)
    {
        const int HeaderWindow = 0x80;
        bool inPool = false;

        for (int r = 0; r < _heapPools.Count; r++)
        {
            (nuint poolBase, int poolLen, (int Off, int Len)[] slots) = _heapPools[r];
            if (addr < poolBase || addr >= poolBase + (nuint)poolLen) continue;
            inPool = true;

            int offset  = (int)(addr - poolBase);
            var records = _poolRecords[r];

            for (int i = 0; i < records.Count; i++)
            {
                int textOff = slots[records[i].Start].Off;
                if (textOff < offset) continue;
                if (textOff - offset > HeaderWindow) break;   // ordered; no closer match ahead

                // Never move backwards, and never leap forwards.
                //
                // Backwards is the backlog: scrolling it re-reads earlier records, and
                // treating that as progress would rewind the write target onto lines
                // already spoken.
                //
                // Forwards needs a limit too, which a 36-slot gym pool never revealed. A
                // Takemi scene armed a 109-slot region holding several scenes, the game
                // read it out of order, and one stray read three seconds in put the cursor
                // at record 63 of 64 — every subsequent line was discarded as "behind the
                // player" and the scene generated nothing at all.
                //
                // Reading advances one record at a time. A jump of sixty is not a player,
                // it is a different part of the pool being touched for its own reasons.
                const int MaxCursorJump = 8;
                int next = i + 1;
                if (next > _poolNextRecord[r] && next - _poolNextRecord[r] <= MaxCursorJump)
                    _poolNextRecord[r] = next;

                // Freeze it. Once the interpreter has read a record the player has seen
                // it, and rewriting it is the bug they described as the text switching.
                // Backlog re-reads freeze too, which is correct — a line in the log has
                // certainly been shown.
                //
                // Foreign is left alone. It is already terminal, and flipping it to
                // Rendered would erase the reason the record was skipped — coverage would
                // then count another character's line as one the queue failed to reach.
                if (i < _plan.Count &&
                    _plan[i].State != RecordState.Rendered &&
                    _plan[i].State != RecordState.Foreign)
                {
                    _plan[i].State = RecordState.Rendered;
                    _modLog!.Info($"[PLAN] #{i} rendered — frozen");
                }
                break;
            }
        }
        return inPool;
    }

    /// <summary>
    /// Parse the BMD header at a record base, through ReadProcessMemory.
    ///
    /// Same reasoning as ReadRecordPreview: the pointer came out of a register inside the
    /// game's own loop and the message may have been freed before this tick looks at it,
    /// so the read has to be one that can fail rather than one that can fault.
    /// </summary>
    private bool TryReadHeader(nuint record, out BmdMessage message)
    {
        message = default;
        return MemoryGuard.TryRead(record, _headerBuf, BmdMessage.HeaderBytes)
            && BmdMessage.TryParse(_headerBuf, out message);
    }

    /// <summary>
    /// First readable English run at or shortly after <paramref name="record"/>.
    ///
    /// Read through ReadProcessMemory rather than by dereferencing. The pointer arrives
    /// from a stub that captured it inside the game's own loop, and by the time the poll
    /// tick looks at it the message may have been freed — checking IsReadable and then
    /// walking raw bytes races the allocator in exactly the way that killed the scan in
    /// Ch. 61, and an access violation there is uncatchable.
    ///
    /// The scan starts at the base and skips forward because a record begins with a
    /// header: the one measured live had its text at +0x28, and that offset is not
    /// guaranteed to be constant across message kinds.
    /// </summary>
    private string ReadRecordPreview(nuint record)
    {
        const int Window = 128;
        if (!MemoryGuard.TryRead(record, _recordBuf, Window)) return "";

        for (int start = 0; start < Window; start++)
        {
            if (!IsPrintable(_recordBuf[start])) continue;

            int end = start;
            while (end < Window && IsPrintable(_recordBuf[end])) end++;
            if (end - start < 8) { start = end; continue; }

            var sb = new System.Text.StringBuilder(end - start);
            for (int i = start; i < end; i++) sb.Append((char)_recordBuf[i]);
            string candidate = sb.ToString();
            if (IsEnglishString(candidate)) return candidate;
            start = end;
        }
        return "";
    }
}
