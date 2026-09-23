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
/// The heap scan: finding the dialogue pool without being told where it is.
///
/// Superseded by the interpreter hook and off by default, but not dead. `heap_scan_enabled`
/// in GenDialogue.json is a live fallback for a scene that ends up with no pool armed at
/// all, which is why this sits beside the pool writer rather than in Mod.Discovery.cs.
///
/// Two facts about it are worth keeping in view. It walks heap regions through
/// ReadProcessMemory rather than dereferencing them, because a direct pointer walk races
/// the game's allocator and faults fatally (Ch. 61). And it cost 33 seconds and 1.4 GB per
/// scene to produce a ranking that was right most of the time, against microseconds and an
/// exact answer from asking the consumer (Ch. 65) - the comparison that made deleting the
/// search the obvious move.
/// </summary>
public partial class Mod
{
    /// <summary>
    /// Locates the scene's dialogue pool on the heap and captures its slots.
    ///
    /// Cheat Engine established the layout: consecutive lines land in the same regions at
    /// increasing offsets — "Protein Lovers gym!" at 0x41DD7F6389, the next line 0x56A
    /// later at 0x41DD7F68F3, with a second region advancing by exactly the same delta.
    /// The scene's lines therefore sit sequentially in one allocation rather than being
    /// reallocated per line, which makes the containing region a stable write target for
    /// the whole conversation.
    ///
    /// Regions are ranked by how many non-system English sentences they hold and the top
    /// candidates are logged, so a wrong pick is visible rather than silent — the same
    /// mistake the mapped-file scan made when it locked onto the shoe shop.
    /// </summary>
    private unsafe void TryFindHeapDialoguePool()
    {
        const uint MEM_COMMIT = 0x1000, PAGE_NOACCESS = 0x01, PAGE_GUARD = 0x100;
        // 8 MB per region, not 256 KB. Sampling only a region's start is what hid the live
        // dialogue: the confirmed address 0x41DD7F6389 sits ~12 MB past the base of the
        // nearest region the scan reported, so it was never read.
        const int  maxScan     = 0x800000;
        const int  maxRegions  = 4096;
        // 2 GB, not 512 MB. At 8 MB per region the old budget stopped after 64 regions —
        // far short of the game's heap — so the scan could exhaust itself before ever
        // reaching the allocation holding the live scene.
        const long totalBudget = 2048L * 1024 * 1024;

        var ranked = new System.Collections.Generic.List<(int Score, nuint Base, int Len, string Sample)>();
        nuint addr = GameHeapStart;
        int   seen = 0;
        long  totalScanned = 0;

        while (addr < GameHeapEnd && seen < maxRegions && totalScanned < totalBudget)
        {
            var (ok, regionBase, regionSize, state, protect) = MemoryGuard.QueryRegion(addr);
            if (!ok || regionSize == 0) break;
            nuint regionEnd = regionBase + regionSize;
            if (regionEnd <= addr) break;
            seen++;

            bool usable = state == MEM_COMMIT
                          && (protect & PAGE_NOACCESS) == 0
                          && (protect & PAGE_GUARD) == 0;

            if (usable && regionSize >= 0x1000 && regionSize <= 0x4000000)
            {
                int len = (int)Math.Min((ulong)regionSize, (ulong)maxScan);

                // Read through TryRead in chunks rather than dereferencing the region
                // directly: the game frees heap on its own threads and a raw walk over
                // megabytes will eventually fault on memory that vanished mid-scan.
                int    score  = 0;
                int    matches = 0;
                string sample = "";
                int    read   = 0;

                for (int chunkOff = 0; chunkOff < len; chunkOff += ScanChunk)
                {
                    int want = Math.Min(ScanChunk, len - chunkOff);
                    if (!MemoryGuard.TryRead(regionBase + (nuint)chunkOff, _scanBuf, want)) break;
                    read += want;

                    foreach (var (a, t) in FindAsciiEnglishBuf(_scanBuf, want,
                                                              regionBase + (nuint)chunkOff, 8192))
                    {
                        int s = DialogueScore(t);
                        if (s == 0) continue;
                        if (sample.Length == 0) sample = t;
                        score += s;
                        matches++;
                    }
                }

                // Rank by average score per matched line, not the sum. A summed score is
                // really a measure of size: a 2MB item-description table outscores a
                // 400KB conversation pool on volume alone, which is how the region that
                // actually renders ended up ranked #14. Per-line, casual speech (ends in
                // punctuation, second-person, contractions) separates cleanly from
                // "Restores 20 HP to one ally."
                if (matches >= 30)
                {
                    int avg = score * 100 / matches;
                    ranked.Add((avg, regionBase, len, sample));
                }
                totalScanned += read;
            }
            addr = regionEnd;
        }

        // Coverage matters as much as the ranking: if the budget ran out, the right region
        // may simply never have been read, which looks identical to a bad ranking.
        _modLog!.Info($"[POOL] scanned {seen} regions, {totalScanned / (1024 * 1024)} MB, " +
                      $"{ranked.Count} scored" +
                      $"{(totalScanned >= totalBudget ? " (BUDGET EXHAUSTED — coverage incomplete)" : "")}");

        if (ranked.Count == 0) return;
        ranked.Sort((a, b) => b.Score.CompareTo(a.Score));

        // Average-per-line ranking reliably puts the live scene at #0 ("Here we are...
        // Protein Lovers gym!", avg 4.72) above item tables at 3.18 and shader source at
        // 1.00. Rank alone is not enough to pick the write set, though, for two reasons
        // learned the hard way:
        //
        //  - Taking the top 3 by rank once armed "ternTableOffset: -1", a data table that
        //    averaged well without being dialogue, and writing it crashed the game. Rank
        //    is a similarity score, not a type check.
        //  - Taking only #0 rendered nothing. P5R holds the scene's dialogue in TWO
        //    buffers: one the text log reads, one the renderer reads. Writing #0 filled
        //    the backlog with generated lines while the speech bubble kept the original.
        //    The twin was sitting at alt #9 (avg 2.35) with an identical sample — far
        //    below any sane rank cutoff, and identical in content.
        //
        // So: anchor on #0 by rank, then include its content-identical siblings. Matching
        // on text rather than score finds the second copy wherever it ranks, and cannot
        // pull in an unrelated structure, because a data table never shares a sample with
        // the scene's dialogue.
        _heapPools.Clear();
        _poolRecords.Clear();
        _poolNextRecord.Clear();
        _plan.Clear();
        string anchorSample = ranked[0].Sample;

        var selected = new System.Collections.Generic.List<int> { 0 };
        for (int i = 1; i < ranked.Count && selected.Count < 8; i++)
        {
            if (ranked[i].Sample == anchorSample) selected.Add(i);
        }

        // MaxWriteRegions still caps rank-based additions, for diagnosing a scene that
        // ranks poorly. Siblings are exempt: they are the same buffer content, so they
        // carry the safety of the anchor rather than the risk of an unranked guess.
        int extraByRank = Math.Max(_cfg.MaxWriteRegions, 1) - 1;
        for (int i = 1; i < ranked.Count && extraByRank > 0; i++)
        {
            if (selected.Contains(i)) continue;
            selected.Add(i);
            extraByRank--;
        }
        selected.Sort();

        // List the runners-up. If the ranking ever shifts and the scene stops landing at
        // the top, or a third copy appears, these are the evidence needed to see it.
        for (int i = 0; i < Math.Min(12, ranked.Count); i++)
        {
            if (selected.Contains(i)) continue;
            _modLog!.Info($"[POOL] alt #{i} 0x{ranked[i].Base:X} avg={ranked[i].Score / 100.0:F2}: " +
                          $"\"{ranked[i].Sample}\"");
        }

        // Snapshot the script from the anchor before anything is armed for writing.
        // Ordering is load-bearing: the write is destructive and the pool is the only
        // copy, so this is the last moment the scene's real dialogue exists to be read.
        if (_sceneScript.Count == 0)
        {
            var script = CaptureSceneScript(ranked[0].Base, ranked[0].Len,
                                            maxLines: 12, maxChars: 480);
            if (script.Length > 0)
            {
                _sceneScript.AddRange(script);
                _modLog!.Info($"[SCRIPT] captured {script.Length} original lines: " +
                              $"\"{script[0][..Math.Min(script[0].Length, 60)]}\"");
            }
        }

        foreach (int i in selected)
        {
            var c = ranked[i];
            var slots = CapturePoolSlotsSafe(c.Base, c.Len);
            if (slots.Length == 0) continue;
            _heapPools.Add((c.Base, c.Len, slots));
            _poolRecords.Add(GroupSlotsIntoRecords(c.Base, slots));
            // -1 means nothing observed yet, so the next write targets record 0.
            _poolNextRecord.Add(0);
            string why = i == 0 ? "anchor" : (c.Sample == anchorSample ? "twin" : "rank");
            _modLog!.Info(
                $"[POOL] ARM #{i} ({why}) 0x{c.Base:X} len={c.Len} avg={c.Score / 100.0:F2} " +
                $"slots={slots.Length}: \"{c.Sample}\"");
            if (i == 0) LogWatchpointTargets(c.Base, slots, maxSlots: 40);
        }

        if (_heapPools.Count == 0) return;
        _poolIsHeap = true;

        // Before the pending write below, for the reason in Ch. 64: this is the
        // last moment each record still holds its scripted line.
        BuildRecordPlan();
        _modLog!.Info($"[POOL] {_heapPools.Count} regions armed for write");

        string? pending = _lastLlmText;
        if (pending != null) WriteNextRecordReactive(pending);
    }

    /// <summary>
    /// Build the record layout, recovering short text the slot scanner skipped.
    ///
    /// The scanner requires a run to read as an English sentence before it counts as a
    /// slot, which is right when deciding whether a region is a script and wrong once
    /// inside one. "you gotta" is nine characters and falls below that floor, so it was
    /// never a slot — and so it survived the write and appeared on screen as
    /// "Let's pump it up then, Joker. you gotta— Wait, that ain't it!".
    ///
    /// Inside a record the question is no longer "is this dialogue" — the record has
    /// already been identified — but "is this text the bubble will draw". A three-
    /// character printable run qualifies. Bytes outside the runs are left alone: the
    /// Shift-JIS dash after "you gotta" is a glyph the game renders, not text we own.
    /// </summary>
    private ((int Off, int Len)[] Slots, System.Collections.Generic.List<(int Start, int Count)> Records)
        BuildRecordLayout(nuint poolBase, (int Off, int Len)[] raw)
    {
        var slots   = new System.Collections.Generic.List<(int, int)>(raw.Length + 8);
        var records = new System.Collections.Generic.List<(int, int)>();
        var gapBuf  = new byte[MaxJoinGap];

        int i = 0;
        while (i < raw.Length)
        {
            int start = slots.Count;
            slots.Add(raw[i]);
            i++;

            while (i < raw.Length && SameRecord(poolBase, raw[i - 1], raw[i], gapBuf))
            {
                AddGapRuns(poolBase, raw[i - 1], raw[i], gapBuf, slots);
                slots.Add(raw[i]);
                i++;
            }
            records.Add((start, slots.Count - start));
        }
        return (slots.ToArray(), records);
    }
}
