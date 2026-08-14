using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace P5RGenSocialLinks;

/// <summary>
/// Runtime configuration loaded from GenDialogue.json next to the mod DLL.
/// Allows tuning throttle/timeout without recompiling.
/// Missing file → silent defaults.
/// </summary>
internal sealed class GenConfig
{
    [JsonPropertyName("server_url")]
    public string ServerUrl { get; init; } = "http://127.0.0.1:8765";

    [JsonPropertyName("throttle_seconds")]
    public double ThrottleSeconds { get; init; } = 3.0;

    [JsonPropertyName("timeout_seconds")]
    public double TimeoutSeconds { get; init; } = 30.0;

    [JsonPropertyName("poll_interval_ms")]
    public int PollIntervalMs { get; init; } = 500;

    // When true, logs every BF instruction byte-dump ([BFInstr]) and StructDiff slot changes.
    // Defaults off — the [MSG] and [LLM] events are sufficient for normal operation.
    [JsonPropertyName("struct_diff_enabled")]
    public bool StructDiffEnabled { get; init; } = false;

    /// <summary>
    /// Master switch for writing generated text into the dialogue pool.
    ///
    /// Set false to run the whole pipeline — session detection, msgId, LLM call — while
    /// touching no game memory. This is the first thing to try when the game becomes
    /// unstable: if crashes stop with it off, the write is the cause, and the log still
    /// shows every generated line.
    /// </summary>
    [JsonPropertyName("pool_write_enabled")]
    public bool PoolWriteEnabled { get; init; } = true;

    /// <summary>
    /// How many of the top-ranked heap regions to write.
    ///
    /// Every armed region is overwritten in full, so this is a blast radius, not a
    /// retry count. Three regions wrote 211 slots across memory that included
    /// "ternTableOffset: -1" — a data table, not dialogue — which is a strong crash
    /// candidate. One keeps the damage to a single mis-ranked region.
    /// </summary>
    [JsonPropertyName("max_write_regions")]
    public int MaxWriteRegions { get; init; } = 1;

    /// <summary>
    /// How many records ahead of the player to keep generated.
    ///
    /// This is what makes pacing irrelevant: at a lookahead of 3, a line is written two
    /// or three bubbles before the player reaches it, so holding fast-forward no longer
    /// beats a 2-second round trip. It cannot be raised without cost — the server runs
    /// one request at a time, so a deep queue spends inference on records the player may
    /// never see if the scene branches.
    ///
    /// 0 disables pre-generation and restores the reactive path, which is the fallback
    /// if pre-generated lines ever read as disconnected from the scene.
    /// </summary>
    [JsonPropertyName("pregen_lookahead")]
    public int PregenLookahead { get; init; } = 3;

    /// <summary>
    /// When true, inject an assembly hook into the BMD message interpreter's byte-fetch
    /// loop to capture the record the game is currently rendering.
    ///
    /// This is the replacement for the heap heuristic: instead of scanning a gigabyte and
    /// scoring regions for English, the game hands over the pointer. It is also the most
    /// invasive thing the mod does — six injected instructions inside a loop that runs per
    /// character of every message in the game — so it gets its own switch. If P5R fails to
    /// start or dies on the first line of dialogue, set this false first.
    /// </summary>
    [JsonPropertyName("msg_hook_enabled")]
    public bool MsgHookEnabled { get; init; } = true;

    /// <summary>
    /// When true, PointerChainResolver logs each chain step with address and dereferenced value.
    /// Useful for diagnosing broken pointer chains after a game patch; leave false in production.
    /// </summary>
    [JsonPropertyName("verbose_chain")]
    public bool VerboseChain { get; init; } = false;

    /// <summary>
    /// Minimum severity of messages written to the Reloaded-II console.
    /// "info" = all messages; "warn" = LLM errors/timeouts only; "off" = silence.
    /// </summary>
    [JsonPropertyName("log_level")]
    public string LogLevel { get; init; } = "info";

    private static readonly JsonSerializerOptions _opts = new() { ReadCommentHandling = JsonCommentHandling.Skip };

    internal static GenConfig Load(string modDirectory)
    {
        string path = Path.Combine(modDirectory, "GenDialogue.json");
        if (!File.Exists(path)) return new GenConfig();

        try
        {
            string json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<GenConfig>(json, _opts) ?? new GenConfig();
        }
        catch (Exception)
        {
            return new GenConfig();
        }
    }
}
