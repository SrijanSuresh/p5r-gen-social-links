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
