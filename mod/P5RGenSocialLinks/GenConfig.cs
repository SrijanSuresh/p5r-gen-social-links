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

    [JsonPropertyName("struct_diff_enabled")]
    public bool StructDiffEnabled { get; init; } = true;

    /// <summary>
    /// When true, PointerChainResolver logs each chain step with address and dereferenced value.
    /// Useful for diagnosing broken pointer chains after a game patch; leave false in production.
    /// </summary>
    [JsonPropertyName("verbose_chain")]
    public bool VerboseChain { get; init; } = false;

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
