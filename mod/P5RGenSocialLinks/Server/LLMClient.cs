using System;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace P5RGenSocialLinks.Server;

internal sealed class GenerateRequest
{
    public int    ConfidantId   { get; init; }
    public int    Rank          { get; init; }
    public string Context       { get; init; } = string.Empty;
    public string CharacterName { get; init; } = string.Empty;

    /// <summary>
    /// Characters the destination record can display, or null when it is not known yet.
    ///
    /// The mod picks the target record before it asks for a line, so it knows exactly how
    /// much room the answer has: about 30 characters for a one-row record and 75 for two.
    /// Sending a fixed budget produced "You're finally here, I've been" on screen — a
    /// 53-character line clipped into a 30-character slot.
    ///
    /// Nullable so a request made before a record is chosen still works, and the server
    /// falls back to its configured default.
    /// </summary>
    public int?   MaxChars      { get; init; }
}

internal sealed class GenerateResponse
{
    public string Text { get; init; } = string.Empty;
}

/// <summary>Thrown when the server is already processing another request.</summary>
internal sealed class InferenceInFlightException : Exception
{
    internal InferenceInFlightException() : base("Server returned 429: inference in-flight.") { }
}

/// <summary>
/// The server answered 503 after every retry: the inference backend is not up.
///
/// Distinct from a generic HTTP failure because the remedy is different and outside the
/// mod — llama-server is down, or the model never finished loading. A scene ran its whole
/// length against a dead backend and the mod's log said nothing at all (Ch. 80).
/// </summary>
internal sealed class BackendUnavailableException : Exception
{
    internal BackendUnavailableException(int retries, TimeSpan delay)
        : base($"inference backend unavailable (503 after {retries} retries over " +
               $"{retries * delay.TotalSeconds:F0}s) — is llama-server up?") { }
}

internal sealed class LLMClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly string     _baseUrl;

    internal LLMClient(string baseUrl = "http://127.0.0.1:8765")
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _http    = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };
    }

    // Pydantic expects snake_case JSON keys (confidant_id, character_name, etc.)
    private static readonly JsonSerializerOptions _jsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    // Number of times to retry on 503 (model still loading) before giving up.
    // 503 on first game boot: model load takes ~20s; 3 retries × 8s = 24s max wait.
    private const int MaxRetries503 = 3;
    private static readonly TimeSpan Retry503Delay = TimeSpan.FromSeconds(8);

    /// <returns>Generated text, or throws <see cref="InferenceInFlightException"/> on 429.</returns>
    internal async Task<string> GenerateAsync(GenerateRequest request, CancellationToken ct = default)
    {
        string json = JsonSerializer.Serialize(request, _jsonOpts);

        for (int attempt = 0; ; attempt++)
        {
            using var content  = new StringContent(json, Encoding.UTF8, "application/json");
            using var response = await _http.PostAsync($"{_baseUrl}/generate", content, ct);

            if (response.StatusCode == HttpStatusCode.TooManyRequests)
                throw new InferenceInFlightException();

            if (response.StatusCode == HttpStatusCode.ServiceUnavailable && attempt < MaxRetries503)
            {
                await Task.Delay(Retry503Delay, ct);
                continue;
            }

            // Name the status rather than letting EnsureSuccessStatusCode phrase it. A 503
            // means the inference backend is not answering, which is a different problem
            // from anything the mod can fix, and the caller's log line is where someone
            // will look first.
            if (response.StatusCode == HttpStatusCode.ServiceUnavailable)
                throw new BackendUnavailableException(MaxRetries503, Retry503Delay);

            response.EnsureSuccessStatusCode();
            string body = await response.Content.ReadAsStringAsync(ct);
            var result  = JsonSerializer.Deserialize<GenerateResponse>(body, _jsonOpts);
            return result?.Text ?? string.Empty;
        }
    }

    public void Dispose() => _http.Dispose();
}