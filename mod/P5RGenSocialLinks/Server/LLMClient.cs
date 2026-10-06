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

    /// <summary>
    /// Who this line is spoken to, or empty when it is not known.
    ///
    /// The prompt assumed Joker was always the listener, which is right for a hang-out
    /// and wrong for every rank-up scene with a cast: Takemi was telling a patient's
    /// father about "your parents" and answering a child's question as though the
    /// protagonist had asked it (Ch. 81).
    ///
    /// Empty is meaningful and is not the same as "Joker". The reactive msgId path has no
    /// record and therefore no turn order to derive this from, and a scene whose archive
    /// did not resolve knows the turn changed hands without knowing to whom. Both send
    /// nothing and the server leaves the addressee out of the prompt, which is where it
    /// was before this field existed.
    /// </summary>
    public string Addressee     { get; init; } = string.Empty;

    /// <summary>
    /// What the player typed, when this request is answering it.
    ///
    /// Only ever set on the one request that collected a line from the store. The server
    /// deliberately does not inject a pending line into whatever request arrives next:
    /// pre-generation runs eight records ahead, so an auto-grabbed line would be answered
    /// several bubbles after the player typed it (Ch. 87).
    /// </summary>
    public string PlayerLine    { get; init; } = string.Empty;
}

/// <summary>The pending typed line, as <c>GET /player-line</c> reports it.</summary>
internal sealed class PlayerLineResponse
{
    public bool   Pending    { get; init; }
    public string Text       { get; init; } = string.Empty;
    public double AgeSeconds { get; init; }
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

    /// <summary>
    /// Ask whether the player has typed something, without taking it.
    ///
    /// Peeking rather than consuming is what makes the feature safe to poll. The mod has
    /// to know a line exists before it commits a record to answering it, and if it
    /// collected first and then found no writable record ahead, the line would be gone
    /// with nowhere to put it (Ch. 88).
    ///
    /// Returns false on any transport failure. A server that is not answering is the
    /// normal case for this mod — it means scripted dialogue, not an error path — and
    /// this runs on a poll tick, so a throw here would be noise on every tick.
    /// </summary>
    internal async Task<bool> HasPlayerLineAsync(CancellationToken ct = default)
    {
        PlayerLineResponse? line = await TryGetPlayerLineAsync(consume: false, ct);
        return line?.Pending == true;
    }

    /// <summary>
    /// Collect the pending typed line, removing it from the store.
    ///
    /// Returns null when there is nothing to collect — it expired between the poll and
    /// this call, or something else took it. Both are ordinary in a two-process system
    /// where the store is shared, and the caller has to put its record back rather than
    /// leave it waiting for a line that is not coming.
    /// </summary>
    internal async Task<string?> TakePlayerLineAsync(CancellationToken ct = default)
    {
        PlayerLineResponse? line = await TryGetPlayerLineAsync(consume: true, ct);
        return line?.Pending == true && line.Text.Length > 0 ? line.Text : null;
    }

    /// <summary>Forget any pending line. Called when a hang-out ends.</summary>
    internal async Task ClearPlayerLineAsync(CancellationToken ct = default)
    {
        try
        {
            using var empty = new StringContent(string.Empty);
            using var response = await _http.PostAsync($"{_baseUrl}/player-line/clear", empty, ct);
        }
        catch (HttpRequestException)
        {
            // Nothing to clear if nobody is listening.
        }
        catch (OperationCanceledException)
        {
        }
    }

    private async Task<PlayerLineResponse?> TryGetPlayerLineAsync(
        bool consume, CancellationToken ct)
    {
        try
        {
            string url = $"{_baseUrl}/player-line" + (consume ? "?consume=true" : "");
            using var response = await _http.GetAsync(url, ct);

            // A build of the server without the channel answers 404. That is a version
            // mismatch rather than a fault, and it must not log once per poll tick.
            if (!response.IsSuccessStatusCode) return null;

            string body = await response.Content.ReadAsStringAsync(ct);
            return JsonSerializer.Deserialize<PlayerLineResponse>(body, _jsonOpts);
        }
        catch (HttpRequestException)
        {
            return null;
        }
        catch (JsonException)
        {
            return null;
        }
        catch (OperationCanceledException)
        {
            return null;
        }
    }

    public void Dispose() => _http.Dispose();
}