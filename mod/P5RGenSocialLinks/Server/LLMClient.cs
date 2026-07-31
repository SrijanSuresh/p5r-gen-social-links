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

            response.EnsureSuccessStatusCode();
            string body = await response.Content.ReadAsStringAsync(ct);
            var result  = JsonSerializer.Deserialize<GenerateResponse>(body, _jsonOpts);
            return result?.Text ?? string.Empty;
        }
    }

    public void Dispose() => _http.Dispose();
}