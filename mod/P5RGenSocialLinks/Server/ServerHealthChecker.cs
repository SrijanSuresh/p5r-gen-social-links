using System;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace P5RGenSocialLinks.Server;

/// <summary>
/// Pings the Python server's /health endpoint at startup and logs its state.
/// Runs fire-and-forget so it never blocks mod initialization.
/// </summary>
internal static class ServerHealthChecker
{
    private static readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(5) };

    internal static void CheckAsync(string serverUrl, Action<string> log)
    {
        _ = Task.Run(async () =>
        {
            await Task.Delay(TimeSpan.FromSeconds(2));  // give server time to finish loading model
            try
            {
                using var resp = await _http.GetAsync($"{serverUrl.TrimEnd('/')}/health");
                string body    = await resp.Content.ReadAsStringAsync();
                using var doc  = JsonDocument.Parse(body);
                string status  = doc.RootElement.GetProperty("status").GetString() ?? "unknown";
                log($"[P5RGenSocialLinks] Server /health → {status} (HTTP {(int)resp.StatusCode})");
            }
            catch (HttpRequestException ex)
            {
                log($"[P5RGenSocialLinks] Server unreachable: {ex.Message}");
                log("[P5RGenSocialLinks] Start server: cd server && python main.py");
            }
            catch (Exception ex)
            {
                log($"[P5RGenSocialLinks] Health check error: {ex.Message}");
            }
        });
    }
}
