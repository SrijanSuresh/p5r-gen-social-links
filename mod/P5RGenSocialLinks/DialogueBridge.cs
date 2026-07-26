using System;
using System.Threading;
using System.Threading.Tasks;
using P5RGenSocialLinks.Memory;
using P5RGenSocialLinks.Server;

namespace P5RGenSocialLinks;

/// <summary>
/// Bridges a SocialLinkSnapshot to the Python LLM server and writes the
/// generated dialogue back into the game buffer.
/// Runs the HTTP call on the thread pool so the game hook returns immediately.
/// </summary>
internal sealed class DialogueBridge
{
    private readonly LLMClient _llm;
    private readonly ILogger   _log;

    // Hard timeout: if the LLM takes longer than this, we fall back to the
    // scripted dialogue that is already in the buffer.
    // 30s: first inference after model load triggers CUDA JIT warmup (~15s).
    // Steady-state responses arrive in ~2s; the timeout only matters on cold start.
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(30);

    internal interface ILogger
    {
        void WriteLine(string message);
    }

    internal DialogueBridge(LLMClient llm, ILogger log)
    {
        _llm = llm;
        _log = log;
    }

    /// <summary>
    /// Fires-and-forgets an LLM request. On success, overwrites the game buffer.
    /// On timeout or error, the original scripted dialogue remains untouched.
    /// </summary>
    internal void DispatchAsync(SocialLinkSnapshot snap, string contextText)
    {
        _ = Task.Run(async () =>
        {
            using var cts = new CancellationTokenSource(Timeout);
            try
            {
                var request = new GenerateRequest
                {
                    ConfidantId   = snap.ConfidantId,
                    Rank          = snap.RankLevel,
                    Context       = contextText,
                    CharacterName = $"Confidant #{snap.ConfidantId}",
                };

                string text = await _llm.GenerateAsync(request, cts.Token);

                if (!string.IsNullOrWhiteSpace(text))
                {
                    // TODO: dialogue write-back requires locating the text buffer offset.
                    // For now, log the generated response so we can verify E2E flow.
                    _log.WriteLine($"[P5RGenSocialLinks] LLM: \"{text[..Math.Min(text.Length, 80)]}\"");
                }
            }
            catch (InferenceInFlightException)
            {
                // Server busy — scripted dialogue stays; no log spam.
            }
            catch (OperationCanceledException)
            {
                _log.WriteLine("[P5RGenSocialLinks] LLM timeout â€” keeping scripted dialogue.");
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                _log.WriteLine($"[P5RGenSocialLinks] LLM error: {ex.Message}");
            }
        });
    }
}