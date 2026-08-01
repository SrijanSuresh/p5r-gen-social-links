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
    private readonly LLMClient     _llm;
    private readonly ILogger       _log;
    private readonly TimeSpan      _timeout;
    private readonly TimeSpan      _minInterval;
    private readonly SessionHistory _history = new();

    private DateTimeOffset _lastDispatch = DateTimeOffset.MinValue;

    internal interface ILogger
    {
        void WriteLine(string message);
    }

    internal DialogueBridge(LLMClient llm, ILogger log, GenConfig? cfg = null)
    {
        _llm         = llm;
        _log         = log;
        _timeout     = TimeSpan.FromSeconds(cfg?.TimeoutSeconds  ?? 30.0);
        _minInterval = TimeSpan.FromSeconds(cfg?.ThrottleSeconds ?? 3.0);
    }

    /// <summary>
    /// Fires-and-forgets an LLM request with leading-edge throttling.
    /// Returns false immediately if called within MinDispatchInterval of the last dispatch.
    /// On success, logs the generated response (write-back pending offset discovery).
    /// On timeout or error, the original scripted dialogue remains untouched.
    /// </summary>
    internal bool DispatchAsync(SocialLinkSnapshot snap, string contextText)
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        if (now - _lastDispatch < _minInterval)
            return false;

        _lastDispatch = now;

        // Include prior LLM lines this session as context for continuity.
        string priorCtx = _history.BuildPriorContext(snap.SessionBase);
        string fullCtx  = string.IsNullOrEmpty(priorCtx)
            ? contextText
            : $"{contextText} {priorCtx}";

        _ = Task.Run(async () =>
        {
            using var cts = new CancellationTokenSource(_timeout);
            try
            {
                var request = new GenerateRequest
                {
                    ConfidantId   = snap.ConfidantId,
                    Rank          = snap.RankLevel,
                    Context       = fullCtx,
                    CharacterName = ConfidantNames.Resolve(snap.ConfidantId),
                };

                string text = await _llm.GenerateAsync(request, cts.Token);

                if (!string.IsNullOrWhiteSpace(text))
                {
                    bool isNew = _history.RecordResponse(snap.SessionBase, text);
                    if (!isNew)
                    {
                        _log.WriteLine("[P5RGenSocialLinks] LLM: duplicate response suppressed.");
                        return;
                    }
                    // TODO: dialogue write-back requires locating the text buffer offset.
                    // For now, log the generated response so we can verify E2E flow.
                    _log.WriteLine($"[P5RGenSocialLinks] LLM: \"{text[..Math.Min(text.Length, 120)]}\"");
                }
            }
            catch (InferenceInFlightException)
            {
                // Server busy — scripted dialogue stays; no log spam.
            }
            catch (OperationCanceledException)
            {
                _log.WriteLine("[P5RGenSocialLinks] LLM timeout — keeping scripted dialogue.");
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                _log.WriteLine($"[P5RGenSocialLinks] LLM error: {ex.Message}");
            }
        });

        return true;
    }
}