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
    private readonly LLMClient      _llm;
    private readonly ILogger        _log;
    private readonly TimeSpan       _timeout;
    private readonly TimeSpan       _minInterval;
    private readonly SessionHistory _history = new();

    private DateTimeOffset _lastDispatch = DateTimeOffset.MinValue;
    private nuint          _poolBase;    // set once per hang-out by SetPoolBase()

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

    /// <summary>Current text pool base; 0 means write-back is disabled for this session.</summary>
    internal nuint PoolBase => _poolBase;

    /// <summary>
    /// Stores the text pool base address found at hang-out start.
    /// The bridge will write LLM responses into pool slot (lineIndex + 1).
    /// Pass 0 to disable write-back for this session.
    /// </summary>
    internal void SetPoolBase(nuint poolBase) => _poolBase = poolBase;

    /// <summary>
    /// Fires-and-forgets an LLM request with leading-edge throttling.
    /// Returns false immediately if called within MinDispatchInterval of the last dispatch.
    /// On success, writes the response into text pool slot (lineIndex + 1) when a pool
    /// base is available; falls back to console logging otherwise.
    /// On timeout or error, the original scripted dialogue remains untouched.
    /// </summary>
    internal bool DispatchAsync(SocialLinkSnapshot snap, string contextText, int lineIndex = 0)
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        if (now - _lastDispatch < _minInterval)
            return false;

        _lastDispatch = now;

        // Capture mutable state before the async hop — lineIndex and poolBase must not be
        // read from fields after awaiting, as they can change on the next poll tick.
        nuint capturedPool      = _poolBase;
        int   capturedLineIndex = lineIndex;

        // Include prior LLM lines this session as context for continuity.
        // Trim to MaxContextChars so we never exceed Pydantic's max_length=1024.
        const int MaxContextChars = 1000;
        string priorCtx = _history.BuildPriorContext(snap.SessionBase);
        string combined = string.IsNullOrEmpty(priorCtx)
            ? contextText
            : $"{contextText} {priorCtx}";
        string fullCtx = combined.Length > MaxContextChars
            ? combined[..MaxContextChars]
            : combined;

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

                    _log.WriteLine($"[P5RGenSocialLinks] LLM: \"{text[..Math.Min(text.Length, 120)]}\"");

                    // Write-back: overwrite the NEXT line (capturedLineIndex + 1) so it's
                    // ready before the player presses confirm on the current line.
                    if (capturedPool != 0)
                    {
                        int targetSlot = capturedLineIndex + 1;
                        bool wrote = Memory.DialogueWriter.WriteAtLineIndex(capturedPool, targetSlot, text);
                        _log.WriteLine(wrote
                            ? $"[P5RGenSocialLinks] Write-back: wrote {text.Length} chars to pool slot {targetSlot}."
                            : $"[P5RGenSocialLinks] Write-back: FAILED for slot {targetSlot} (pool=0x{capturedPool:X}).");
                    }
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

    /// <summary>
    /// Clears session history and resets the dispatch throttle.
    /// Call when the hang-out ends (session pointer drops to 0) to ensure the next
    /// hang-out starts with a clean slate and no throttle carry-over.
    /// </summary>
    internal void ResetSession()
    {
        _history.Reset();
        _lastDispatch = DateTimeOffset.MinValue;
        _poolBase     = 0;
    }
}