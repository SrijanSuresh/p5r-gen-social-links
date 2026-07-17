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
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(8);

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
                    unsafe { DialogueWriter.Write(snap.SessionBase + (nuint)P5ROffsets.DIALOGUE_BUFFER, text); }
                    _log.WriteLine($"[P5RGenSocialLinks] Injected: \"{text[..Math.Min(text.Length, 60)]}…\"");
                }
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
    }
}