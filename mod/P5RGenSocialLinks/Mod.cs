using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Reloaded.Hooks.Definitions;
using Reloaded.Hooks.Definitions.X64;
using Reloaded.Mod.Interfaces.Internal;
using P5RGenSocialLinks.Memory;
using P5RGenSocialLinks.Server;

namespace P5RGenSocialLinks;

public class Mod : IModV1
{
    private ILoggerV2?     _logger;
    private LLMClient?     _llmClient;
    private SocialLinkReader? _reader;

    // Polling (temporary — replaced by hook in Phase 2)
    private PeriodicTimer?           _timer;
    private Task?                    _pollTask;
    private CancellationTokenSource  _cts = new();

    // Hook (wired in Micro-step 4 once we have the function address)
    private IHook<ConversationInitDelegate>? _conversationHook;

    // Matches: void __fastcall SocialLink_BeginConversation(SocialLinkSession* session)
    // Placeholder — signature must be verified against the actual p5r.exe binary.
    [Function(CallingConventions.Microsoft)]
    public delegate void ConversationInitDelegate(nuint sessionPtr);

    public void Start(IModLoaderV1 loader)
    {
        _logger    = (ILoggerV2)loader.GetLogger();
        _llmClient = new LLMClient();

        nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
        _logger.WriteLine($"[P5RGenSocialLinks] Base: 0x{moduleBase:X}");

        _reader = new SocialLinkReader(moduleBase);

        StartPollLoop();
        _logger.WriteLine("[P5RGenSocialLinks] Started (polling mode).");
    }

    // ── Poll loop (runs until hook is active) ──────────────────────────────

    private void StartPollLoop()
    {
        _cts      = new CancellationTokenSource();
        _timer    = new PeriodicTimer(TimeSpan.FromMilliseconds(250));
        _pollTask = Task.Run(() => PollLoopAsync(_cts.Token));
    }

    private async Task PollLoopAsync(CancellationToken ct)
    {
        while (await _timer!.WaitForNextTickAsync(ct))
        {
            SocialLinkSnapshot? snap = _reader!.TryReadSnapshot();
            if (snap is null)
                continue;

            _logger!.WriteLine(
                $"[P5RGenSocialLinks] Confidant={snap.ConfidantId} Rank={snap.RankLevel} Line={snap.DialogueIndex}");

            // TODO Micro-step 4: call LLM and inject dialogue
        }
    }

    // ── Hook handler (Micro-step 4 wires this up) ─────────────────────────

    private void OnConversationInit(nuint sessionPtr)
    {
        _logger?.WriteLine($"[P5RGenSocialLinks] Hook fired: sessionPtr=0x{sessionPtr:X}");
        // TODO: read session from sessionPtr, call LLM, inject dialogue
        _conversationHook?.OriginalFunction(sessionPtr);
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    public void Suspend()
    {
        _cts.Cancel();
        _pollTask?.Wait(TimeSpan.FromSeconds(2));
        _logger?.WriteLine("[P5RGenSocialLinks] Suspended.");
    }

    public void Resume()
    {
        StartPollLoop();
        _logger?.WriteLine("[P5RGenSocialLinks] Resumed.");
    }

    public void Unload()
    {
        _cts.Cancel();
        _pollTask?.Wait(TimeSpan.FromSeconds(2));
        _timer?.Dispose();
        _llmClient?.Dispose();
        _conversationHook?.Disable();
        _logger?.WriteLine("[P5RGenSocialLinks] Unloaded.");
    }

    public bool CanUnload()  => true;
    public bool CanSuspend() => true;
    public Action Disposing  => () => { };
}