using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Reloaded.Hooks;
using Reloaded.Hooks.Definitions;
using Reloaded.Hooks.Definitions.X64;
using Reloaded.Mod.Interfaces.Internal;
using P5RGenSocialLinks.Memory;
using P5RGenSocialLinks.Server;

namespace P5RGenSocialLinks;

public class Mod : IModV1
{
    private ILoggerV2?        _logger;
    private LLMClient?        _llmClient;
    private DialogueBridge?   _bridge;
    private SocialLinkReader? _reader;

    // Adapts Reloaded's ILoggerV2 to DialogueBridge's internal ILogger contract.
    private sealed class LoggerAdapter : DialogueBridge.ILogger
    {
        private readonly ILoggerV2 _inner;
        internal LoggerAdapter(ILoggerV2 inner) => _inner = inner;
        public void WriteLine(string msg) => _inner.WriteLine(msg);
    }

    // Polling (fallback while hook is placeholder)
    private PeriodicTimer?          _timer;
    private Task?                   _pollTask;
    private CancellationTokenSource _cts = new();

    // Conversation-init detour
    private IHook<ConversationInitDelegate>? _conversationHook;

    [Function(CallingConventions.Microsoft)]
    public delegate void ConversationInitDelegate(nuint sessionPtr);

    public void Start(IModLoaderV1 loader)
    {
        _logger    = (ILoggerV2)loader.GetLogger();
        _llmClient = new LLMClient();

        nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
        _logger.WriteLine($"[P5RGenSocialLinks] Base: 0x{moduleBase:X}");

        _reader = new SocialLinkReader(moduleBase);
        _bridge = new DialogueBridge(_llmClient!, new LoggerAdapter(_logger!));

        TryActivateHook();
        StartPollLoop();

        _logger.WriteLine("[P5RGenSocialLinks] Started.");
    }

    private void TryActivateHook()
    {
        try
        {
            using var scanner = new FunctionScanner();
            nuint funcAddr = scanner.FindOrThrow(Signatures.BeginConversation);
            _logger!.WriteLine($"[P5RGenSocialLinks] Hook target: 0x{funcAddr:X}");

            var hooks = ReloadedHooks.Instance;
            _conversationHook = hooks
                .CreateHook<ConversationInitDelegate>(OnConversationInit, (long)funcAddr)
                .Activate();

            _logger.WriteLine("[P5RGenSocialLinks] Conversation hook active.");
        }
        catch (InvalidOperationException ex)
        {
            _logger!.WriteLine($"[P5RGenSocialLinks] Hook skipped (sig not found): {ex.Message}");
            _logger.WriteLine("[P5RGenSocialLinks] Falling back to poll loop.");
        }
    }

    private void OnConversationInit(nuint sessionPtr)
    {
        // Run original first so session fields are initialised before we read them
        _conversationHook!.OriginalFunction(sessionPtr);

        _logger?.WriteLine($"[P5RGenSocialLinks] sessionPtr=0x{sessionPtr:X}");
        _logger?.WriteLine($"[P5RGenSocialLinks] HexDump:{SocialLinkReader.HexDump(sessionPtr)}");

        SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(sessionPtr);
        if (snap is null)
            return;

        _logger?.WriteLine(
            $"[P5RGenSocialLinks] Hook: Confidant={snap.ConfidantId} Rank={snap.RankLevel}");

        _bridge!.DispatchAsync(snap, ContextBuilder.ReadAndBuild(snap));
    }

    // ── Poll loop (fallback) ───────────────────────────────────────────────

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
            if (_conversationHook is not null) break;  // hook is live, stop polling
            SocialLinkSnapshot? snap = _reader!.TryReadSnapshot();
            if (snap is not null)
                _logger!.WriteLine(
                    $"[P5RGenSocialLinks] Poll: Confidant={snap.ConfidantId}");
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    public void Suspend()
    {
        _cts.Cancel();
        _conversationHook?.Disable();
        _logger?.WriteLine("[P5RGenSocialLinks] Suspended.");
    }

    public void Resume()
    {
        _conversationHook?.Enable();
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