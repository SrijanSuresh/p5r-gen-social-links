using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Reloaded.Hooks.Definitions;
using Reloaded.Hooks.Definitions.X64;
using Reloaded.Mod.Interfaces;
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

    // CMM_EXEC_EVENT detour — fires when a Social Link community event executes.
    // The native function reads from globals (no meaningful parameters).
    private IHook<CmmExecEventDelegate>? _conversationHook;
    private IReloadedHooks?              _hooks;

    [Function(CallingConventions.Microsoft)]
    public delegate nint CmmExecEventDelegate();

    public void Start(IModLoaderV1 loader)
    {
        _logger    = (ILoggerV2)loader.GetLogger();
        _llmClient = new LLMClient();

        nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
        _logger.WriteLine($"[P5RGenSocialLinks] Base: 0x{moduleBase:X}");

        _reader = new SocialLinkReader(moduleBase);
        _bridge = new DialogueBridge(_llmClient!, new LoggerAdapter(_logger!));

        // Hooks implementation provided by reloaded.sharedlib.hooks at runtime.
        loader.GetController<IReloadedHooks>()?.TryGetTarget(out _hooks);

        TryActivateHook();
        StartPollLoop();

        _logger.WriteLine("[P5RGenSocialLinks] Started.");
    }

    private void TryActivateHook()
    {
        try
        {
            using var scanner = new FunctionScanner();
            nuint funcAddr = scanner.FindOrThrow(Signatures.CmmExecEvent);
            _logger!.WriteLine($"[P5RGenSocialLinks] CmmExecEvent hook target: 0x{funcAddr:X}");

            if (_hooks is null)
            {
                _logger!.WriteLine("[P5RGenSocialLinks] IReloadedHooks not available — is reloaded.sharedlib.hooks installed?");
                return;
            }
            _conversationHook = _hooks
                .CreateHook<CmmExecEventDelegate>(OnCmmExecEvent, (long)funcAddr)
                .Activate();

            _logger.WriteLine("[P5RGenSocialLinks] CmmExecEvent hook active.");
        }
        catch (InvalidOperationException ex)
        {
            _logger!.WriteLine($"[P5RGenSocialLinks] Hook skipped: {ex.Message}");
            _logger.WriteLine("[P5RGenSocialLinks] Falling back to poll loop.");
        }
    }

    private nint OnCmmExecEvent()
    {
        // Run original first — it populates the session sub-object we are about to read.
        nint result = _conversationHook!.OriginalFunction();

        try
        {
            if (!_reader!.TryResolve(out nuint session))
            {
                _logger?.WriteLine("[P5RGenSocialLinks] CmmExecEvent: session chain unresolved.");
                return result;
            }

            _logger?.WriteLine($"[P5RGenSocialLinks] CmmExecEvent session=0x{session:X}");
            _logger?.WriteLine($"[P5RGenSocialLinks] HexDump:{SocialLinkReader.HexDump(session)}");

            SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(session);
            if (snap is null) return result;

            _logger?.WriteLine(
                $"[P5RGenSocialLinks] Confidant={snap.ConfidantId} Rank={snap.RankLevel}");
            _bridge!.DispatchAsync(snap, ContextBuilder.ReadAndBuild(snap));
        }
        catch (Exception ex)
        {
            _logger?.WriteLine($"[P5RGenSocialLinks] OnCmmExecEvent error: {ex.Message}");
        }

        return result;
    }

    // ── Poll loop (fallback) ───────────────────────────────────────────────

    private void StartPollLoop()
    {
        _cts      = new CancellationTokenSource();
        _timer    = new PeriodicTimer(TimeSpan.FromSeconds(1));
        _pollTask = Task.Run(() => PollLoopAsync(_cts.Token));
    }

    private async Task PollLoopAsync(CancellationToken ct)
    {
        nuint lastSession = 0;
        while (await _timer!.WaitForNextTickAsync(ct))
        {
            if (!_reader!.TryResolve(out nuint session)) { lastSession = 0; continue; }
            if (session == lastSession) continue;  // only log on change
            lastSession = session;
            _logger!.WriteLine($"[P5RGenSocialLinks] Poll: session=0x{session:X}");
            _logger!.WriteLine($"[P5RGenSocialLinks] Poll HexDump:{SocialLinkReader.HexDump(session)}");
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