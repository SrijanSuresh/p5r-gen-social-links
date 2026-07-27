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

            // Discovery pass: log the first hit of the broad LEA pattern so we
            // can inspect it in CE's disassembler and build a precise signature.
            nuint? candidate = scanner.TryFindFirst(Signatures.BeginConversation);
            if (candidate is nuint addr)
            {
                _logger!.WriteLine(
                    $"[P5RGenSocialLinks] LEA 0x62B8 candidate: 0x{addr:X} " +
                    $"(offset from base: 0x{addr - (nuint)System.Diagnostics.Process.GetCurrentProcess().MainModule!.BaseAddress:X})");
                _logger.WriteLine("[P5RGenSocialLinks] Open CE disassembler at that address to find function start.");
            }
            else
            {
                _logger!.WriteLine("[P5RGenSocialLinks] Pattern 48 8D ?? B8 62 00 00 not found — 0x62B8 offset may differ in this build.");
            }
        }
        catch (Exception ex)
        {
            _logger!.WriteLine($"[P5RGenSocialLinks] Scanner error: {ex.Message}");
        }
    }

    private void OnConversationInit(nuint sessionPtr)
    {
        // Run original first so session fields are initialised before we read them
        _conversationHook!.OriginalFunction(sessionPtr);

        _logger?.WriteLine($"[P5RGenSocialLinks] sessionPtr=0x{sessionPtr:X}");

        if (sessionPtr == 0)
        {
            _logger?.WriteLine("[P5RGenSocialLinks] sessionPtr is null — skipping.");
            return;
        }

        try
        {
            // LEA RBX,[RCX+0x62B8] in the prologue tells us the SL session struct
            // begins 0x62B8 bytes into the manager object that RCX (sessionPtr) points to.
            const nuint SESSION_OFFSET = 0x62B8;
            nuint slSession = sessionPtr + SESSION_OFFSET;
            _logger?.WriteLine($"[P5RGenSocialLinks] slSession=0x{slSession:X}");
            _logger?.WriteLine($"[P5RGenSocialLinks] HexDump@slSession:{SocialLinkReader.HexDump(slSession)}");

            SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(slSession);
            if (snap is null)
                return;

            _logger?.WriteLine(
                $"[P5RGenSocialLinks] Hook: Confidant={snap.ConfidantId} Rank={snap.RankLevel}");

            _bridge!.DispatchAsync(snap, ContextBuilder.ReadAndBuild(snap));
        }
        catch (Exception ex)
        {
            _logger?.WriteLine($"[P5RGenSocialLinks] OnConversationInit error: {ex.Message}");
        }
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