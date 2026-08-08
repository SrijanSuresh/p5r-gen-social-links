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
    private ModLogger?        _modLog;
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
    private int                          _cmmExecFireCount;

    [Function(CallingConventions.Microsoft)]
    public delegate nint CmmExecEventDelegate();

    private GenConfig _cfg = new();

    public void Start(IModLoaderV1 loader)
    {
        _logger    = (ILoggerV2)loader.GetLogger();

        // Load user-editable runtime config from GenDialogue.json next to the DLL.
        string modDir = System.IO.Path.GetDirectoryName(
            System.Reflection.Assembly.GetExecutingAssembly().Location) ?? ".";
        _cfg    = GenConfig.Load(modDir);
        _modLog = new ModLogger(_logger, _cfg.LogLevel);
        _modLog.Always($"[P5RGenSocialLinks] Config: throttle={_cfg.ThrottleSeconds}s timeout={_cfg.TimeoutSeconds}s url={_cfg.ServerUrl} logLevel={_cfg.LogLevel}");

        _llmClient = new LLMClient(_cfg.ServerUrl);

        // Async health check — fires after a 2s delay to let the model finish loading.
        ServerHealthChecker.CheckAsync(_cfg.ServerUrl, msg => _logger.WriteLine(msg));

        nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
        _logger.WriteLine($"[P5RGenSocialLinks] Base: 0x{moduleBase:X}");

        _reader = new SocialLinkReader(moduleBase, _cfg.VerboseChain,
            msg => _logger!.WriteLine(msg));
        _bridge = new DialogueBridge(_llmClient!, new LoggerAdapter(_logger!), _cfg);

        // Hooks implementation provided by reloaded.sharedlib.hooks at runtime.
        loader.GetController<IReloadedHooks>()?.TryGetTarget(out _hooks);

        TryActivateHook();
        StartPollLoop();
        ModuleProbe.LogWriteInstructionModule(msg => _logger.WriteLine(msg));

        _logger.WriteLine($"[P5RGenSocialLinks] Started — hook:{(_hookActive ? "ON" : "OFF")} poll:ON");
    }

    // Tracks whether the hook wired successfully — logged on Start() completion.
    private bool _hookActive;

    private void TryActivateHook()
    {
        _logger!.WriteLine("[P5RGenSocialLinks] TryActivateHook: begin.");
        _logger!.WriteLine($"[P5RGenSocialLinks] IReloadedHooks available: {_hooks is not null}");

        try
        {
            using var scanner = new FunctionScanner();
            nuint funcAddr = scanner.FindOrThrow(Signatures.CmmExecEvent);
            _logger!.WriteLine($"[P5RGenSocialLinks] CmmExecEvent sig scan OK: 0x{funcAddr:X}");

            if (_hooks is null)
            {
                _logger!.WriteLine("[P5RGenSocialLinks] Hook skipped — IReloadedHooks null (install reloaded.sharedlib.hooks).");
                return;
            }

            _conversationHook = _hooks
                .CreateHook<CmmExecEventDelegate>(OnCmmExecEvent, (long)funcAddr)
                .Activate();

            _hookActive = true;
            _logger.WriteLine("[P5RGenSocialLinks] CmmExecEvent hook ACTIVE.");
        }
        catch (InvalidOperationException ex)
        {
            _logger!.WriteLine($"[P5RGenSocialLinks] Sig scan FAILED: {ex.Message}");
            _logger.WriteLine("[P5RGenSocialLinks] Falling back to poll loop only.");
        }
    }

    private nint OnCmmExecEvent()
    {
        // Run original first — it populates the session sub-object we are about to read.
        nint result = _conversationHook!.OriginalFunction();

        int fireCount = System.Threading.Interlocked.Increment(ref _cmmExecFireCount);

        try
        {
            if (!_reader!.TryResolve(out nuint session))
            {
                _logger?.WriteLine($"[P5RGenSocialLinks] CmmExecEvent #{fireCount}: session chain unresolved.");
                return result;
            }

            _logger?.WriteLine($"[P5RGenSocialLinks] CmmExecEvent #{fireCount} session=0x{session:X}");

            SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(session);
            if (snap is null) return result;

            _logger?.WriteLine(
                $"[P5RGenSocialLinks] CmmExec #{fireCount}: Confidant={snap.ConfidantId} Rank={snap.RankLevel} Scene={snap.SceneNumber}");
            bool dispatched = _bridge!.DispatchAsync(snap, ContextBuilder.Build(snap));
            if (!dispatched)
                _logger?.WriteLine($"[P5RGenSocialLinks] CmmExec #{fireCount}: throttled (< 3s since last dispatch).");
        }
        catch (Exception ex)
        {
            _logger?.WriteLine($"[P5RGenSocialLinks] OnCmmExecEvent error: {ex.Message}");
        }

        return result;
    }

    // ── Poll loop (fallback + struct discovery) ───────────────────────────

    private readonly StructDiffScanner    _diffScanner    = new();
    private readonly PointerFollowScanner _ptrFollower    = new();
    private readonly LineCounterMonitor   _lineMonitor    = new();
    private readonly HeapCounterScanner   _heapScanner    = new();
    private int _sessionTick;

    private void StartPollLoop()
    {
        _cts      = new CancellationTokenSource();
        _timer    = new PeriodicTimer(TimeSpan.FromMilliseconds(_cfg.PollIntervalMs));
        _pollTask = Task.Run(() => PollLoopAsync(_cts.Token));
    }

    private async Task PollLoopAsync(CancellationToken ct)
    {
        nuint lastSession = 0;

        while (await _timer!.WaitForNextTickAsync(ct))
        {
            if (!_reader!.TryResolve(out nuint session))
            {
                if (lastSession != 0)
                {
                    byte finalCount = _lineMonitor.CurrentValue();
                    _modLog!.Info(
                        $"[LineCounter] Session end: 0x{P5ROffsets.CMM_LINE_COUNTER_ADDR:X} = {finalCount}");

                    var hits = _heapScanner.FindIncreased(minDelta: 1, maxDelta: 50, maxResults: 50);
                    if (hits.Count > 0)
                    {
                        _modLog!.Info($"[HeapScan] Session-end: {hits.Count} byte(s) increased ≤50 — top candidates:");
                        foreach (string h in hits)
                            _modLog!.Info($"[HeapScan]   {h}");
                    }
                    else
                    {
                        _modLog!.Info("[HeapScan] Session-end: no increases ≤50 — counter outside 0x10000-0x20000000 or not yet moved.");
                    }
                    _heapScanner.Clear();

                    _modLog!.Info("[PtrFollow] Session-end cumulative deltas from capture baseline:");
                    var endCumul = _ptrFollower.CumulativeDiff();
                    if (endCumul.Count > 0)
                        foreach (string c in endCumul) _modLog!.Info($"[P5RGenSocialLinks]   {c}");
                    else
                        _modLog!.Info("[PtrFollow]   No cumulative increases in sub-objects.");

                    _diffScanner.Reset();
                    _ptrFollower.Reset();
                    _lineMonitor.Deactivate();
                    _bridge!.ResetSession();
                    _modLog!.Info("[P5RGenSocialLinks] Hang-out ended — session cleared.");
                }
                lastSession = 0;
                continue;
            }

            if (session != lastSession)
            {
                lastSession   = session;
                _sessionTick  = 0;
                _diffScanner.Reset();
                _ptrFollower.Reset();
                _lineMonitor.Activate();
                _heapScanner.TakeSnapshot(session, msg => _modLog!.Info(msg));
                LineCounterMonitor.Diagnose(msg => _modLog!.Info(msg));

                SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(session);
                if (snap is null) continue;

                _modLog!.Info(
                    $"[P5RGenSocialLinks] Hang-out: Confidant={snap.ConfidantId} Rank={snap.RankLevel} Scene={snap.SceneNumber} (0x{session:X})");

                // Locate the text pool once per session; bridge uses it for write-back.
                nuint poolBase = DialogueTextPoolFinder.Find(session, msg => _modLog!.Info(msg));
                _bridge!.SetPoolBase(poolBase);

                if (!_hookActive)
                    _bridge!.DispatchAsync(snap, ContextBuilder.Build(snap), lineIndex: 0);

                continue;
            }

            // Per-line trigger: fire LLM when dialogue line counter advances.
            if (_lineMonitor.HasAdvanced())
            {
                int lineIndex = (int)_lineMonitor.CurrentValue();
                SocialLinkSnapshot? lineSnap = SocialLinkReader.TryReadFromPtr(session);
                if (lineSnap is not null)
                {
                    // Retry pool scan on the first line advance if it missed at session start.
                    // BF engine loads text lazily; pool is guaranteed in memory once a line renders.
                    if (_bridge!.PoolBase == 0)
                    {
                        nuint retryPool = DialogueTextPoolFinder.Find(session, msg => _modLog!.Info(msg));
                        _bridge!.SetPoolBase(retryPool);
                    }

                    _modLog!.Info($"[P5RGenSocialLinks] Line advanced (counter={lineIndex}) — dispatching LLM.");
                    _bridge!.DispatchAsync(lineSnap, ContextBuilder.Build(lineSnap), lineIndex);
                }
            }

            // Mid-session checkpoint: ~20 s in (tick 40), after loading wipe finishes.
            // HeapScan looks for small byte increments in high heap near session struct.
            // PtrFollow cumulative diff reports totals since each sub-object was captured.
            _sessionTick++;
            if (_sessionTick == 40)
            {
                _modLog!.Info("[HeapScan] Mid-session (20s) — scanning high heap for small increments:");
                var midHits = _heapScanner.FindIncreased(minDelta: 1, maxDelta: 15, maxResults: 50);
                if (midHits.Count > 0)
                    foreach (string h in midHits) _modLog!.Info($"[HeapScan]   {h}");
                else
                    _modLog!.Info("[HeapScan]   No increases ≤15 — counter may be far from session struct.");

                _modLog!.Info("[PtrFollow] Mid-session cumulative deltas from capture baseline:");
                var midCumul = _ptrFollower.CumulativeDiff();
                if (midCumul.Count > 0)
                    foreach (string c in midCumul) _modLog!.Info($"[P5RGenSocialLinks]   {c}");
                else
                    _modLog!.Info("[PtrFollow]   No cumulative increases ≤100 in sub-objects.");
            }

            // Struct diff — passive discovery of per-line changing fields.
            if (_cfg.StructDiffEnabled)
            {
                _ptrFollower.Update(session, msg => _modLog!.Info($"[P5RGenSocialLinks] {msg}"));

                string? diff = _diffScanner.Diff(session);
                if (diff is not null)
                    _modLog!.Info($"[P5RGenSocialLinks] {diff}");

                foreach (string ptrDiff in _ptrFollower.Diff())
                    _modLog!.Info($"[P5RGenSocialLinks] {ptrDiff}");
            }
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    public void Suspend()
    {
        _cts.Cancel();
        _conversationHook?.Disable();
        _diffScanner.Reset();
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
        _diffScanner.Reset();
        _logger?.WriteLine("[P5RGenSocialLinks] Unloaded.");
    }

    public bool CanUnload()  => true;
    public bool CanSuspend() => true;
    public Action Disposing  => () => { };
}