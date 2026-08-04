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

    private sealed class LoggerAdapter : DialogueBridge.ILogger
    {
        private readonly ILoggerV2 _inner;
        internal LoggerAdapter(ILoggerV2 inner) => _inner = inner;
        public void WriteLine(string msg) => _inner.WriteLine(msg);
    }

    private PeriodicTimer?          _timer;
    private Task?                   _pollTask;
    private CancellationTokenSource _cts = new();

    // CmmExecEvent fires once at hang-out init (not per-line, kept for session context).
    private IHook<CmmExecEventDelegate>? _conversationHook;
    private IReloadedHooks?              _hooks;
    private int                          _cmmExecFireCount;

    [Function(CallingConventions.Microsoft)]
    public delegate nint CmmExecEventDelegate();

    // Per-line hook: intercepts FUN_1405a8570 (the REP MOVSB inner copy function).
    // The dialogue system calls this DIRECTLY via function pointer (0x158131be0),
    // bypassing the outer dispatcher FUN_1405a8590 that uses standard registers.
    // Custom convention: RCX=dst, R10=src (non-standard!), R8=count.
    [Function(
        new FunctionAttribute.Register[] {
            FunctionAttribute.Register.rcx,
            FunctionAttribute.Register.r10,
            FunctionAttribute.Register.r8 },
        FunctionAttribute.Register.rax, false)]
    private delegate void MemcpyInnerDelegate(nuint dst, nuint src, nuint count);

    private IHook<MemcpyInnerDelegate>? _memcpyHook;

    // Dialogue heap sits above 256 GB; CLR/runtime copies are all below 4 GB.
    private static readonly nuint HeapLow = unchecked((nuint)0x4000000000UL);

    private GenConfig _cfg = new();

    public void Start(IModLoaderV1 loader)
    {
        _logger = (ILoggerV2)loader.GetLogger();

        string modDir = System.IO.Path.GetDirectoryName(
            System.Reflection.Assembly.GetExecutingAssembly().Location) ?? ".";
        _cfg    = GenConfig.Load(modDir);
        _modLog = new ModLogger(_logger, _cfg.LogLevel);
        _modLog.Always($"[P5RGenSocialLinks] Config: throttle={_cfg.ThrottleSeconds}s timeout={_cfg.TimeoutSeconds}s url={_cfg.ServerUrl} logLevel={_cfg.LogLevel}");

        _llmClient = new LLMClient(_cfg.ServerUrl);
        ServerHealthChecker.CheckAsync(_cfg.ServerUrl, msg => _logger.WriteLine(msg));

        nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
        _logger.WriteLine($"[P5RGenSocialLinks] Base: 0x{moduleBase:X}");

        _reader = new SocialLinkReader(moduleBase, _cfg.VerboseChain,
            msg => _logger!.WriteLine(msg));
        _bridge = new DialogueBridge(_llmClient!, new LoggerAdapter(_logger!), _cfg);

        loader.GetController<IReloadedHooks>()?.TryGetTarget(out _hooks);

        TryActivateHook();
        SetupMemcpyHook();
        StartPollLoop();

        _logger.WriteLine($"[P5RGenSocialLinks] Started — hook:{(_hookActive ? "ON" : "OFF")} poll:ON");
    }

    private bool _hookActive;

    private void SetupMemcpyHook()
    {
        if (_hooks is null)
        {
            _logger!.WriteLine("[P5RGenSocialLinks] Memcpy inner hook skipped — IReloadedHooks null.");
            return;
        }
        nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
        // Hook the REP MOVSB inner function directly (not the outer dispatcher).
        // The dialogue system reaches FUN_1405a8570 via a function pointer, bypassing
        // FUN_1405a8590, so hooking the inner function catches all paths.
        nuint addr = moduleBase + 0x5A8570;
        _memcpyHook = _hooks.CreateHook<MemcpyInnerDelegate>(OnGameMemcpy, (long)addr).Activate();
        _logger!.WriteLine($"[P5RGenSocialLinks] Memcpy inner hook ACTIVE at 0x{addr:X}");
    }

    private unsafe void OnGameMemcpy(nuint dst, nuint src, nuint count)
    {
        // Diagnostic logging disabled — memcpy hook served its purpose (confirmed the
        // BF script is loaded once before session detection, not per-line).
        // [BFLine] probe via ProbeBfLine() is the active diagnostic path now.
        _memcpyHook!.OriginalFunction(dst, src, count);
    }

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
                _logger!.WriteLine("[P5RGenSocialLinks] Hook skipped — IReloadedHooks null.");
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
        nint result    = _conversationHook!.OriginalFunction();
        int  fireCount = Interlocked.Increment(ref _cmmExecFireCount);

        try
        {
            if (!_reader!.TryResolve(out nuint session))
            {
                _logger?.WriteLine($"[P5RGenSocialLinks] CmmExecEvent #{fireCount}: session chain unresolved.");
                return result;
            }

            SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(session);
            if (snap is null) return result;

            _modLog!.Info(
                $"[P5RGenSocialLinks] CmmExec #{fireCount}: Confidant={snap.ConfidantId} Rank={snap.RankLevel} Scene={snap.SceneNumber}");

            // Retry text pool discovery on first few fires — the BF interpreter may not have
            // decompressed the script into heap memory at raw session-detection time, but it
            // is guaranteed to be loaded by the time CmmExecEvent fires for the first line.
            if (_bridge!.PoolBase == 0 && fireCount <= 5)
            {
                nuint pool = Memory.DialogueTextPoolFinder.Find(session, msg => _modLog!.Info(msg));
                if (pool != 0)
                {
                    _bridge!.SetPoolBase(pool);
                    _modLog!.Info($"[P5RGenSocialLinks] Text pool found on fire #{fireCount}: 0x{pool:X}");
                }
            }

            bool dispatched = _bridge!.DispatchAsync(snap, ContextBuilder.Build(snap), lineIndex: fireCount);
            if (!dispatched)
                _modLog!.Info($"[P5RGenSocialLinks] CmmExec #{fireCount}: throttled.");
        }
        catch (Exception ex)
        {
            _logger?.WriteLine($"[P5RGenSocialLinks] OnCmmExecEvent error: {ex.Message}");
        }

        return result;
    }

    // ── Poll loop — session lifecycle + text pool discovery ───────────────

    private readonly StructDiffScanner _diffScanner = new();
    private int    _poolFindRetries;
    private ushort _lastBfPc;       // change-detection: fires only when PC moves

    /// <summary>
    /// Fires whenever the BF program counter (uint16 @ session+0x20) advances.
    /// Session struct layout varies by scene type, so instead of hardcoding a
    /// pointer offset, we scan EVERY 8-byte heap address in the first 512 bytes
    /// of the session struct and probe [ptr + pc] for printable text.
    /// The BF script buffer is the one where [ptr + pc] contains ≥8 printable bytes.
    /// </summary>
    private unsafe void ProbeBfLine(nuint session)
    {
        const int scanBytes = 512;

        // BF program counter — confirmed stable at session+0x20
        if (!Memory.MemoryGuard.IsReadable(session + 0x20, 2)) return;
        ushort pc = *(ushort*)((byte*)session + 0x20);
        if (pc == _lastBfPc) return;   // no new line yet
        _lastBfPc = pc;

        if (!Memory.MemoryGuard.IsReadable(session, scanBytes)) return;
        byte* sp = (byte*)session;

        bool anyHit = false;
        for (int off = 0; off + 8 <= scanBytes; off += 8)
        {
            nuint ptr = *(nuint*)(sp + off);
            if (ptr < HeapLow) continue;

            nuint lineAddr = ptr + pc;
            if (!Memory.MemoryGuard.IsReadable(lineAddr, 64)) continue;
            byte* b = (byte*)lineAddr;

            // Count printable bytes; skip if this looks like binary (animation/texture)
            int printable = 0;
            for (int i = 0; i < 64; i++)
                if (b[i] >= 0x20 && b[i] <= 0x7E) printable++;
            if (printable < 8) continue;

            var hex  = new System.Text.StringBuilder(24);
            for (int i = 0; i < 8; i++) hex.Append($"{b[i]:X2} ");

            var text = new System.Text.StringBuilder(128);
            for (int i = 0; i < 64; i++)
                if (b[i] >= 0x20 && b[i] <= 0x7E) text.Append((char)b[i]);

            _modLog!.Info(
                $"[BFLine] pc=0x{pc:X4} [sess+0x{off:X3}+pc] [{hex}]: \"{text}\"");
            anyHit = true;
        }

        if (!anyHit)
            _modLog!.Info($"[BFLine] pc=0x{pc:X4}: no heap-ptr+pc with ≥8 printable bytes");
    }

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
                    _diffScanner.Reset();
                    _bridge!.ResetSession();
                    _poolFindRetries = 0;
                    _lastBfPc        = 0;
                    _modLog!.Info("[P5RGenSocialLinks] Hang-out ended — session cleared.");
                }
                lastSession = 0;
                continue;
            }

            if (session != lastSession)
            {
                lastSession = session;
                _diffScanner.Reset();
                _poolFindRetries = 0;

                SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(session);
                if (snap is null) continue;

                _modLog!.Info(
                    $"[P5RGenSocialLinks] Hang-out: Confidant={snap.ConfidantId} Rank={snap.RankLevel} Scene={snap.SceneNumber} (0x{session:X})");

                // Attempt 0: probe at detection time (struct may not be populated yet).
                // diagnoseOnFail=false — retries will diagnose if all 10 attempts fail.
                nuint poolBase = Memory.DialogueTextPoolFinder.Find(
                    session, msg => _modLog!.Info(msg), diagnoseOnFail: false);
                _bridge!.SetPoolBase(poolBase);
                if (poolBase != 0)
                    Memory.DialogueTextPoolFinder.LogPoolContents(poolBase, 8, msg => _modLog!.Info(msg));

                // Fallback dispatch if hook isn't active.
                if (!_hookActive)
                    _bridge!.DispatchAsync(snap, ContextBuilder.Build(snap), lineIndex: 0);

                continue;
            }

            // Retry pool discovery on subsequent ticks — the text pool is allocated lazily by
            // the BF interpreter (not at session start), so a one-shot probe at detection time
            // consistently misses it. Retry up to 10 ticks; only emit the verbose Diag on
            // the final attempt so the log stays clean during the intermediate tries.
            if (_bridge!.PoolBase == 0 && _poolFindRetries < 10)
            {
                _poolFindRetries++;
                bool isFinalRetry = _poolFindRetries == 10;
                nuint pool = Memory.DialogueTextPoolFinder.Find(
                    lastSession,
                    msg => _modLog!.Info(msg),
                    diagnoseOnFail: isFinalRetry);

                if (pool != 0)
                {
                    _bridge!.SetPoolBase(pool);
                    _modLog!.Info($"[P5RGenSocialLinks] Text pool found on poll retry #{_poolFindRetries}: 0x{pool:X}");
                    Memory.DialogueTextPoolFinder.LogPoolContents(pool, 8, msg => _modLog!.Info(msg));
                }
                else if (isFinalRetry)
                {
                    _modLog!.Info("[TextPoolFinder] No pool found after 10 retries — write-back disabled.");
                }
            }

            // Passive struct discovery — disabled by default, enable in GenDialogue.json.
            if (_cfg.StructDiffEnabled)
            {
                string? diff = _diffScanner.Diff(session);
                if (diff is not null)
                    _modLog!.Info($"[P5RGenSocialLinks] {diff}");
            }

            // BF line probe: read current dialogue text from bfBase+pc on every tick.
            // Fires only when the PC moves (= a new dialogue line is loaded), so it's
            // quiet between advances and doesn't need StructDiffEnabled.
            ProbeBfLine(session);
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    public void Suspend()
    {
        _cts.Cancel();
        _conversationHook?.Disable();
        _memcpyHook?.Disable();
        _diffScanner.Reset();
        _logger?.WriteLine("[P5RGenSocialLinks] Suspended.");
    }

    public void Resume()
    {
        _conversationHook?.Enable();
        _memcpyHook?.Enable();
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
        _memcpyHook?.Disable();
        _diffScanner.Reset();
        _logger?.WriteLine("[P5RGenSocialLinks] Unloaded.");
    }

    public bool CanUnload()  => true;
    public bool CanSuspend() => true;
    public Action Disposing  => () => { };
}
