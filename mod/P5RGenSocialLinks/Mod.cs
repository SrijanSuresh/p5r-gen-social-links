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

public partial class Mod : IModV1
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

    // Address of the interpreter's MOVZX, resolved by signature at load. Zero means the
    // scan did not find it and the pool heuristic remains the only path to the text.
    private nuint                        _msgByteFetch;
    private MsgInterpreterWatch?         _msgWatch;

    // Last record pointer reported to the log. The hook fires once per character, so the
    // interesting event is the value changing, not the value existing.
    private nuint                        _lastWatchedRecord;

    // Scratch for previewing a watched record. Separate from _scanBuf so a preview can
    // never disturb a pool scan mid-walk, even though both run on the poll thread today.
    private readonly byte[]              _recordBuf = new byte[128];
    private readonly byte[]              _mirrorBuf = new byte[128];
    private readonly byte[]              _headerBuf = new byte[BmdMessage.HeaderBytes];
    private bool                         _speakerAudited;

    // Guards the pool write path. Flushing now happens from two places — the poll tick
    // and the thread-pool continuation that finishes a generation — and both walk the
    // plan and write into game memory. Arming takes it too, because replacing _heapPools
    // under a flush would have it writing through a base it is halfway done with.
    private readonly object              _writeLock = new();

    // Signed distance from a record in an armed pool to the same record in the copy the
    // speech bubble reads, discovered by watching both be read within a millisecond of
    // each other with identical text.
    //
    // The game keeps every scene's dialogue in two places at a constant offset, and which
    // one the bubble reads is not something ranking can determine — measured at
    // 0x3C60B8F0 one run and 0x2E67F270 the next. Twin arming by content match found it
    // sometimes and missed it in the run that produced this field: one region armed, and
    // every write landed in the text log while the bubble kept the script.
    private long                         _twinDelta;

    // True between a hang-out starting and ending. Arming reads the interpreter's own
    // pointer, and the interpreter serves every string in the game, so this is what
    // keeps a menu label from being mistaken for the scene's dialogue pool.
    private volatile bool                _sessionActive;

    // Consecutive dialogue reads landing outside the armed pool. Two means the scene
    // moved to a different pool; one is a menu or a name plate passing through.
    private int                          _outsideReads;
    private (nuint Addr, string Text)    _lastWatched;

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

    // BF opcode dispatcher hook: FUN_14024EE00.
    // Called for every BF instruction. RCX=channel(0x2B), RDX=opcode type byte,
    // R8=opcode_struct[+0x08], R9=opcode_struct[+0x10].
    // When RDX==5 this is a dialogue instruction — log R8/R9 to find the text pointer.
    // Return type is nuint (not void) — the dispatcher may return a status value
    // that callers (e.g. FUN_141844f20) check. Declaring void leaves RAX garbage.
    [Function(CallingConventions.Microsoft)]
    private delegate nuint BfOpcodeDispatchDelegate(nuint channel, nuint typeAndFlags,
                                                     nuint arg2, nuint arg3);

    private IHook<BfOpcodeDispatchDelegate>? _bfDispatchHook;

    // Dialogue heap sits above 256 GB; CLR/runtime copies are all below 4 GB.
    private static readonly nuint HeapLow = unchecked((nuint)0x4000000000UL);

    // Cached BF script buffer address — found by TryFindBfBuffer() on any tick,
    // then used by ProbeBfLine() for the rest of the session.
    private nuint  _bfBufferBase;
    private int    _bfBufferOff;   // session struct offset where we found it (for logging)

    // BMD text pool: the first page in the ±512KB bfBase vicinity that has ≥5 English
    // dialogue strings (null-terminated, ≥2 spaces, ≥10 printable chars). Found once
    // per session after bfBase is confirmed; written via VirtualProtect(PAGE_WRITECOPY).
    private nuint _bmdTextPool;
    private bool  _bmdScanDoneV2;
    private int   _bmdScanAttempts;

    // (offset, original length) of every dialogue-looking string in _bmdTextPool.
    // Captured once at discovery, BEFORE any write — see CapturePoolSlots for why.
    private (int Off, int Len)[]? _poolSlots;

    // Byte length of the pool region — the MSG1 file size when found via magic,
    // otherwise a single page. Used to size the VirtualProtect call.
    private int _bmdPoolLen = 0x1000;

    // The MSG1 files reachable from bfBase are all global item/skill tables, so the pool
    // write is now a one-shot probe of the write path rather than a per-message action:
    // if skill descriptions in-game show the mock text, write→render is proven.
    private bool _poolWriteDone;

    // True when the pool is the heap dialogue allocation rather than a mapped MSG1 file.
    // The heap pool is the correct target, so it is rewritten on every message; the
    // one-shot guard applies only to the MSG1 item tables.
    private bool _poolIsHeap;

    // Every high-scoring heap region, not just the winner. Ranking by content picked a
    // shader block first and, once fixed, still would not have found the live scene: the
    // confirmed address 0x41DD7F6389 fell outside every region the scan surfaced. Writing
    // to all strong candidates removes the need to guess correctly on the first try.
    private readonly System.Collections.Generic.List<(nuint Base, int Len, (int Off, int Len)[] Slots)>
        _heapPools = new();

    // Slots grouped into records, one list per armed pool, and the index of the record
    // each pool is expected to render next.
    //
    // Writing every record with one generated line is what makes the text log repeat the
    // same sentence at three different widths: those are three records, all overwritten.
    // Measured timing says we do not have to. The MSG dispatch fires ~3s before the
    // renderer reads the record and the write completes ~1.4s before it, so there is room
    // to write exactly the record that is about to be drawn.
    private readonly System.Collections.Generic.List<System.Collections.Generic.List<(int Start, int Count)>>
        _poolRecords = new();
    private readonly System.Collections.Generic.List<int> _poolNextRecord = new();

    // The scene's records as a plan: capacity and original line per record, plus where
    // each one is in its life cycle. Built once from region 0 at arm time, because every
    // armed region is a copy of the same script with identical record widths.
    //
    // This is what makes generation independent of the player. All of it is knowable
    // before a single line is displayed; waiting for a dispatch event to learn a record's
    // size or its scripted text was only ever necessary while the buffer was a guess.
    private readonly System.Collections.Generic.List<RecordPlan> _plan = new();

    // UTF-16 hunt state. _utf16CpyLogged caps the memcpy log; the sweep cursor lets a
    // multi-GB heap scan resume across ticks instead of stalling one.
    private int   _utf16CpyLogged;
    private nuint _heapSweepCursor;
    private int   _utf16SweepHits;
    private bool  _heapSweepDone;

    // Every game object pointer observed so far lands in 0x42xxxxxxxx, while the linear
    // sweep starts at HeapLow (0x4000000000) and spends whole sessions crawling through
    // DirectInput and font data around 0x4188xxxxxx without ever reaching game memory.
    // Confirmed dialogue addresses were 0x41DD7F6389 and 0x42102CAAA9, so the sweep must
    // start below 0x42 — the system-string filter handles the DirectInput and font data
    // that sits around 0x4188xxxxxx.
    private static readonly nuint GameHeapStart = unchecked((nuint)0x4100000000UL);

    // Upper bound for the scan. Without one the walk continued into 0x7FF... DLL space and
    // scored the CLR's own resource strings ("Cannot create an abstract class."), which
    // wasted the region budget and armed writes against runtime memory that cannot hold
    // dialogue and must not be modified.
    private static readonly nuint GameHeapEnd = unchecked((nuint)0x4400000000UL);

    // Pointers harvested from the per-message D0 array — swept before the linear scan,
    // since they point at objects the game is using for the message on screen right now.
    private readonly System.Collections.Generic.List<nuint> _sweepSeeds = new();
    private readonly object _seedLock = new();

    // Most recent LLM response. Written on the async LLM thread; read volatilely on the
    // game thread inside OnGameMemcpy so no lock is needed (reference read/write is
    // atomic on x64, and we only need eventual visibility, not strict ordering).
    private volatile string? _lastLlmText;

    /// <summary>
    /// The scene's original scripted lines, read out of the pool before we overwrite it.
    ///
    /// This is the context the model was missing: it knew who and where, but never what
    /// the conversation was actually about, so Ryuji answered the room rather than the
    /// scene. The lines we destroy are exactly the lines needed to respond to.
    ///
    /// Capture is once-only, at arm time. After the first write every slot holds our own
    /// generated text, so a later read would feed the model its own words back as if they
    /// were the script.
    /// </summary>
    private readonly System.Collections.Generic.List<string> _sceneScript = new();

    // Large-copy log: OnGameMemcpy records the dst of every heap-to-heap copy
    // ≥ 500 bytes. TryFindBfBuffer probes these at session start to locate the
    // BF script buffer, which is loaded in one bulk copy before session detection.
    private readonly System.Collections.Generic.List<nuint> _largeCopyDsts = new();
    private readonly object _largeCopyLock = new();

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
        _logger.WriteLine("[P5RGenSocialLinks] Post-health — entering setup.");

        try
        {
            nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
            _logger.WriteLine($"[P5RGenSocialLinks] Base: 0x{moduleBase:X}");

            _reader = new SocialLinkReader(moduleBase, _cfg.VerboseChain,
                msg => _logger!.WriteLine(msg));
            _logger.WriteLine("[P5RGenSocialLinks] SocialLinkReader OK.");

            _bridge = new DialogueBridge(_llmClient!, new LoggerAdapter(_logger!), _cfg);
            _logger.WriteLine("[P5RGenSocialLinks] DialogueBridge OK.");

            loader.GetController<IReloadedHooks>()?.TryGetTarget(out _hooks);
            _logger.WriteLine($"[P5RGenSocialLinks] IReloadedHooks: {(_hooks is not null ? "OK" : "null")}");

            TryActivateHook();
            TryResolveMsgInterpreter();
            SetupMemcpyHook();
            // BfDispatch hook crashes regardless of handler — abandoned, hunting text ptr via CE instead
            StartPollLoop();

            _modLog.Always($"[P5RGenSocialLinks] Started — hook:{(_hookActive ? "ON" : "OFF")} poll:ON");
            _modLog.Always($"[P5RGenSocialLinks] Log mirrored to {ModLogger.LogPath}");
        }
        catch (Exception ex)
        {
            _logger.WriteLine($"[P5RGenSocialLinks] STARTUP CRASH: {ex.GetType().Name}: {ex.Message}");
            _logger.WriteLine(ex.StackTrace ?? "(no stack trace)");
        }
    }

    private bool _hookActive;


    // Diagnostic: bare pass-through — if this crashes, the hook address/convention is wrong.
    // If this is stable, the crash is in our logging/memory-read code above.
    private static int _bfCallCount;

    // Windows x64 user-mode addresses are capped at 128 TB (bit 47).
    // Values above this are data misread as pointers — reject them.
    private static readonly nuint UserAddrMax = unchecked((nuint)0x7FFFFFFFFFFFUL);


    // ── Poll loop — session lifecycle + text pool discovery ───────────────

    private readonly StructDiffScanner _diffScanner = new();
    private uint   _lastBfPc;        // change-detection: fires only when PC moves (32-bit per mov [rbx+20],eax)
    private ushort _msgIdCandidate;  // current candidate from the last BF instruction window
    private int    _msgIdStreak;     // consecutive windows with the same candidate
    private ushort _currentMsgId;    // last confirmed msg_id (3+ consecutive windows)
    private nuint  _capturedSession; // session address when _currentMsgId was last confirmed
    private nuint  _confirmedBfBase; // real BF script base (session+0x18, memory-mapped <4 GB)
    private nuint  _bmdBase;         // BMD text table found by TryScanForBmd(); 0 until found
    private bool   _bmdScanDone;     // true after the one-shot scan fires this session
    private nuint  _currentMsgTextAddr; // session+0xD0 snapshot at msgId confirmation — direct write target


    // Reusable scan buffer — the heap sweep reads hundreds of megabytes per pass and
    // must not allocate a fresh array per chunk.
    private const int ScanChunk = 0x40000; // 256 KB
    private readonly byte[] _scanBuf = new byte[ScanChunk];


    private const int MaxJoinGap = 32;


    // ── BMD scanner ───────────────────────────────────────────────────────


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
            ReportWatchedRecord();

            if (!_reader!.TryResolve(out nuint session))
            {
                if (lastSession != 0)
                {
                    _diffScanner.Reset();
                    _bridge!.ResetSession();
                    _lastBfPc        = 0;
                    _bfBufferBase    = 0;
                    _bfBufferOff     = 0;
                    _currentMsgId    = 0;
                    _msgIdCandidate  = 0;
                    _msgIdStreak     = 0;
                    _capturedSession = 0;
                    _confirmedBfBase = 0;
                    _bmdBase             = 0;
                    _bmdScanDone         = false;
                    _currentMsgTextAddr  = 0;
                    _bmdTextPool         = 0;
                    _bmdScanDoneV2       = false;
                    _bmdScanAttempts     = 0;
                    _poolSlots           = null;
                    _bmdPoolLen          = 0x1000;
                    _poolWriteDone       = false;
                    _poolIsHeap          = false;
                    _heapPools.Clear();
                    _poolRecords.Clear();
                    _poolNextRecord.Clear();
                    LogSceneCoverage();
                    _plan.Clear();
                    // Cleared with the session: the next hang-out is a different scene,
                    // and a stale script would describe a conversation that already ended.
                    _sceneScript.Clear();
                    _utf16CpyLogged      = 0;
                    _heapSweepCursor     = 0;
                    _utf16SweepHits      = 0;
                    _heapSweepDone       = false;
                    _lastLlmText         = null;
                    lock (_largeCopyLock) _largeCopyDsts.Clear();
                    _sessionActive = false;
                    _modLog!.Info("[P5RGenSocialLinks] Hang-out ended — session cleared.");
                }
                lastSession = 0;
                continue;
            }

            if (session != lastSession)
            {
                lastSession = session;
                _diffScanner.Reset();

                SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(session);
                if (snap is null) continue;

                _modLog!.Info(
                    $"[P5RGenSocialLinks] Hang-out: Confidant={snap.ConfidantId} Rank={snap.RankLevel} Scene={snap.SceneNumber} (0x{session:X})");
                _sessionActive = true;

                // Fallback dispatch if hook isn't active.
                if (!_hookActive)
                    _bridge!.DispatchAsync(snap, ContextBuilder.Build(snap),
                                          lineIndex: 0,
                                          maxChars: NextRecordCapacity());

                continue;
            }

            // Keep the queue ahead of the player, and land anything it has finished.
            //
            // Flush first: text generated on an earlier tick should reach memory before a
            // new request is issued, so a busy server delays the next line rather than the
            // one already paid for.
            SocialLinkSnapshot? live = SocialLinkReader.TryReadFromPtr(session);
            if (live is not null)
            {
                FlushReadyRecords();
                PumpPregen(live);
            }

            // BF buffer discovery: live scan of session struct every tick.
            TryFindBfBuffer(session);

            // Always maintain the diff snapshot — TryFindBfBufferFromSnapshot reads it
            // to catch the BF script pointer even after it has been cleared from live
            // memory (it exists for only one poll interval at session+0x60).
            string? diff = _diffScanner.Diff(session);
            if (_cfg.StructDiffEnabled && diff is not null)
                _modLog!.Info($"[P5RGenSocialLinks] {diff}");

            // Snapshot-based discovery: catches transient ptrs that TryFindBfBuffer
            // missed because the game set+cleared them between our two live reads.
            TryFindBfBufferFromSnapshot(session);

            // BF line probe: read current dialogue text from bfBase+pc on every tick.
            ProbeBfLine(session);

            // Lazy textAddr re-probe: session+0xD0 may not be populated at BF-PC detection
            // time (cold first-message path). Retry every tick until the C++ dialogue
            // system catches up; the LLM latency (3-30s) gives us plenty of cycles.
            if (_currentMsgId != 0 && _currentMsgTextAddr == 0 && _capturedSession != 0)
            {
                nuint recovered = TryReadTextAddr(_capturedSession);
                if (recovered != 0)
                {
                    _currentMsgTextAddr = recovered;
                    _modLog!.Info($"[MSG] TextAddr recovered=0x{recovered:X} (msgId=0x{_currentMsgId:X})");
                }
            }

            // Primary target: the heap dialogue pool. The mapped-file scans below reach
            // only the global item/skill tables — the live conversation is on the heap,
            // as ASCII, confirmed at 0x41DD7F6389 / 0x42102CAAA9.
            if (_cfg.HeapScanEnabled && _currentMsgId != 0 && _heapPools.Count == 0)
                TryFindHeapDialoguePool();

            // Diagnostic sweeps, off by default now that the pool location is known.
            if (_cfg.StructDiffEnabled)
            {
                if (_confirmedBfBase != 0 && !_bmdScanDoneV2) TryScanBmdVicinity();
                if (_currentMsgId != 0)                       SweepHeapForUtf16();
            }

            // One-shot BMD scan per session — fires once after first msgId confirmation.
            if (_confirmedBfBase != 0 && _bmdBase == 0 && !_bmdScanDone)
            {
                _bmdScanDone = true;
                TryScanForBmd();
            }
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    public void Suspend()
    {
        _cts.Cancel();
        _conversationHook?.Disable();
        _memcpyHook?.Disable();
        _bfDispatchHook?.Disable();
        _diffScanner.Reset();
        _logger?.WriteLine("[P5RGenSocialLinks] Suspended.");
    }

    public void Resume()
    {
        _conversationHook?.Enable();
        _memcpyHook?.Enable();
        _bfDispatchHook?.Enable();
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
        _bfDispatchHook?.Disable();
        _msgWatch?.Dispose();
        _msgWatch = null;
        _diffScanner.Reset();
        _logger?.WriteLine("[P5RGenSocialLinks] Unloaded.");
    }

    public bool CanUnload()  => true;
    public bool CanSuspend() => true;
    public Action Disposing  => () => { };
}
