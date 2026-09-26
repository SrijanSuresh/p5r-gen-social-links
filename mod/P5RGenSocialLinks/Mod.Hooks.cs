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

/// <summary>
/// The injected-code surface: every hook this mod installs, and the callback it
/// installs it for.
///
/// Gathered into one part because this is the only code that runs on P5R's threads rather
/// than the mod's. The interpreter stub is six instructions inside a loop that executes
/// once per character of every message in the game (Ch. 68), and the memcpy and BF
/// dispatch hooks are survivors of the offset-discovery era. A fault in any of them takes
/// the game down, not the mod, so the blast radius is worth being able to read in one
/// place.
///
/// See learning.md Ch. 82 for why the class is split at all.
/// </summary>
public partial class Mod
{


    /// <summary>
    /// Locate the message interpreter's byte-fetch instruction and record its address.
    ///
    /// This is the first half of replacing the heap heuristic. Everything the pool code
    /// does today - scan tens of megabytes, score regions for English, arm the top one
    /// plus its content-identical twin - exists only because we could not ask the game
    /// which bytes it was about to render. This instruction can be asked: the struct in
    /// RBX holds the record pointer and the cursor, so hooking it turns a guess into a
    /// read (learning.md Ch. 65-66).
    ///
    /// Resolution is separated from hooking on purpose. A signature that resolves proves
    /// the pattern survived this build; a hook that misbehaves is a different failure,
    /// and diagnosing them together in a live game means restarting P5R for each guess.
    /// </summary>
    private void TryResolveMsgInterpreter()
    {
        try
        {
            using var scanner = new FunctionScanner();
            nuint? addr = scanner.TryFindFirst(Signatures.MsgByteFetch);
            if (addr is null)
            {
                _modLog!.Always(
                    "[P5RGenSocialLinks] MsgByteFetch sig NOT FOUND — p5r.exe build differs " +
                    "from the one the pattern was taken from. Pool heuristic stays in charge.");
                return;
            }

            _msgByteFetch = addr.Value + Signatures.MsgByteFetchToMovzx;
            nuint rva     = addr.Value - scanner.ModuleBase;

            // Expected P5R.exe+17A3D1F. Logging the offset rather than the absolute
            // address is what makes this comparable to a disassembler across runs.
            _modLog!.Always(
                $"[P5RGenSocialLinks] MsgByteFetch sig OK: P5R.exe+{rva:X} " +
                $"(abs 0x{addr.Value:X}, movzx at 0x{_msgByteFetch:X})");

            if (!_cfg.MsgHookEnabled)
            {
                _modLog!.Always("[P5RGenSocialLinks] Msg watch disabled by config.");
                return;
            }
            if (_hooks is null)
            {
                _modLog!.Always("[P5RGenSocialLinks] Msg watch skipped — IReloadedHooks null.");
                return;
            }

            try
            {
                _msgWatch = new MsgInterpreterWatch(_hooks, _msgByteFetch);
                _msgWatch.StartSampling();
                _modLog!.Always("[P5RGenSocialLinks] Msg watch ACTIVE (sampling at 5ms).");
            }
            catch (Exception ex)
            {
                // Deliberately broad, and only around this one call. Assembling the stub
                // runs through a third-party assembler whose failure type is not part of
                // the interface we reference, and every path after this point still works
                // without the watch — so an unknown assembler error must degrade to the
                // pool heuristic rather than abort mod startup.
                _modLog!.Always($"[P5RGenSocialLinks] Msg watch FAILED to install: {ex.Message}");
                _msgWatch = null;
            }
        }
        catch (InvalidOperationException ex)
        {
            _modLog!.Always($"[P5RGenSocialLinks] MsgByteFetch scan FAILED: {ex.Message}");
        }
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

            bool dispatched = _bridge!.DispatchAsync(snap, ContextBuilder.Build(snap),
                                                       lineIndex: fireCount,
                                                       maxChars: NextRecordCapacity());
            if (!dispatched)
                _modLog!.Info($"[P5RGenSocialLinks] CmmExec #{fireCount}: throttled.");
        }
        catch (Exception ex)
        {
            _logger?.WriteLine($"[P5RGenSocialLinks] OnCmmExecEvent error: {ex.Message}");
        }

        return result;
    }
}
