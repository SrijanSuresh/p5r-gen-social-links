using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Reloaded.Mod.Interfaces.Internal;
using P5RGenSocialLinks.Memory;
using P5RGenSocialLinks.Server;

namespace P5RGenSocialLinks;

public class Mod : IModV1
{
    private ILoggerV2?   _logger;
    private LLMClient?   _llmClient;
    private PeriodicTimer? _timer;
    private Task?          _pollTask;
    private CancellationTokenSource _cts = new();

    public void Start(IModLoaderV1 loader)
    {
        _logger = (ILoggerV2)loader.GetLogger();
        _logger.WriteLine("[P5RGenSocialLinks] Starting...");

        nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
        _logger.WriteLine($"[P5RGenSocialLinks] Module base: 0x{moduleBase:X}");

        var reader = new SocialLinkReader(moduleBase);
        _llmClient = new LLMClient();

        _timer    = new PeriodicTimer(TimeSpan.FromMilliseconds(250));
        _pollTask = Task.Run(() => PollLoopAsync(reader, _cts.Token));

        _logger.WriteLine("[P5RGenSocialLinks] Poll loop active.");
    }

    private async Task PollLoopAsync(SocialLinkReader reader, CancellationToken ct)
    {
        while (await _timer!.WaitForNextTickAsync(ct))
        {
            SocialLinkSnapshot? snap = reader.TryReadSnapshot();
            if (snap is null)
                continue;

            _logger!.WriteLine(
                $"[P5RGenSocialLinks] Confidant={snap.ConfidantId} Rank={snap.RankLevel} Line={snap.DialogueIndex}");

            // TODO Micro-step 4: invoke LLM and inject dialogue
        }
    }

    public void Suspend()  { _cts.Cancel(); }
    public void Resume()
    {
        _cts = new CancellationTokenSource();
        // reader state is rebuilt on next Start; log only
        _logger?.WriteLine("[P5RGenSocialLinks] Resumed.");
    }

    public void Unload()
    {
        _cts.Cancel();
        _pollTask?.Wait(TimeSpan.FromSeconds(2));
        _timer?.Dispose();
        _llmClient?.Dispose();
        _logger?.WriteLine("[P5RGenSocialLinks] Unloaded.");
    }

    public bool CanUnload()  => true;
    public bool CanSuspend() => true;
    public Action Disposing  => () => { };
}