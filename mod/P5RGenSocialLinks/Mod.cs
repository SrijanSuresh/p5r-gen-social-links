using Reloaded.Mod.Interfaces;

namespace P5RGenSocialLinks;

public class Mod : IMod
{
    private IModLoader _modLoader = null!;
    private ILogger _logger = null!;

    public void Start(IModLoaderV1 loader)
    {
        _modLoader = (IModLoader)loader;
        _logger = (ILogger)_modLoader.GetLogger();
        _logger.WriteLine("[P5RGenSocialLinks] Mod loaded. Social Link AI is initializing...");
    }

    public void Suspend()  { }
    public void Resume()   { }
    public void Unload()   { _logger.WriteLine("[P5RGenSocialLinks] Unloaded."); }
    public bool CanUnload()  => true;
    public bool CanSuspend() => false;
    public Action Disposing => () => { };
}
