using System.Collections.Generic;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Rolling buffer of LLM responses generated in the current hang-out session.
/// Used to build richer per-line context: "we already said X; now generate Y."
/// Cleared when the session pointer changes (new hang-out begins).
/// </summary>
internal sealed class SessionHistory
{
    private const int MaxEntries = 8;

    private readonly List<string> _entries = new(MaxEntries);
    private nuint _currentSession;

    internal void RecordResponse(nuint sessionPtr, string response)
    {
        if (sessionPtr != _currentSession)
        {
            _entries.Clear();
            _currentSession = sessionPtr;
        }
        if (_entries.Count >= MaxEntries)
            _entries.RemoveAt(0);
        _entries.Add(response);
    }

    /// <summary>
    /// Returns a compact "Prior dialogue: [line1] | [line2] | ..." string,
    /// or empty string if no prior lines this session.
    /// </summary>
    internal string BuildPriorContext(nuint sessionPtr)
    {
        if (sessionPtr != _currentSession || _entries.Count == 0)
            return string.Empty;

        return "Prior dialogue: " + string.Join(" | ", _entries);
    }

    internal int Count => _entries.Count;

    internal void Reset()
    {
        _entries.Clear();
        _currentSession = 0;
    }
}
