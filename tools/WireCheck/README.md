# WireCheck

Drives the mod's real `LLMClient` against a running inference server, to check the HTTP
contract the two processes share.

```powershell
# terminal 1
cd server
.\start-mock.bat          # no GPU or model needed

# terminal 2
dotnet run --project tools/WireCheck
```

It exercises the Phase 5 input channel end to end — peek, submit, peek again, collect,
collect twice, clear — plus a `/generate` carrying a `player_line`, and the three ways the
server can be absent:

| Case | Required behaviour |
|---|---|
| Server not running | "nothing typed", no exception |
| `server_url` with no scheme | "nothing typed", no exception |
| Store empty | "nothing typed", no exception |

Those last three matter more than they look. The mod's answer to every server failure is to
leave the scripted dialogue alone, and both the poll and the clear run on fire-and-forget
tasks — so an exception escaping either one is invisible: no log line, no crash, just a
feature that never works. That is exactly the bug the first run of this found.

Exits non-zero if any check fails, so it can gate a release by hand. CI does not run it:
it needs a live server on a port, and what it tests is the boundary rather than the logic
either side of it. The logic is covered by `mod/P5RGenSocialLinks.Tests` and `server/tests`.
