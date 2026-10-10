using System;
using System.Threading.Tasks;
using P5RGenSocialLinks.Server;

internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        // The server's own default port. Override it for a mock instance on another port.
        string baseUrl = args.Length > 0 && args[0].StartsWith("http")
            ? args[0]
            : "http://127.0.0.1:8765";
        using var client = new LLMClient(baseUrl);
        int failures = 0;

        void Check(string label, bool ok, string detail)
        {
            Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {label}: {detail}");
            if (!ok) failures++;
        }

        Console.WriteLine($"Driving the real LLMClient against {baseUrl}");

        await client.ClearPlayerLineAsync();
        Check("empty store reports nothing waiting",
              await client.HasPlayerLineAsync() == false, "HasPlayerLineAsync() == false");

        Check("nothing to collect from an empty store",
              await client.TakePlayerLineAsync() is null, "TakePlayerLineAsync() == null");

        using var http = new System.Net.Http.HttpClient();
        using var say = new System.Net.Http.StringContent(
            "{\"text\":\"  So... how have you  been, really?  \"}",
            System.Text.Encoding.UTF8, "application/json");
        var posted = await http.PostAsync($"{baseUrl}/say", say);
        Check("the page can submit a line", posted.IsSuccessStatusCode,
              $"POST /say -> {(int)posted.StatusCode}");

        Check("the mod sees it waiting",
              await client.HasPlayerLineAsync(), "HasPlayerLineAsync() == true");

        Check("peeking did not consume it",
              await client.HasPlayerLineAsync(), "still waiting after a peek");

        string? taken = await client.TakePlayerLineAsync();
        Check("collecting returns the normalised line",
              taken == "So... how have you been, really?", $"got {taken ?? "null"}");

        Check("the line is gone after one collection",
              await client.TakePlayerLineAsync() is null, "second take == null");

        Check("and nothing is waiting any more",
              await client.HasPlayerLineAsync() == false, "HasPlayerLineAsync() == false");

        using var say2 = new System.Net.Http.StringContent(
            "{\"text\":\"never mind\"}", System.Text.Encoding.UTF8, "application/json");
        await http.PostAsync($"{baseUrl}/say", say2);
        await client.ClearPlayerLineAsync();
        Check("clearing discards a pending line",
              await client.HasPlayerLineAsync() == false, "cleared");

        // A request carrying a player line must round-trip through the real endpoint.
        string text = await client.GenerateAsync(new GenerateRequest
        {
            ConfidantId = 8, Rank = 4, Context = "gym",
            CharacterName = "Ryuji Sakamoto", MaxChars = 60,
            Addressee = "Joker", PlayerLine = "So... how have you been, really?",
        });
        Check("generate accepts a player_line", text.Length > 0, $"\"{text}\"");

        using var malformed = new LLMClient("127.0.0.1:8799");   // no scheme
        Check("a server_url with no scheme reports nothing rather than throwing",
              await malformed.HasPlayerLineAsync() == false, "no exception");
        await malformed.ClearPlayerLineAsync();
        Check("and clearing against it is a no-op", true, "no exception");

        using var dead = new LLMClient("http://127.0.0.1:1");
        Check("an unreachable server reports nothing waiting rather than throwing",
              await dead.HasPlayerLineAsync() == false, "no exception");
        Check("an unreachable server collects nothing rather than throwing",
              await dead.TakePlayerLineAsync() is null, "no exception");
        await dead.ClearPlayerLineAsync();
        Check("clearing against an unreachable server is a no-op", true, "no exception");

        Console.WriteLine(failures == 0 ? "\nALL PASS" : $"\n{failures} FAILED");
        return failures == 0 ? 0 : 1;
    }
}
