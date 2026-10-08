using System.Text.Json;
using P5RGenSocialLinks.Server;
using Xunit;

namespace P5RGenSocialLinks.Tests;

/// <summary>
/// The JSON the mod and the server have to agree on.
///
/// Two processes, two languages, one contract, and nothing compiles across the boundary.
/// The literals below are copied from what the server actually answers — the same strings
/// `server/tests/test_say_endpoint.py` asserts on — so a field renamed on either side
/// fails here instead of becoming a feature that silently stops working.
/// </summary>
public class WireContractTests
{
    // Mirrors LLMClient's own options. Pydantic expects snake_case, so the policy has to
    // apply in both directions: it names fields on the way out and matches them on the
    // way in.
    private static readonly JsonSerializerOptions Opts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    // --- GET /player-line ---------------------------------------------------------

    [Fact]
    public void A_pending_line_deserialises()
    {
        const string body =
            """{"pending":true,"text":"So... how have you been, really?","age_seconds":0.09}""";

        var line = JsonSerializer.Deserialize<PlayerLineResponse>(body, Opts);

        Assert.NotNull(line);
        Assert.True(line!.Pending);
        Assert.Equal("So... how have you been, really?", line.Text);
        Assert.Equal(0.09, line.AgeSeconds, precision: 4);
    }

    [Fact]
    public void An_empty_store_deserialises()
    {
        const string body = """{"pending":false,"text":"","age_seconds":0.0}""";

        var line = JsonSerializer.Deserialize<PlayerLineResponse>(body, Opts);

        Assert.NotNull(line);
        Assert.False(line!.Pending);
        Assert.Equal(string.Empty, line.Text);
    }

    [Fact]
    public void A_response_missing_fields_does_not_throw()
    {
        // An older server, or a 404 body that happens to be JSON. The client treats a
        // non-pending answer as "nothing typed", so the defaults have to be safe.
        var line = JsonSerializer.Deserialize<PlayerLineResponse>("{}", Opts);

        Assert.NotNull(line);
        Assert.False(line!.Pending);
        Assert.Equal(string.Empty, line.Text);
    }

    // --- POST /generate -----------------------------------------------------------

    [Fact]
    public void A_generate_request_serialises_to_the_names_pydantic_expects()
    {
        var req = new GenerateRequest
        {
            ConfidantId   = 14,
            Rank          = 5,
            Context       = "clinic",
            CharacterName = "Tae Takemi",
            MaxChars      = 60,
            Addressee     = "Joker",
            PlayerLine    = "is she going to be okay?",
        };

        string json = JsonSerializer.Serialize(req, Opts);

        Assert.Contains("\"confidant_id\":14", json);
        Assert.Contains("\"character_name\":\"Tae Takemi\"", json);
        Assert.Contains("\"max_chars\":60", json);
        Assert.Contains("\"addressee\":\"Joker\"", json);
        Assert.Contains("\"player_line\":\"is she going to be okay?\"", json);
    }

    [Fact]
    public void An_ordinary_request_sends_an_empty_player_line()
    {
        // Every request except the one answering a typed line. Empty rather than absent
        // is fine — the server's field defaults to "" and omits the claim either way.
        var req = new GenerateRequest
        {
            ConfidantId = 8, Rank = 4, Context = "gym", CharacterName = "Ryuji Sakamoto",
        };

        string json = JsonSerializer.Serialize(req, Opts);

        Assert.Contains("\"player_line\":\"\"", json);
        Assert.Contains("\"addressee\":\"\"", json);
    }

    [Fact]
    public void A_generated_response_deserialises()
    {
        const string body =
            """{"text":"Not quite famous enough, anyway.","session_id":3}""";

        var result = JsonSerializer.Deserialize<GenerateResponse>(body, Opts);

        Assert.NotNull(result);
        Assert.Equal("Not quite famous enough, anyway.", result!.Text);
    }
}
