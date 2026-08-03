"""Canned mock responses for MOCK_LLM=1 mode — one per confidant."""

# Maps confidant_id to a short in-character line for E2E testing without a real model.
# Lines are written to match each character's established voice.
MOCK_LINES: dict[int, str] = {
    1:  "The future is unwritten, young trickster. Show me what you are capable of.",
    2:  "You can't just slack off, Joker! Our reputation as phantom thieves is at stake.",
    3:  "We need to analyse the situation carefully before we act.",
    4:  "I hope we can make wonderful memories together today.",
    5:  "Beauty is found in the earnest pursuit of one's art. Let us channel that today.",
    6:  "Stop dawdling. Coffee's getting cold.",
    7:  "I know things feel hard right now, but we can get through it together, okay?",
    8:  "Dude, you're seriously the only one who gets me. Let's crush this!",
    9:  "How interesting. I didn't expect you to choose that.",
    10: "Analysis complete — optimal course of action: spend time here. Efficiency: high.",
    11: "The stars are aligned favorably for you today. Take courage.",
    12: "You're not half bad, kid. Don't get cocky.",
    13: "You're still here? I suppose my bedside manner is improving.",
    14: "Oh, it's fine. I was just... resting my eyes between lessons.",
    15: "This story is going to be big. I can feel it.",
    16: "You think you can beat me? Show me what you've got.",
    17: "Every sacrifice on the board teaches something. What will you learn today?",
    18: "The Phantom Thieves are amazing! I'm updating the fan site right now.",
    19: "An honest politician may seem like a fantasy. I intend to prove otherwise.",
    20: "I will find the truth, no matter what it takes.",
    22: "Gymnastics is about trusting your body completely. Maybe life is too.",
    23: "It's remarkable how much people can endure when they have someone to talk to.",
}

_DEFAULT = "This is a canned mock response for testing without a real model."


def get_mock_response(confidant_id: int, rank: int) -> str:
    line = MOCK_LINES.get(confidant_id, _DEFAULT)
    return f"[MOCK rank {rank}] {line}"
