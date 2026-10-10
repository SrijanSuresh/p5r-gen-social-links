"""
The page the player types into.

A single self-contained string rather than a template directory or a static mount: it is
one form, it has no assets, and the server should not grow a templating dependency to
serve it. Inline styles and script for the same reason — the page is fetched from
127.0.0.1 by the person running the server, so there is nothing to cache-bust and nothing
to bundle.

Kept out of main.py so the route reads as a route. See learning.md Ch. 87.
"""

from __future__ import annotations

SAY_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Say something</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0b0b10;
    --panel: #15151e;
    --edge: #2a2a38;
    --ink: #f2f2f7;
    --dim: #9b9bb0;
    --accent: #e4002b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    display: grid;
    place-items: center;
    padding: 1.5rem;
    background: var(--bg);
    color: var(--ink);
    font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { width: 100%; max-width: 34rem; }
  h1 {
    margin: 0 0 .25rem;
    font-size: 1.5rem;
    letter-spacing: -.02em;
  }
  p.sub { margin: 0 0 1.25rem; color: var(--dim); font-size: .9rem; }
  form {
    background: var(--panel);
    border: 1px solid var(--edge);
    border-radius: .75rem;
    padding: 1rem;
  }
  textarea {
    width: 100%;
    min-height: 5.5rem;
    resize: vertical;
    padding: .7rem .8rem;
    border-radius: .5rem;
    border: 1px solid var(--edge);
    background: #0f0f16;
    color: var(--ink);
    font: inherit;
  }
  textarea:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .75rem;
    margin-top: .75rem;
  }
  button {
    padding: .55rem 1.1rem;
    border: 0;
    border-radius: .5rem;
    background: var(--accent);
    color: #fff;
    font: inherit;
    font-weight: 600;
    cursor: pointer;
  }
  button:disabled { opacity: .5; cursor: default; }
  .count { color: var(--dim); font-size: .8rem; font-variant-numeric: tabular-nums; }
  .count.over { color: var(--accent); }
  #status { margin-top: .9rem; min-height: 1.4rem; font-size: .9rem; }
  #status.ok { color: #6ee7a0; }
  #status.err { color: var(--accent); }
  kbd {
    padding: .1rem .35rem;
    border: 1px solid var(--edge);
    border-bottom-width: 2px;
    border-radius: .25rem;
    background: #0f0f16;
    font: inherit;
    font-size: .8em;
  }
</style>
</head>
<body>
<main>
  <h1>Say something</h1>
  <p class="sub">
    Type a line and the confidant will answer it. Keep this tab open next to the game.
  </p>

  <form id="f">
    <label for="t" class="sub">Your line, as Joker</label>
    <textarea id="t" maxlength="280" autofocus
              placeholder="So... how have you been, really?"></textarea>
    <div class="row">
      <span class="count" id="c">0 / 280</span>
      <button id="b" type="submit">Send</button>
    </div>
  </form>

  <div id="status" role="status" aria-live="polite"></div>
  <p class="sub">
    <kbd>Ctrl</kbd> + <kbd>Enter</kbd> sends. A line you do not send within two minutes
    expires, and sending again replaces a line the game has not picked up yet.
  </p>
</main>

<script>
  const form = document.getElementById('f');
  const box = document.getElementById('t');
  const count = document.getElementById('c');
  const button = document.getElementById('b');
  const status = document.getElementById('status');

  function render() {
    const n = box.value.length;
    count.textContent = n + ' / 280';
    count.classList.toggle('over', n >= 280);
    button.disabled = box.value.trim().length === 0;
  }

  function show(message, kind) {
    status.textContent = message;
    status.className = kind;
  }

  box.addEventListener('input', render);

  // Ctrl+Enter rather than plain Enter: the box is a textarea so a stray newline should
  // not fire a request the player did not mean to send.
  box.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = box.value.trim();
    if (!text) return;

    button.disabled = true;
    show('Sending...', '');
    try {
      const response = await fetch('/say', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        // FastAPI reports a validation failure in `detail`, which may itself be a list.
        const why = Array.isArray(body.detail)
          ? body.detail.map((d) => d.msg).join('; ')
          : body.detail;
        show(why || ('Rejected (' + response.status + ')'), 'err');
        return;
      }
      show('Sent: "' + body.text + '"', 'ok');
      box.value = '';
    } catch (err) {
      // The server is the only thing this page talks to, so a network error means it is
      // not running — which is the single most likely thing to have gone wrong.
      show('Could not reach the server. Is start.bat still running?', 'err');
    } finally {
      render();
      box.focus();
    }
  });

  render();
</script>
</body>
</html>
"""
