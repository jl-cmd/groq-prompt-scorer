# groq-prompt-scorer

A Claude Code hook that uses Groq's free LLM API to catch vague prompts before they waste tokens. Clear prompts pass through instantly. Underspecified prompts get flagged so Claude asks you clarifying questions first.

## How it works

```
You type a prompt
       |
  Hook fires (UserPromptSubmit)
       |
  Bypass? (slash commands, short prompts, * prefix)
       |-- yes --> pass through silently
       |-- no  --> send to Groq for classification
                       |
                  verdict: "pass"  --> silent pass-through
                  verdict: "guide" --> inject guidance context
                       |
                  Claude asks clarifying questions
                  You refine, Claude executes
```

The hook uses Groq's `llama-3.3-70b-versatile` as the primary classifier, falling back to `llama-3.1-8b-instant` on rate limits. Both are free tier.

**Fail-open:** If Groq is unreachable, the API key is missing, or anything errors, the hook exits silently and your prompt passes through unchanged. No disruption.

## Examples

| Your prompt | Verdict | What happens |
|---|---|---|
| `fix the TypeError in src/auth.ts line 42` | pass | Passes through, Claude acts immediately |
| `refactor getUserData to handle null returns` | pass | Specific target + action, passes through |
| `add dark mode` | guide | Claude asks: which components? toggle mechanism? color system? |
| `build me a full stack app` | guide | Claude asks: which stack? what features? starting state? |
| `it's broken fix everything` | guide | Claude asks: what error? which file? reproduction steps? |

## Requirements

- [Claude Code](https://claude.ai/claude-code) v2.0+
- Python 3.10+
- Free [Groq API key](https://console.groq.com/keys)

## Install

### 1. Install the groq package

```bash
pip install groq
```

### 2. Copy the hook

Copy `groq-prompt-scorer.py` to your Claude Code hooks directory:

```bash
# macOS / Linux
cp groq-prompt-scorer.py ~/.claude/hooks/session/

# Windows (PowerShell)
Copy-Item groq-prompt-scorer.py "$env:USERPROFILE\.claude\hooks\session\"
```

### 3. Set your API key

Add your Groq API key to Claude Code's local settings so hooks can access it. Edit (or create) `.claude/settings.local.json` in your project directory:

```json
{
  "env": {
    "GROQ_API_KEY": "gsk_your_key_here"
  }
}
```

Or add it to your user-level settings at `~/.claude/settings.local.json` to apply globally.

> `settings.local.json` is automatically gitignored by Claude Code. Your key stays out of version control.

### 4. Register the hook

Add the hook to `~/.claude/settings.json` under `hooks.UserPromptSubmit`. The exact command depends on your hook runner setup.

**If you have a hook runner** (like `run-hook-wrapper.js`):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "node -e \"process.argv.splice(1,0,'_');require(require('os').homedir()+'/.claude/hooks/run-hook-wrapper.js')\" \"session/groq-prompt-scorer.py\"",
            "timeout": 5000
          }
        ]
      }
    ]
  }
}
```

**Direct registration** (no hook runner):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude/hooks/session/groq-prompt-scorer.py || python ~/.claude/hooks/session/groq-prompt-scorer.py",
            "timeout": 5000
          }
        ]
      }
    ]
  }
}
```

### 5. Restart Claude Code

The hook activates on next session start.

## Bypass

- Prefix your prompt with `*` to skip classification: `* just do it`
- Slash commands (`/commit`, `/help`) bypass automatically
- Prompts under 15 characters bypass automatically

## Logging

Every classification is logged to `~/.claude/cache/groq-prompt-scorer.log`:

```
[2026-03-31 16:26:01] verdict=guide prompt="add search to the app" gaps=['file scope', 'search functionality details']
[2026-03-31 16:26:07] verdict=pass prompt="fix the TypeError in src/auth.ts line 42"
```

Watch live:

```bash
# macOS / Linux
tail -f ~/.claude/cache/groq-prompt-scorer.log

# Windows (PowerShell)
Get-Content ~/.claude/cache/groq-prompt-scorer.log -Wait
```

## Groq free tier limits

| Model | Requests/day | Tokens/day | Speed |
|---|---|---|---|
| llama-3.3-70b-versatile (primary) | 1,000 | 100,000 | 280 T/sec |
| llama-3.1-8b-instant (fallback) | 14,400 | 500,000 | 560 T/sec |

Typical classification uses ~500 tokens. The 70B handles ~200 prompts/day before the 8B takes over with 14,400 more.

## Tests

```bash
pip install pytest
pytest test_groq_prompt_scorer.py -v
```

21 tests covering bypass logic, classification, model fallback, logging, and error handling.

## How the classifier decides

The system prompt teaches the LLM to check for:

**Clear signals** (pass through):
- Specific target (file path, function name, error message)
- Defined action (fix, refactor, add, remove)
- Bounded scope (what to touch and what not to)

**Vague signals** (flag for guidance):
- Vague verbs without targets
- Multiple unrelated tasks in one prompt
- Feature requests without file scope
- Emotional descriptions instead of specifics
- Greenfield requests without starting state
- Open-ended permissions without stop conditions

The classifier is conservative: when in doubt, it passes through. False positives (blocking clear prompts) are worse than false negatives.

## Credits

Classification criteria adapted from [prompt-mini](https://github.com/nidhinjs/prompt-mini) by nidhinjs (MIT), specifically the 35 credit-killing anti-patterns taxonomy.

## License

MIT
