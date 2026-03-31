"""Tests for groq-prompt-scorer hook."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

HOOK_PATH = Path(__file__).parent / "groq-prompt-scorer.py"
spec = importlib.util.spec_from_file_location("groq_prompt_scorer", HOOK_PATH)
assert spec and spec.loader
groq_prompt_scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(groq_prompt_scorer)

MINIMUM_PROMPT_LENGTH = groq_prompt_scorer.MINIMUM_PROMPT_LENGTH
classify_prompt = groq_prompt_scorer.classify_prompt
format_guidance = groq_prompt_scorer.format_guidance
log_classification = groq_prompt_scorer.log_classification
should_bypass = groq_prompt_scorer.should_bypass


class TestShouldBypass:
    def test_empty_prompt_bypasses(self) -> None:
        assert should_bypass("") is True

    def test_short_prompt_bypasses(self) -> None:
        assert should_bypass("fix it") is True

    def test_prompt_at_minimum_length_does_not_bypass(self) -> None:
        prompt = "x" * MINIMUM_PROMPT_LENGTH
        assert should_bypass(prompt) is False

    def test_asterisk_prefix_bypasses(self) -> None:
        assert should_bypass("* do whatever you want with auth") is True

    def test_slash_prefix_bypasses(self) -> None:
        assert should_bypass("/commit this change now") is True

    def test_hash_prefix_bypasses(self) -> None:
        assert should_bypass("# this is a comment") is True

    def test_specific_prompt_does_not_bypass(self) -> None:
        assert should_bypass("fix the TypeError in src/auth.ts line 42") is False

    def test_whitespace_only_bypasses(self) -> None:
        assert should_bypass("        ") is True


class TestFormatGuidance:
    def test_formats_gaps_and_suggestion(self) -> None:
        result = {
            "verdict": "guide",
            "gaps": ["no file path specified", "no success criteria"],
            "suggestion": "Specify which file and what 'done' looks like.",
        }
        output = format_guidance(result)
        assert "PROMPT IMPROVEMENT SUGGESTION" in output
        assert "no file path specified" in output
        assert "no success criteria" in output
        assert "Specify which file" in output
        assert "Ask the user clarifying questions" in output

    def test_formats_without_gaps(self) -> None:
        result = {
            "verdict": "guide",
            "gaps": [],
            "suggestion": "Be more specific.",
        }
        output = format_guidance(result)
        assert "Missing:" not in output
        assert "Be more specific." in output

    def test_formats_without_suggestion(self) -> None:
        result = {
            "verdict": "guide",
            "gaps": ["too vague"],
            "suggestion": "",
        }
        output = format_guidance(result)
        assert "too vague" in output
        assert "Consider:" not in output


class TestClassifyPrompt:
    def _make_mock_groq(
        self, content: str | None, side_effect: Exception | None = None,
    ) -> MagicMock:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_client = MagicMock()
        if side_effect:
            mock_client.chat.completions.create.side_effect = side_effect
        else:
            mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_returns_parsed_json_from_groq(self) -> None:
        expected = {
            "verdict": "guide",
            "gaps": ["no scope"],
            "suggestion": "Add a file path.",
        }
        mock_client = self._make_mock_groq(json.dumps(expected))

        with patch("groq.Groq", return_value=mock_client):
            result = classify_prompt("add dark mode", "fake-key")

        assert result == expected
        mock_client.chat.completions.create.assert_called_once()

    def test_returns_none_on_empty_content(self) -> None:
        mock_client = self._make_mock_groq(None)

        with patch("groq.Groq", return_value=mock_client):
            result = classify_prompt("add dark mode", "fake-key")

        assert result is None

    def test_tries_primary_model_first(self) -> None:
        mock_client = self._make_mock_groq('{"verdict": "pass"}')

        with patch("groq.Groq", return_value=mock_client):
            classify_prompt("fix src/auth.ts line 42", "test-key")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == groq_prompt_scorer.GROQ_PRIMARY_MODEL
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["temperature"] == 0.1

    def test_falls_back_on_rate_limit(self) -> None:
        from groq import RateLimitError

        mock_response_ok = MagicMock()
        mock_response_ok.choices = [MagicMock()]
        mock_response_ok.choices[0].message.content = '{"verdict": "pass"}'

        mock_rate_limit_response = MagicMock()
        mock_rate_limit_response.status_code = 429
        mock_rate_limit_response.headers = {}
        mock_rate_limit_response.json.return_value = {
            "error": {"message": "rate limited"},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            RateLimitError(
                message="rate limited",
                response=mock_rate_limit_response,
                body={"error": {"message": "rate limited"}},
            ),
            mock_response_ok,
        ]

        with patch("groq.Groq", return_value=mock_client):
            result = classify_prompt("add dark mode", "fake-key")

        assert result == {"verdict": "pass"}
        calls = mock_client.chat.completions.create.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["model"] == groq_prompt_scorer.GROQ_PRIMARY_MODEL
        assert calls[1][1]["model"] == groq_prompt_scorer.GROQ_FALLBACK_MODEL

    def test_returns_none_when_both_models_fail(self) -> None:
        from groq import RateLimitError

        mock_rate_limit_response = MagicMock()
        mock_rate_limit_response.status_code = 429
        mock_rate_limit_response.headers = {}
        mock_rate_limit_response.json.return_value = {
            "error": {"message": "rate limited"},
        }

        rate_limit_error = RateLimitError(
            message="rate limited",
            response=mock_rate_limit_response,
            body={"error": {"message": "rate limited"}},
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            rate_limit_error,
            rate_limit_error,
        ]

        with patch("groq.Groq", return_value=mock_client):
            result = classify_prompt("add dark mode", "fake-key")

        assert result is None


class TestLogClassification:
    def test_writes_log_entry_with_verdict_and_prompt(self, tmp_path: Path) -> None:
        log_file = tmp_path / "groq-prompt-scorer.log"
        result = {"verdict": "guide", "gaps": ["no scope"], "suggestion": "Be specific."}

        log_classification("add dark mode", result, log_file)

        content = log_file.read_text()
        assert "guide" in content
        assert "add dark mode" in content
        assert "no scope" in content

    def test_appends_to_existing_log(self, tmp_path: Path) -> None:
        log_file = tmp_path / "groq-prompt-scorer.log"
        log_file.write_text("existing line\n")

        log_classification("test prompt here", {"verdict": "pass"}, log_file)

        content = log_file.read_text()
        assert "existing line" in content
        assert "pass" in content

    def test_includes_timestamp(self, tmp_path: Path) -> None:
        log_file = tmp_path / "groq-prompt-scorer.log"

        log_classification("test prompt here", {"verdict": "pass"}, log_file)

        content = log_file.read_text()
        assert "2026" in content or "202" in content

    def test_survives_unwritable_path(self) -> None:
        bad_path = Path("/nonexistent/dir/file.log")
        log_classification("test", {"verdict": "pass"}, bad_path)


class TestMainFlow:
    def test_exits_silently_without_api_key(self) -> None:
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            with pytest.raises(SystemExit) as exc_info:
                groq_prompt_scorer.main()
            assert exc_info.value.code == 0
