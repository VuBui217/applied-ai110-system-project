"""
Integration tests for DocuBot's RAG and agentic modes.

All Gemini API calls are mocked — no API key or network access required.

Covers:
- answer_rag: snippet passing, fallback, error propagation
- answer_agentic: confidence-gated retry loop, rephrase call, best-result selection
- GeminiClient._call_model: empty response guard, exception handler
- GeminiClient.rephrase_query: fallback to original on failure
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docubot import DocuBot, AGENTIC_CONFIDENCE_THRESHOLD, AGENTIC_MAX_RETRIES
from dataset import load_fallback_documents
from llm_client import GeminiClient, EMPTY_RESPONSE_FALLBACK


# -----------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------

@pytest.fixture
def mock_llm():
    """A MagicMock that stands in for a GeminiClient instance."""
    client = MagicMock(spec=GeminiClient)
    client.answer_from_snippets.return_value = "Mocked LLM answer."
    client.rephrase_query.return_value = "What is the process for token generation?"
    return client


@pytest.fixture
def bot(mock_llm):
    """DocuBot loaded with fallback docs and the mock LLM client."""
    instance = DocuBot.__new__(DocuBot)
    instance.llm_client = mock_llm
    instance.docs_folder = "docs"
    instance.documents = load_fallback_documents()
    instance.index = instance.build_index(instance.documents)
    instance.paragraphs = instance._extract_all_paragraphs()
    instance.vectorizer, instance.paragraph_vectors = instance._fit_tfidf()
    return instance


@pytest.fixture
def bot_no_llm():
    """DocuBot with no LLM client — tests RuntimeError guards."""
    instance = DocuBot.__new__(DocuBot)
    instance.llm_client = None
    instance.docs_folder = "docs"
    instance.documents = load_fallback_documents()
    instance.index = instance.build_index(instance.documents)
    instance.paragraphs = instance._extract_all_paragraphs()
    instance.vectorizer, instance.paragraph_vectors = instance._fit_tfidf()
    return instance


# -----------------------------------------------------------
# answer_rag tests
# -----------------------------------------------------------

class TestAnswerRag:

    def test_calls_llm_with_snippet_pairs(self, bot, mock_llm):
        """answer_rag must pass (filename, text) pairs — not triples — to the LLM."""
        bot.answer_rag("Where is the auth token generated?")
        assert mock_llm.answer_from_snippets.called
        _, kwargs_or_args = mock_llm.answer_from_snippets.call_args_list[0], None
        args = mock_llm.answer_from_snippets.call_args[0]
        snippets_passed = args[1]
        # Every element must be a 2-tuple, not a 3-tuple
        for item in snippets_passed:
            assert len(item) == 2, (
                f"Expected (filename, text) pair, got tuple of length {len(item)}"
            )

    def test_passes_original_query_to_llm(self, bot, mock_llm):
        """The query passed to the LLM must match what the user asked."""
        query = "Which endpoint lists all users?"
        bot.answer_rag(query)
        args = mock_llm.answer_from_snippets.call_args[0]
        assert args[0] == query

    def test_returns_llm_answer(self, bot, mock_llm):
        """answer_rag should return whatever the LLM client returns."""
        mock_llm.answer_from_snippets.return_value = "The answer is 42."
        result = bot.answer_rag("Where is the auth token generated?")
        assert result == "The answer is 42."

    def test_returns_fallback_when_no_snippets(self, bot, mock_llm):
        """When retrieve returns nothing, answer_rag must not call the LLM."""
        with patch.object(bot, "retrieve", return_value=[]):
            result = bot.answer_rag("xyzzy quux frobnicator blorp")
        mock_llm.answer_from_snippets.assert_not_called()
        assert result == "I do not know based on these docs."

    def test_raises_when_no_llm_client(self, bot_no_llm):
        """answer_rag must raise RuntimeError if no LLM client is set."""
        with pytest.raises(RuntimeError, match="RAG mode requires an LLM client"):
            bot_no_llm.answer_rag("auth token")

    def test_snippet_filenames_are_strings(self, bot, mock_llm):
        """All filenames passed to the LLM must be strings."""
        bot.answer_rag("Where is the auth token generated?")
        args = mock_llm.answer_from_snippets.call_args[0]
        for filename, _ in args[1]:
            assert isinstance(filename, str)


# -----------------------------------------------------------
# answer_agentic tests
# -----------------------------------------------------------

class TestAnswerAgentic:

    def test_answers_directly_when_confidence_is_high(self, bot, mock_llm):
        """If the first retrieval is confident enough, rephrase must not be called."""
        high_conf_snippets = [
            ("AUTH.md", "token text", 0.5),
            ("AUTH.md", "more auth text", 0.4),
        ]
        with patch.object(bot, "retrieve", return_value=high_conf_snippets):
            bot.answer_agentic("Where is the auth token generated?")
        mock_llm.rephrase_query.assert_not_called()
        mock_llm.answer_from_snippets.assert_called_once()

    def test_rephrases_when_confidence_is_low(self, bot, mock_llm):
        """If avg confidence is below threshold, rephrase_query must be called."""
        low_conf_snippets = [("AUTH.md", "some text", 0.01)]
        high_conf_snippets = [("AUTH.md", "auth token text", 0.5)]

        with patch.object(
            bot, "retrieve",
            side_effect=[low_conf_snippets, high_conf_snippets]
        ):
            bot.answer_agentic("auth token")

        mock_llm.rephrase_query.assert_called_once()

    def test_uses_rephrased_query_for_second_retrieval(self, bot, mock_llm):
        """After rephrasing, the second retrieve call must use the new query."""
        low_conf = [("AUTH.md", "text", 0.01)]
        high_conf = [("AUTH.md", "auth token text", 0.5)]
        rephrased = "How is the access token created?"
        mock_llm.rephrase_query.return_value = rephrased

        retrieve_calls = []
        def fake_retrieve(query, top_k=3):
            retrieve_calls.append(query)
            return low_conf if len(retrieve_calls) == 1 else high_conf

        with patch.object(bot, "retrieve", side_effect=fake_retrieve):
            bot.answer_agentic("auth token")

        assert retrieve_calls[1] == rephrased

    def test_uses_best_snippets_across_attempts(self, bot, mock_llm):
        """The LLM should receive the highest-confidence snippets found."""
        low_conf  = [("AUTH.md", "weak result", 0.01)]
        high_conf = [("AUTH.md", "strong result", 0.8)]

        with patch.object(
            bot, "retrieve",
            side_effect=[low_conf, high_conf]
        ):
            bot.answer_agentic("auth token")

        args = mock_llm.answer_from_snippets.call_args[0]
        snippets_passed = args[1]
        texts = [text for _, text in snippets_passed]
        assert "strong result" in texts

    def test_returns_fallback_when_all_attempts_fail(self, bot, mock_llm):
        """If every attempt returns empty snippets, return the fallback string."""
        with patch.object(bot, "retrieve", return_value=[]):
            result = bot.answer_agentic("xyzzy quux frobnicator")
        assert result == "I do not know based on these docs."
        mock_llm.answer_from_snippets.assert_not_called()

    def test_max_retries_not_exceeded(self, bot, mock_llm):
        """rephrase_query must be called at most AGENTIC_MAX_RETRIES times."""
        low_conf = [("AUTH.md", "text", 0.01)]
        with patch.object(bot, "retrieve", return_value=low_conf):
            bot.answer_agentic("auth token")
        assert mock_llm.rephrase_query.call_count <= AGENTIC_MAX_RETRIES

    def test_raises_when_no_llm_client(self, bot_no_llm):
        """answer_agentic must raise RuntimeError if no LLM client is set."""
        with pytest.raises(RuntimeError, match="Agentic mode requires an LLM client"):
            bot_no_llm.answer_agentic("auth token")


# -----------------------------------------------------------
# GeminiClient._call_model guardrail tests
# -----------------------------------------------------------

class TestCallModelGuardrails:

    def _make_client(self):
        """Creates a GeminiClient with a mocked underlying model."""
        with patch("llm_client.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
                client = GeminiClient()
        return client

    def test_returns_empty_response_fallback_on_empty_text(self):
        """An empty string from the model should return EMPTY_RESPONSE_FALLBACK."""
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.text = ""
        client.client.models.generate_content.return_value = mock_response

        result = client._call_model("some prompt")
        assert result == EMPTY_RESPONSE_FALLBACK

    def test_returns_safe_string_on_api_exception(self):
        """Any API exception must be caught and return a readable error string."""
        client = self._make_client()
        client.client.models.generate_content.side_effect = RuntimeError("API timeout")

        result = client._call_model("some prompt")
        assert "error occurred" in result.lower()
        assert "API timeout" in result

    def test_returns_model_text_on_success(self):
        """A normal response should be returned stripped."""
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.text = "  Here is the answer.  "
        client.client.models.generate_content.return_value = mock_response

        result = client._call_model("some prompt")
        assert result == "Here is the answer."


# -----------------------------------------------------------
# GeminiClient.rephrase_query tests
# -----------------------------------------------------------

class TestRephraseQuery:

    def _make_client(self):
        with patch("llm_client.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
                client = GeminiClient()
        return client

    def test_returns_rephrased_text_on_success(self):
        """A successful rephrase returns the model's output."""
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.text = "How is the access token generated?"
        client.client.models.generate_content.return_value = mock_response

        result = client.rephrase_query("auth token?")
        assert result == "How is the access token generated?"

    def test_falls_back_to_original_on_api_error(self):
        """An API error during rephrase must return the original query."""
        client = self._make_client()
        client.client.models.generate_content.side_effect = RuntimeError("timeout")

        original = "auth token?"
        result = client.rephrase_query(original)
        assert result == original

    def test_falls_back_to_original_on_empty_response(self):
        """An empty model response during rephrase must return the original query."""
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.text = ""
        client.client.models.generate_content.return_value = mock_response

        original = "auth token?"
        result = client.rephrase_query(original)
        assert result == original
