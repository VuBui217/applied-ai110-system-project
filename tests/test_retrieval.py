"""
Unit tests for DocuBot's retrieval pipeline.

All tests use the FALLBACK_DOCS corpus from dataset.py so no API key,
no docs/ folder, and no network access is required.

Covers:
- build_index: inverted index construction
- score_document: TF-IDF cosine similarity scoring
- retrieve: end-to-end retrieval with confidence filtering
- average_confidence: helper used by agentic mode
- answer_retrieval_only: formatted output including confidence labels
"""

import sys
import os
import pytest

# Ensure the project root is on the path when running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docubot import DocuBot, MIN_CONFIDENCE
from dataset import load_fallback_documents


# -----------------------------------------------------------
# Shared fixture
# -----------------------------------------------------------

@pytest.fixture
def bot():
    """
    Returns a DocuBot instance loaded with FALLBACK_DOCS.
    No LLM client, no filesystem access beyond the fixture itself.
    """
    instance = DocuBot.__new__(DocuBot)
    instance.llm_client = None
    instance.docs_folder = "docs"
    instance.documents = load_fallback_documents()
    instance.index = instance.build_index(instance.documents)
    instance.paragraphs = instance._extract_all_paragraphs()
    instance.vectorizer, instance.paragraph_vectors = instance._fit_tfidf()
    return instance


# -----------------------------------------------------------
# build_index tests
# -----------------------------------------------------------

class TestBuildIndex:

    def test_contains_expected_tokens(self, bot):
        """Common domain words should appear in the index."""
        assert "token" in bot.index
        assert "database" in bot.index
        assert "authentication" in bot.index

    def test_token_maps_to_auth_file(self, bot):
        """The word 'token' should point to AUTH.md."""
        assert "AUTH.md" in bot.index.get("token", [])

    def test_database_maps_to_database_file(self, bot):
        """The word 'database' should point to DATABASE.md."""
        assert "DATABASE.md" in bot.index.get("database", [])

    def test_no_duplicate_filenames_per_word(self, bot):
        """Each word's file list should contain no duplicates."""
        for word, filenames in bot.index.items():
            assert len(filenames) == len(set(filenames)), (
                f"Duplicate filenames for word '{word}': {filenames}"
            )

    def test_all_values_are_lists(self, bot):
        """Every value in the index should be a list."""
        for word, filenames in bot.index.items():
            assert isinstance(filenames, list)


# -----------------------------------------------------------
# score_document tests
# -----------------------------------------------------------

class TestScoreDocument:

    def test_relevant_text_scores_higher_than_irrelevant(self, bot):
        """A snippet matching the query should score above an unrelated one.
        Both strings use words present in the fitted vocabulary so that
        TF-IDF overlap is deterministic."""
        query = "token login authentication"
        # From AUTH.md fallback — contains 'token' and 'login'
        relevant = (
            "Clients authenticate by sending a POST request to /api/login. "
            "They receive a token which must be included in the Authorization header."
        )
        # From DATABASE.md fallback — no token / login / auth overlap
        irrelevant = "The projects table contains project_id, name, description, status, owner_id."
        assert bot.score_document(query, relevant) > bot.score_document(query, irrelevant)

    def test_score_is_float_between_0_and_1(self, bot):
        """TF-IDF cosine similarity must always be in [0.0, 1.0]."""
        score = bot.score_document("how do I authenticate?", "Tokens are created by auth_utils.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_identical_text_scores_near_1(self, bot):
        """Scoring a query against itself should return a high score."""
        text = "generate access token authentication secret key"
        score = bot.score_document(text, text)
        assert score > 0.8

    def test_empty_text_scores_zero(self, bot):
        """An empty snippet should score zero against any query."""
        score = bot.score_document("auth token", "")
        assert score == 0.0

    def test_completely_unrelated_text_scores_low(self, bot):
        """Random unrelated text should score very low."""
        score = bot.score_document("authentication token", "apple banana orange fruit salad")
        assert score < 0.1


# -----------------------------------------------------------
# retrieve tests
# -----------------------------------------------------------

class TestRetrieve:

    def test_auth_query_returns_auth_file(self, bot):
        """Auth-related query should surface AUTH.md."""
        results = bot.retrieve("Where is the auth token generated?")
        filenames = [fname for fname, _, _ in results]
        assert "AUTH.md" in filenames

    def test_database_query_returns_database_file(self, bot):
        """Database query should surface DATABASE.md."""
        results = bot.retrieve("How do I connect to the database?")
        filenames = [fname for fname, _, _ in results]
        assert "DATABASE.md" in filenames

    def test_users_endpoint_query_returns_api_file(self, bot):
        """API endpoint query should surface API_REFERENCE.md."""
        results = bot.retrieve("Which endpoint lists all users?")
        filenames = [fname for fname, _, _ in results]
        assert "API_REFERENCE.md" in filenames

    def test_nonsense_query_returns_empty(self, bot):
        """A query with no matching terms should return no results."""
        results = bot.retrieve("xyzzy quux frobnicator blorp")
        assert results == []

    def test_returns_at_most_top_k(self, bot):
        """Results must never exceed top_k."""
        results = bot.retrieve("auth token database users", top_k=2)
        assert len(results) <= 2

    def test_default_top_k_is_3(self, bot):
        """Default call should return at most 3 results."""
        results = bot.retrieve("auth token database users api endpoint")
        assert len(results) <= 3

    def test_results_sorted_by_confidence_descending(self, bot):
        """Results should be ordered highest confidence first."""
        results = bot.retrieve("auth token generated secret key")
        confidences = [conf for _, _, conf in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_all_confidence_scores_above_min_threshold(self, bot):
        """Every returned snippet must meet MIN_CONFIDENCE."""
        results = bot.retrieve("authentication token secret key login")
        for _, _, confidence in results:
            assert confidence >= MIN_CONFIDENCE, (
                f"Snippet confidence {confidence:.3f} is below MIN_CONFIDENCE {MIN_CONFIDENCE}"
            )

    def test_each_result_is_a_triple(self, bot):
        """Every result must be a (filename, snippet, confidence) triple."""
        results = bot.retrieve("auth token")
        for item in results:
            assert len(item) == 3
            filename, snippet, confidence = item
            assert isinstance(filename, str)
            assert isinstance(snippet, str)
            assert isinstance(confidence, float)

    def test_snippet_text_is_non_empty(self, bot):
        """No result should contain an empty snippet."""
        results = bot.retrieve("authentication access token refresh")
        for _, snippet, _ in results:
            assert snippet.strip() != ""


# -----------------------------------------------------------
# average_confidence tests
# -----------------------------------------------------------

class TestAverageConfidence:

    def test_returns_zero_for_empty_list(self, bot):
        assert bot.average_confidence([]) == 0.0

    def test_returns_correct_average(self, bot):
        snippets = [
            ("AUTH.md", "text a", 0.4),
            ("API_REFERENCE.md", "text b", 0.6),
        ]
        assert abs(bot.average_confidence(snippets) - 0.5) < 1e-9

    def test_single_item_returns_its_confidence(self, bot):
        snippets = [("AUTH.md", "some text", 0.75)]
        assert bot.average_confidence(snippets) == 0.75


# -----------------------------------------------------------
# answer_retrieval_only tests
# -----------------------------------------------------------

class TestAnswerRetrievalOnly:

    def test_returns_fallback_when_no_snippets(self, bot):
        """Should return the 'I do not know' string for unmatched queries."""
        result = bot.answer_retrieval_only("xyzzy quux frobnicator blorp")
        assert result == "I do not know based on these docs."

    def test_output_includes_filename(self, bot):
        """Retrieved output should name the source file."""
        result = bot.answer_retrieval_only("Where is the auth token generated?")
        assert "AUTH.md" in result

    def test_output_includes_confidence_label(self, bot):
        """Retrieved output should include a confidence score label."""
        result = bot.answer_retrieval_only("Where is the auth token generated?")
        assert "confidence:" in result
