"""
Gemini client wrapper used by DocuBot.

Handles:
- Configuring the Gemini client from the GEMINI_API_KEY environment variable
- Naive "generation only" answers over the full docs corpus (Mode 1)
- RAG-style answers that use only retrieved snippets (Mode 3)
- Query rephrasing for agentic mode (Mode 4)
- Structured logging of every request and response to logs/docubot.log
- Safe error handling: Gemini failures are caught, logged, and return a
  safe fallback string instead of crashing the program
"""

import os
import logging

from google import genai

# -----------------------------------------------------------
# Logging setup
# -----------------------------------------------------------
# All LLM activity is appended to logs/docubot.log.
# The logs/ directory is created automatically if missing.
# -----------------------------------------------------------

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "docubot.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger("docubot")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)

    # File handler — full detail
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

# Central place to update the model name if needed.
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Fallback message returned whenever Gemini produces an empty response.
EMPTY_RESPONSE_FALLBACK = (
    "The model returned an empty response. "
    "Please try rephrasing your question."
)


class GeminiClient:
    """
    Wrapper around the Gemini generative model.

    All public methods:
    - Log the query and result to logs/docubot.log
    - Catch API exceptions and return a safe fallback string
    - Guard against empty Gemini responses
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY environment variable. "
                "Set it in your shell or .env file to enable LLM features."
            )

        self.client = genai.Client(api_key=api_key)
        logger.info("GeminiClient initialised with model: %s", GEMINI_MODEL_NAME)

    # -----------------------------------------------------------
    # Internal helper
    # -----------------------------------------------------------

    def _call_model(self, prompt, context_label="request"):
        """
        Calls model.generate_content with error handling.

        - Logs the prompt (truncated for readability).
        - Returns the response text on success.
        - Logs and returns EMPTY_RESPONSE_FALLBACK for empty responses.
        - Logs and returns a safe error string on any exception.
        """
        logger.debug("[%s] Prompt (first 300 chars): %.300s", context_label, prompt)
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL_NAME, contents=prompt
            )
            text = (response.text or "").strip()

            if not text:
                logger.warning("[%s] Gemini returned an empty response.", context_label)
                return EMPTY_RESPONSE_FALLBACK

            logger.debug("[%s] Response (first 300 chars): %.300s", context_label, text)
            return text

        except Exception as exc:
            logger.error("[%s] Gemini API error: %s", context_label, exc, exc_info=True)
            return (
                f"An error occurred while contacting the model: {exc}. "
                "Please check your API key and try again."
            )

    # -----------------------------------------------------------
    # Mode 1: naive generation over full docs
    # -----------------------------------------------------------

    def naive_answer_over_full_docs(self, query, all_text):
        """
        Sends only the question to Gemini with the full docs as context.
        Logs the query and answer.
        """
        logger.info("[naive] Query: %s", query)

        prompt = f"""You are a documentation assistant.
Answer this developer question using the documentation provided below.

Documentation:
{all_text}

Developer question:
{query}
"""
        answer = self._call_model(prompt, context_label="naive")
        logger.info("[naive] Answer (first 200 chars): %.200s", answer)
        return answer

    # -----------------------------------------------------------
    # Mode 3: RAG-style generation over retrieved snippets
    # -----------------------------------------------------------

    def answer_from_snippets(self, query, snippets):
        """
        Generates a grounded answer using only the retrieved snippets.

        snippets: list of (filename, text) tuples from DocuBot.retrieve()

        The prompt instructs Gemini to:
        - Use only the provided snippets
        - Refuse to guess when evidence is insufficient
        - Cite the filenames it relied on
        """
        logger.info("[rag] Query: %s", query)

        if not snippets:
            logger.warning("[rag] No snippets provided — returning fallback.")
            return "I do not know based on the docs I have."

        context_blocks = [f"File: {fname}\n{text}" for fname, text in snippets]
        context = "\n\n".join(context_blocks)

        logger.debug(
            "[rag] Snippets used: %s",
            [fname for fname, _ in snippets]
        )

        prompt = f"""You are a cautious documentation assistant helping developers understand a codebase.

You will receive:
- A developer question
- A small set of snippets from project files

Your job:
- Answer the question using only the information in the snippets.
- If the snippets do not provide enough evidence, refuse to guess.

Snippets:
{context}

Developer question:
{query}

Rules:
- Use only the information in the snippets. Do not invent new functions,
  endpoints, or configuration values.
- If the snippets are not enough to answer confidently, reply exactly:
  "I do not know based on the docs I have."
- When you do answer, briefly mention which files you relied on.
"""
        answer = self._call_model(prompt, context_label="rag")
        logger.info("[rag] Answer (first 200 chars): %.200s", answer)
        return answer

    # -----------------------------------------------------------
    # Mode 4: query rephrasing for agentic loop
    # -----------------------------------------------------------

    def rephrase_query(self, query):
        """
        Asks Gemini to rephrase a developer question using different
        wording that might match technical documentation better.

        Returns the rephrased query string.
        Falls back to the original query if the model fails.
        """
        logger.info("[agent] Rephrasing query: %s", query)

        prompt = f"""A developer asked the following question but the documentation search returned low-confidence results.

Original question: {query}

Rewrite this question using different technical wording that is more likely to match how documentation is written.
Return only the rephrased question — no explanation, no quotes, no punctuation changes beyond the question itself.
"""
        rephrased = self._call_model(prompt, context_label="rephrase")

        # If the model errored or returned the fallback, return the original
        if rephrased.startswith("An error occurred") or rephrased == EMPTY_RESPONSE_FALLBACK:
            logger.warning("[agent] Rephrase failed — keeping original query.")
            return query

        logger.info("[agent] Rephrased query: %s", rephrased)
        return rephrased
