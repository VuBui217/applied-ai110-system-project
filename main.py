"""
CLI runner for DocuBot.

Supports four modes:
1. Naive LLM  — Gemini answers from the full docs corpus (no retrieval)
2. Retrieval  — TF-IDF search returns snippets with confidence scores (no LLM)
3. RAG        — TF-IDF retrieval feeds grounded snippets to Gemini
4. Agent      — multi-step query refinement loop: retrieves, checks confidence,
                rephrases with Gemini if needed, then answers
"""

from dotenv import load_dotenv
load_dotenv()

from docubot import DocuBot
from llm_client import GeminiClient
from dataset import SAMPLE_QUERIES


def try_create_llm_client():
    """
    Tries to create a GeminiClient.
    Returns (llm_client, has_llm: bool).
    """
    try:
        client = GeminiClient()
        return client, True
    except RuntimeError as exc:
        print("Warning: LLM features are disabled.")
        print(f"Reason: {exc}")
        print("You can still run retrieval only mode.\n")
        return None, False


def choose_mode(has_llm):
    """
    Asks the user which mode to run.
    Returns "1", "2", "3", "4", or "q".
    """
    unavailable = " (unavailable, no GEMINI_API_KEY)"
    print("Choose a mode:")
    print(f"  1) Naive LLM over full docs (no retrieval){'' if has_llm else unavailable}")
    print("  2) Retrieval only — TF-IDF with confidence scores (no LLM)")
    print(f"  3) RAG — retrieval + LLM{'' if has_llm else unavailable}")
    print(f"  4) Agent — multi-step query refinement + LLM{'' if has_llm else unavailable}")
    print("  q) Quit")

    choice = input("Enter choice: ").strip().lower()
    return choice


def get_query_or_use_samples():
    """
    Ask the user if they want to run all sample queries or a single custom query.

    Returns:
        queries: list of strings
        label: short description of the source of queries
    """
    print("\nPress Enter to run built in sample queries.")
    custom = input("Or type a single custom query: ").strip()

    if custom:
        return [custom], "custom query"
    else:
        return SAMPLE_QUERIES, "sample queries"


def run_naive_llm_mode(bot, has_llm):
    """
    Mode 1:
    Naive LLM generation over the full docs corpus.
    """
    if not has_llm or bot.llm_client is None:
        print("\nNaive LLM mode is not available (no GEMINI_API_KEY).\n")
        return

    queries, label = get_query_or_use_samples()
    print(f"\nRunning naive LLM mode on {label}...\n")

    all_text = bot.full_corpus_text()

    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        answer = bot.llm_client.naive_answer_over_full_docs(query, all_text)
        print("Answer:")
        print(answer)
        print()


def run_retrieval_only_mode(bot):
    """
    Mode 2:
    Retrieval only answers. No LLM involved.
    """
    queries, label = get_query_or_use_samples()
    print(f"\nRunning retrieval only mode on {label}...\n")

    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        answer = bot.answer_retrieval_only(query)
        print("Retrieved snippets:")
        print(answer)
        print()


def run_rag_mode(bot, has_llm):
    """
    Mode 3:
    Retrieval plus LLM generation.
    """
    if not has_llm or bot.llm_client is None:
        print("\nRAG mode is not available (no GEMINI_API_KEY).\n")
        return

    queries, label = get_query_or_use_samples()
    print(f"\nRunning RAG mode on {label}...\n")

    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        answer = bot.answer_rag(query)
        print("Answer:")
        print(answer)
        print()


def run_agentic_mode(bot, has_llm):
    """
    Mode 4: agentic multi-step query refinement.

    For each query the agent:
      1. Retrieves snippets with the original query.
      2. Checks average confidence.
      3. If confidence is low, asks Gemini to rephrase and retries.
      4. Answers using the best snippets found across all attempts.

    Intermediate steps are printed so the reasoning chain is visible.
    """
    if not has_llm or bot.llm_client is None:
        print("\nAgent mode is not available (no GEMINI_API_KEY).\n")
        return

    queries, label = get_query_or_use_samples()
    print(f"\nRunning agent mode on {label}...\n")

    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        answer = bot.answer_agentic(query)
        print("Answer:")
        print(answer)
        print()


def main():
    print("DocuBot Tinker Activity")
    print("=======================\n")

    llm_client, has_llm = try_create_llm_client()
    bot = DocuBot(llm_client=llm_client)

    while True:
        choice = choose_mode(has_llm)

        if choice == "q":
            print("\nGoodbye.")
            break
        elif choice == "1":
            run_naive_llm_mode(bot, has_llm)
        elif choice == "2":
            run_retrieval_only_mode(bot)
        elif choice == "3":
            run_rag_mode(bot, has_llm)
        elif choice == "4":
            run_agentic_mode(bot, has_llm)
        else:
            print("\nUnknown choice. Please pick 1, 2, 3, 4, or q.\n")


if __name__ == "__main__":
    main()
