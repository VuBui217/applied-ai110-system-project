"""
Evaluation harness for DocuBot.

Compares retrieval quality across all sample queries and prints a
structured report with per-query confidence scores and a final summary.

Metrics reported:
- Hit rate:         fraction of queries where at least one retrieved
                    snippet came from an expected source file
- Avg confidence:   mean TF-IDF cosine similarity across all returned
                    snippets for a query (0.0 = no match, 1.0 = perfect)
- Pass / Fail:      PASS when the query is a hit, FAIL otherwise

Run with:
    python evaluation.py
"""

from dataset import SAMPLE_QUERIES


# -----------------------------------------------------------
# Expected document signals
# -----------------------------------------------------------
# Maps a query substring to the filename(s) that should appear
# in the retrieval results for that query.
# -----------------------------------------------------------

EXPECTED_SOURCES = {
    "auth token":          ["AUTH.md"],
    "environment variables": ["AUTH.md"],
    "database":            ["DATABASE.md"],
    "users":               ["API_REFERENCE.md"],
    "projects":            ["API_REFERENCE.md"],
    "refresh":             ["AUTH.md"],
    "users table":         ["DATABASE.md"],
}


def expected_files_for_query(query):
    """Returns expected filenames based on simple substring matching."""
    query_lower = query.lower()
    matches = []
    for key, files in EXPECTED_SOURCES.items():
        if key in query_lower:
            matches.extend(files)
    return matches


# -----------------------------------------------------------
# Core evaluation function
# -----------------------------------------------------------

def evaluate_retrieval(bot, top_k=3):
    """
    Runs DocuBot retrieval against every query in SAMPLE_QUERIES.

    Returns (hit_rate, detailed_results) where detailed_results is a
    list of dicts, each containing:
        query           — the original query string
        expected        — list of expected source filenames
        retrieved       — list of (filename, snippet, confidence) triples
        retrieved_files — just the filenames, for quick comparison
        avg_confidence  — mean confidence across retrieved snippets
        hit             — True if at least one retrieved file was expected
        verdict         — "PASS" or "FAIL"
    """
    results = []
    hits = 0

    for query in SAMPLE_QUERIES:
        expected = expected_files_for_query(query)
        retrieved = bot.retrieve(query, top_k=top_k)   # list of (fname, snippet, conf)

        retrieved_files = [fname for fname, _, _ in retrieved]
        avg_conf = bot.average_confidence(retrieved)

        hit = any(f in retrieved_files for f in expected) if expected else False
        if hit:
            hits += 1

        results.append({
            "query":            query,
            "expected":         expected,
            "retrieved":        retrieved,
            "retrieved_files":  retrieved_files,
            "avg_confidence":   avg_conf,
            "hit":              hit,
            "verdict":          "PASS" if hit else "FAIL",
        })

    hit_rate = hits / len(SAMPLE_QUERIES) if SAMPLE_QUERIES else 0.0
    return hit_rate, results


# -----------------------------------------------------------
# Pretty-printing
# -----------------------------------------------------------

def print_eval_results(hit_rate, results):
    """Prints a structured per-query report followed by a summary block."""

    total   = len(results)
    passed  = sum(1 for r in results if r["hit"])
    failed  = total - passed
    overall_avg_conf = (
        sum(r["avg_confidence"] for r in results) / total if total else 0.0
    )

    print("\nDocuBot Retrieval Evaluation")
    print("=" * 60)

    for item in results:
        verdict_label = f"[{item['verdict']}]"
        print(f"\n{verdict_label} {item['query']}")
        print(f"  Expected files : {item['expected'] or '(none defined)'}")
        print(f"  Retrieved files: {item['retrieved_files'] or '(no results)'}")
        print(f"  Avg confidence : {item['avg_confidence']:.3f}")

        # Show per-snippet confidence detail
        if item["retrieved"]:
            for fname, _, conf in item["retrieved"]:
                bar = "#" * int(conf * 20)
                print(f"    {fname:<25} conf={conf:.3f}  |{bar}")

    # Summary block
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Queries evaluated : {total}")
    print(f"  Passed (hits)     : {passed}")
    print(f"  Failed (misses)   : {failed}")
    print(f"  Hit rate          : {hit_rate:.0%}")
    print(f"  Avg confidence    : {overall_avg_conf:.3f}")
    print("=" * 60)


# -----------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    from docubot import DocuBot

    print("Running DocuBot retrieval evaluation...\n")
    bot = DocuBot()

    hit_rate, results = evaluate_retrieval(bot)
    print_eval_results(hit_rate, results)
