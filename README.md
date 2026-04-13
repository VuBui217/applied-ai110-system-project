# DocuBot — AI Documentation Assistant

A documentation assistant that answers developer questions by searching
project docs and generating grounded answers with Google Gemini.

---

## Original Project

This project extends **DocuBot**, originally built during CodePath AI110 Modules 1–3.
The original DocuBot was a CLI tool that answered developer questions about a codebase
using three modes: naive LLM generation over raw docs, keyword-based retrieval, and a
basic RAG pipeline that passed retrieved snippets to Gemini. It demonstrated the core
trade-off between fluent but ungrounded LLM answers and accurate but unformatted retrieval
results, using a small set of markdown documentation files as its knowledge base.

---

## Title and Summary

**DocuBot** is an AI-powered documentation assistant that helps developers quickly find
accurate answers about a codebase without leaving their terminal.

Instead of asking a general-purpose LLM that may hallucinate project-specific details,
DocuBot searches your own documentation first — then uses Gemini only to format and
explain what it found. The upgraded system adds TF-IDF retrieval with confidence scoring,
a multi-step agentic query refinement loop, structured logging, and a full pytest test
harness — making it suitable as a reference implementation for production-grade RAG systems.

**Why it matters:** Developers waste significant time hunting through documentation.
A grounded AI assistant that cites its sources and refuses to guess is more trustworthy
than one that always produces a fluent answer.

---

## Architecture Overview

```
                        ┌──────────────┐
                        │     User     │
                        └──────┬───────┘
                               │ query
                    ┌──────────▼──────────┐
                    │       main.py       │
                    │   (mode selector)   │
                    └──┬───┬────┬─────┬───┘
                       │   │    │     │
              Mode 1   │   │    │     │  Mode 4
           ┌───────────┘   │    │     └──────────────────┐
           ▼          Mode 2    │ Mode 3                  ▼
   ┌───────────────┐   ┌────────▼──────────┐   ┌──────────────────────┐
   │  Full corpus  │   │   TF-IDF Index    │   │   Agentic Loop       │
   │  → Gemini     │   │   (docubot.py)    │   │   retrieve → check   │
   └───────────────┘   │                   │   │   confidence → if    │
                       │ 1. Vectorize docs │   │   low: rephrase with │
                       │ 2. Score query    │   │   Gemini → retry     │
                       │ 3. Filter by      │   └──────────┬───────────┘
                       │    MIN_CONFIDENCE │              │
                       │ 4. Return top-k   │              │ best snippets
                       │    (fname,text,   │◄─────────────┘
                       │     confidence)   │
                       └────────┬──────────┘
                                │ snippets
                  ┌─────────────▼─────────────┐
                  │       llm_client.py        │
                  │  answer_from_snippets()    │
                  │  - grounded prompt         │
                  │  - cite sources rule       │
                  │  - "I do not know" rule    │
                  │  - logs to docubot.log     │
                  │  - catches API errors      │
                  └─────────────┬──────────────┘
                                │ answer
                        ┌───────▼────────┐
                        │     User       │
                        └────────────────┘

   Testing layer (runs independently, no API key needed):
   ┌──────────────────────────────────────────────────────┐
   │  pytest tests/          evaluation.py                │
   │  - test_retrieval.py    - hit rate per query         │
   │  - test_rag.py          - avg confidence per query   │
   │    (mocked LLM)         - PASS / FAIL summary        │
   └──────────────────────────────────────────────────────┘
```

> See `assets/architecture.png` for the exported diagram.

The system has four layers:
- **User interface** (`main.py`) — mode selection and query input
- **Retrieval engine** (`docubot.py`) — TF-IDF indexing, scoring, and confidence filtering
- **LLM layer** (`llm_client.py`) — grounded prompting, logging, and error handling
- **Reliability layer** (`tests/`, `evaluation.py`) — pytest suite and hit-rate evaluation

---

## Setup Instructions

### 1. Clone the repository and enter the project folder

```bash
git clone <your-repo-url>
cd applied-ai110-system-project
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API key

```bash
cp .env.example .env
```

Edit `.env` and add your key:

```
GEMINI_API_KEY=your_api_key_here
```

> Modes 2 (Retrieval only) and the test suite work without a key.
> Modes 1, 3, and 4 require a valid Gemini API key.

### 5. Run DocuBot

```bash
python main.py
```

### 6. Run the test suite (no API key needed)

```bash
python -m pytest tests/ -v
```

### 7. Run the retrieval evaluation

```bash
python evaluation.py
```

---

## Sample Interactions

### Mode 2 — Retrieval Only

```
Question: Where is the auth token generated?

Retrieved snippets:
[AUTH.md] (confidence: 0.16)
# Authentication Guide
Tokens are created by the generate_access_token function inside auth_utils.py.
They are signed using the AUTH_SECRET_KEY environment variable.
...
```

**Why this is useful:** The developer sees the exact source file and section,
with a confidence score showing how closely it matched their query.
No hallucination is possible because no LLM is involved.

---

### Mode 3 — RAG

```
Question: How does a client refresh an access token?

Answer:
Based on AUTH.md, a client can refresh an access token by sending a POST
request to /api/refresh. The file does not describe additional parameters
required for this request beyond having a valid existing token.
```

**Why this is useful:** Gemini synthesizes a readable answer but is
constrained to only use the retrieved snippet — it cites the source file
and refuses to invent details not present in the docs.

---

### Mode 4 — Agent (query refinement)

```
Question: How do I set up credentials?

  [Agent attempt 1] Query: "How do I set up credentials?"
  [Agent attempt 1] Avg confidence: 0.08
  [Agent] Low confidence — asking Gemini to rephrase query...
  [Agent] Rephrased query: "What environment variables are required for authentication?"
  [Agent attempt 2] Query: "What environment variables are required for authentication?"
  [Agent attempt 2] Avg confidence: 0.38
  [Agent] Confidence threshold met. Proceeding to answer.

Answer:
Based on AUTH.md and SETUP.md, you need to set AUTH_SECRET_KEY before
running the application. DATABASE_URL is also required for database access.
```

**Why this is useful:** The original query used the word "credentials" which
does not appear in the docs. The agent automatically rephrased it to use
documentation vocabulary ("environment variables", "authentication") and
retrieved a high-confidence answer on the second attempt.

---

## Design Decisions

### TF-IDF over keyword counting

The original DocuBot counted raw word matches. This was fast but treated
every word equally — common words like `name` or `is` scored as high as
rare, specific terms like `generate_access_token`. TF-IDF weights each
word by how rare it is across the corpus, so domain-specific terms
carry more signal. The trade-off is a startup cost (fitting the vectorizer)
and a dependency on `scikit-learn`, but retrieval precision improved
measurably — the "what is my name?" false-positive that plagued the
original system now returns zero results.

### Section-level chunking over full-document scoring

Splitting docs on markdown headings means each retrieved snippet includes
its section header and stays self-contained. The trade-off is that very
long sections can still return large chunks. Sentence-level chunking would
be more precise but would lose the heading context that tells the user
which feature or endpoint a snippet belongs to.

### Confidence threshold as a guardrail

`MIN_CONFIDENCE = 0.10` filters out snippets with weak TF-IDF overlap
before they reach the LLM. This prevents the model from generating a
confident-sounding answer based on a tangential match. The trade-off is
that tightly phrased queries that use different vocabulary from the docs
may return no results — but a honest "I do not know" is safer than a
fabricated answer.

### Agentic loop over a single retrieval pass

When confidence is low, asking Gemini to rephrase the query in
"documentation vocabulary" often closes the gap between how a developer
phrases a question and how the docs were written. The loop is capped at
`AGENTIC_MAX_RETRIES = 2` to prevent runaway API calls. The trade-off
is latency — a refinement pass adds one extra Gemini call before answering.

### Mocked integration tests over live API tests

The test suite uses `unittest.mock` to replace the Gemini client entirely.
This means tests run in under 2 seconds, cost nothing, and work without a
key. The trade-off is that prompt quality and actual Gemini behavior are
not tested — those require manual evaluation with a real API key.

---

## Testing Summary

### Automated test suite

```
$ python -m pytest tests/ -v
45 passed in 1.6s
```

| File | Tests | Coverage |
|---|---|---|
| `tests/test_retrieval.py` | 26 | Index construction, TF-IDF scoring, retrieval ranking, confidence thresholds, output format |
| `tests/test_rag.py` | 19 | Snippet passing, fallback strings, API error handling, agentic retry logic, rephrase fallback |

**What worked well:**
- TF-IDF reliably filters out the noise queries that word-count scoring
  let through (e.g. "what is my name?" now correctly returns nothing).
- The agentic rephrase loop measurably improves results for queries that
  use non-documentation vocabulary.
- Error handling in `_call_model` ensures the app never crashes on a bad
  API response — it logs the error and returns a readable fallback string.

**What didn't work as expected:**
- `"Which fields are stored in the users table?"` scores below
  `MIN_CONFIDENCE` because `users` and `table` are high-frequency words
  across all docs, giving them low IDF weight. Hit rate: **6/8 (75%)**.
- TF-IDF does not stem words, so `token` and `tokens` are treated as
  different vocabulary terms. Queries must use the exact form that appears
  in the docs.

### Retrieval evaluation

```
$ python evaluation.py

Queries evaluated : 8
Passed (hits)     : 6
Failed (misses)   : 2
Hit rate          : 75%
Avg confidence    : 0.158
```

---

## Reflection and Ethics

### What this project taught about AI and problem-solving

Building DocuBot made the gap between "AI that sounds right" and "AI that
is right" concrete and measurable. Mode 1 (naive LLM) consistently
produced fluent, confident answers that had nothing to do with the actual
project configuration. The evaluation harness made this visible with numbers
rather than anecdotes — hit rate and confidence scores turned a vague
sense of "this feels wrong" into a quantified comparison.

The agentic loop was the most surprising result: the single biggest source
of retrieval failure was vocabulary mismatch between how developers ask
questions and how documentation is written. Having the model rephrase the
query before retrying closed that gap without any changes to the underlying
retrieval system. This suggests that query translation is often a higher
return-on-investment improvement than tuning the retrieval algorithm itself.

### Limitations and biases

- **Vocabulary sensitivity:** TF-IDF requires the query to use the same
  words as the docs. Synonyms, abbreviations, or paraphrases score zero.
- **Doc coverage bias:** The system can only be as good as the docs in
  `docs/`. Outdated or missing documentation produces outdated or missing
  answers with no warning to the user.
- **English-only stop words:** The vectorizer filters English stop words,
  so non-English documentation would need a different configuration.
- **Small corpus limitation:** IDF weights are computed over a tiny 4-file
  corpus. In a larger real-world codebase, term weights would be more
  meaningful and retrieval quality would improve.

### Potential misuse and prevention

- **Presenting AI output as ground truth:** A developer could copy a RAG
  answer into a PR without verifying it against the source file. Mitigation:
  always display the source filename alongside the answer so readers know
  where to verify.
- **Stale docs producing stale answers:** If the `docs/` folder is not
  kept up to date, DocuBot will confidently answer from outdated
  information. Mitigation: add a doc freshness warning (e.g. check file
  modification dates and warn if any doc is older than 90 days).
- **Prompt injection via docs:** A malicious actor who can write to the
  `docs/` folder could inject instructions into the LLM prompt via a
  documentation file. Mitigation: treat `docs/` as untrusted input and
  sanitize or sandbox in production deployments.

### Surprises during reliability testing

The most surprising finding was that the "users table" query — which
intuitively should be an easy hit for `DATABASE.md` — returned zero
results. Both `users` and `table` are common words across all four docs,
so TF-IDF assigns them near-zero IDF weight. A query that felt obvious to
a human was completely invisible to the retrieval system. This is a
concrete example of why retrieval systems need evaluation harnesses: human
intuition about what "should" match is often wrong.

### Collaboration with AI during this project

**Helpful suggestion:** When the `google.generativeai` SDK raised a
deprecation warning, the AI assistant identified the correct replacement
package (`google-genai`) and updated both the import and the API call
pattern (`genai.Client` + `client.models.generate_content`) in one step,
saving significant time that would otherwise have been spent reading
migration documentation.

**Flawed suggestion:** In the initial test for `score_document`, the AI
generated a "relevant" text string using words like `generate_access_token`
and `tokens` (plural) — neither of which matched the query's vocabulary
in the fitted TF-IDF model, causing the test to fail with both scores at
0.0. The fix required understanding that sklearn's `TfidfVectorizer` does
not stem words by default and that test inputs must use vocabulary present
in the training corpus. The AI's suggestion looked correct on the surface
but ignored the stemming constraint of the underlying library.
