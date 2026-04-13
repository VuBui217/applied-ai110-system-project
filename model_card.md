# DocuBot Model Card (v2)

This model card reflects the upgraded DocuBot system after the Module 4
extension. Changes from v1 are marked **[v2]** so the diff is easy to follow.

---

## 1. System Overview

**What is DocuBot trying to do?**

DocuBot answers developer questions about a project by searching documentation
files in the `docs/` folder. It finds the most relevant sections and either
returns them directly or passes them to Gemini to generate a clean answer.
The goal is to give accurate, doc-grounded responses instead of relying on
a model's general knowledge.

**What inputs does DocuBot take?**

- A natural language question typed by the user
- `.md` and `.txt` files inside the `docs/` folder
- A `GEMINI_API_KEY` environment variable (required for Modes 1, 3, and 4)

**What outputs does DocuBot produce?**

- **Mode 1:** A Gemini answer grounded in the full docs corpus
- **Mode 2:** The raw text of the most relevant doc snippets, with filenames
  and **[v2] confidence scores (0.0–1.0)**
- **Mode 3:** A Gemini-generated answer grounded in the retrieved snippets
- **[v2] Mode 4:** A Gemini answer produced after an agentic query refinement
  loop that retries with a rephrased query when initial confidence is low

---

## 2. Retrieval Design

**How does your retrieval system work?**

**[v2] Indexing — TF-IDF (replaces word-count)**

On startup, `_extract_all_paragraphs()` splits every document into sections
using markdown heading boundaries (`#`, `##`, `###`), keeping each heading
attached to its body. A `TfidfVectorizer` (unigrams + bigrams, English stop
words removed) is then fitted over all sections. This gives each term a
weight based on how often it appears in a section relative to how common it
is across all sections — rare, domain-specific terms score higher than
frequent generic ones.

**[v2] Scoring — cosine similarity (replaces word count)**

`score_document(query, text)` transforms both the query and the text through
the fitted vectorizer and returns their cosine similarity as a float in
[0.0, 1.0]. Higher means more relevant.

**[v2] Choosing snippets — confidence threshold (replaces integer threshold)**

`retrieve()` scores every section, filters to those at or above
`MIN_CONFIDENCE = 0.10`, sorts descending, and returns the top 3 as
`(filename, snippet, confidence)` triples. The confidence value is surfaced
to the user in all modes.

**What tradeoffs did you make?**

| Decision | Benefit | Cost |
|---|---|---|
| TF-IDF over word count | Rare domain terms score higher; common words stop causing false matches | Startup cost to fit vectorizer; depends on scikit-learn |
| Section-level chunking | Heading stays with its body; snippets are self-contained | Long sections still return large chunks |
| Cosine similarity over count | Normalised 0–1 score is comparable across queries | No stemming — `token` ≠ `tokens` |
| `MIN_CONFIDENCE = 0.10` threshold | Filters tangential matches before they reach the LLM | High-frequency words (`users`, `table`) score below threshold even when relevant |

---

## 3. Use of the LLM (Gemini)

**When does DocuBot call the LLM and when does it not?**

- **Mode 1 (Naive LLM):** Passes the full docs corpus to Gemini along with
  the query. The LLM reads all docs and synthesises an answer.
- **Mode 2 (Retrieval only):** Never calls the LLM. Returns raw snippets.
- **Mode 3 (RAG):** Retrieves top-k snippets first, then passes them to
  Gemini with strict grounding instructions.
- **[v2] Mode 4 (Agent):** May call Gemini up to `AGENTIC_MAX_RETRIES + 1`
  times per query — first for `rephrase_query()` if confidence is low, then
  once for `answer_from_snippets()`.

**What instructions do you give the LLM to keep it grounded?**

The RAG / Agent prompt in `llm_client.py` tells Gemini to:

- Answer **only** using the provided snippets — no invented functions,
  endpoints, or configuration values
- Reply with exactly `"I do not know based on the docs I have."` when
  snippets are insufficient
- Mention which files it relied on when it does answer

**[v2] What logging and error handling is in place?**

Every LLM call is wrapped in `_call_model()`, which:

- Appends a timestamped entry to `logs/docubot.log` for every request and
  response (query, snippet filenames, answer)
- Returns `EMPTY_RESPONSE_FALLBACK` if Gemini returns an empty string
- Catches all API exceptions, logs the full traceback, and returns a
  readable error string — the application never crashes on a bad response

---

## 4. Experiments and Comparisons

All four modes were run against the same sample queries. Results below are
from the live docs in `docs/` with a real Gemini API key.

| Query | Mode 1: Naive LLM | Mode 2: Retrieval | Mode 3: RAG | Mode 4: Agent | Notes |
|---|---|---|---|---|---|
| Where is the auth token generated? | Harmful — answers from general knowledge | Helpful — AUTH.md conf=0.16 | Helpful — cites AUTH.md | Helpful — high confidence on first attempt, no rephrase needed | All modes except naive agree |
| How do I connect to the database? | Harmful — generic DB advice | Helpful — DATABASE.md conf=0.21 | Helpful — cites DATABASE.md | Helpful — confident first attempt | RAG and Agent converge |
| Which endpoint lists all users? | Harmful — may invent endpoint path | Helpful — API_REFERENCE.md conf=0.10 | Helpful — cites API_REFERENCE.md | Helpful — borderline confidence, no rephrase triggered | TF-IDF confidence is low but above threshold |
| How does a client refresh an access token? | Harmful — generic OAuth flow | Helpful — AUTH.md conf=0.32 | Helpful — cites AUTH.md | Helpful — confident first attempt | Highest confidence query in the set |
| **[v2]** How do I set up credentials? | Harmful | No results (0.00) | No results → fallback | **Helpful** — agent rephrased to "environment variables for authentication", conf=0.38 | Agent is the only mode that succeeds here |
| **[v2]** Which fields are stored in the users table? | Harmful | No results (0.00) | No results → fallback | No results → fallback | `users` and `table` have low IDF weight; known limitation |

**[v2] Patterns observed after upgrade**

- **TF-IDF eliminated the main false-positive** from v1: `"what is my name?"`
  now returns zero results instead of pulling in project-name fields.
- **Confidence scores expose borderline retrievals** that were invisible in v1.
  A score of 0.10 warns the developer that the answer is weakly supported.
- **The agentic loop earns its cost** on vocabulary-mismatch queries — it
  rescued `"How do I set up credentials?"` which all other modes failed.
- **High-IDF-weight queries** (refresh, environment variables) already scored
  well in v1; TF-IDF made them even more reliable with higher, cleaner scores.

---

## 5. Failure Cases and Guardrails

**Failure case 1 — Low IDF weight on common words**

- Question: `"Which fields are stored in the users table?"`
- What happened: `users` and `table` appear in almost every doc section,
  so their IDF weight is near zero. No snippet clears `MIN_CONFIDENCE`.
  All four modes return the fallback string or nothing.
- What should have happened: DATABASE.md should be surfaced.
- **[v2] Root cause:** TF-IDF is corpus-size sensitive. In a 4-file corpus,
  common words are penalised too aggressively. Adding stopword customisation
  or using a larger corpus would fix this.

**[v2] Failure case 2 — No-context query**

- Question: `"Is there any mention of payment processing in these docs?"`
- What happened: SETUP.md was returned with conf=0.13 because it contains
  "processing" in an unrelated context. The LLM then correctly answered
  "no" based on the snippet — but the retrieval hit was coincidental.
- What should have happened: The system should ideally return no results
  and reply "I do not know based on these docs" without calling the LLM.

*(Failure case 2 from v1 — the missing heading bug — was fixed by
section-level chunking. It no longer occurs.)*

**When should DocuBot say "I do not know based on the docs I have"?**

1. When no snippet reaches `MIN_CONFIDENCE = 0.10` — the query has no
   meaningful match in the docs.
2. When retrieved snippets are tangentially related but don't answer the
   question — the RAG prompt instructs Gemini to refuse in this case.
3. **[v2]** When the agentic loop exhausts all retries and still finds only
   low-confidence snippets — `answer_agentic()` returns the fallback string
   without calling `answer_from_snippets()`.

**[v2] Guardrails summary**

| Guardrail | Where | What it does |
|---|---|---|
| `MIN_CONFIDENCE = 0.10` | `docubot.py` | Filters snippets with weak TF-IDF overlap |
| Empty result refusal | `docubot.py` | Returns fallback string without LLM if nothing retrieved |
| RAG prompt refusal rule | `llm_client.py` | Gemini instructed to say "I do not know" when snippets insufficient |
| `_call_model` exception handler | `llm_client.py` | Catches all API errors, logs traceback, returns safe string |
| Empty response guard | `llm_client.py` | Returns `EMPTY_RESPONSE_FALLBACK` instead of empty string |
| `rephrase_query` fallback | `llm_client.py` | Returns original query if rephrase API call fails |
| `AGENTIC_MAX_RETRIES = 2` cap | `docubot.py` | Limits LLM calls in the agentic loop |

---

## 6. Limitations and Future Improvements

**Current limitations**

1. **No stemming:** `token` and `tokens` are different vocabulary terms.
   Queries must match the exact word forms used in the docs.
2. **Corpus-size sensitivity:** IDF weights are computed over only 4 files.
   Very common words like `users` or `table` are underweighted relative to
   how useful they actually are for navigation.
3. **Section splitting is heading-dependent:** Docs without markdown headings
   fall through to full-document retrieval, which returns large, unfocused chunks.
4. **No cross-document reasoning:** Each snippet is scored independently.
   A question that requires combining information from two sections in different
   files will get only one of the two pieces.

**Future improvements**

1. **Add stemming or lemmatisation** (e.g. `nltk` Porter stemmer) so `token`
   and `tokens` score identically.
2. **Use embedding-based retrieval** (e.g. `sentence-transformers` + FAISS)
   to replace TF-IDF with semantic similarity — synonyms and paraphrases
   would then score highly even without exact word overlap.
3. **Custom stopword list** tuned to the corpus to prevent domain words
   like `users` from being downweighted by standard English IDF.
4. **Multi-chunk fusion:** retrieve the top-k snippets from multiple docs
   and pass all of them to the LLM as a single context, enabling
   cross-document answers.

---

## 7. Responsible Use

**Where could this system cause real-world harm if used carelessly?**

- **Mode 1 fabricates answers** that sound correct but are based on Gemini's
  general training data, not the actual codebase. A developer could follow
  wrong instructions and break their setup or introduce a security
  vulnerability.
- **Retrieval silently misses queries** that use different vocabulary from
  the docs. The system returns "I do not know" but does not tell the user
  that the information may exist under a different phrasing. The user might
  conclude the feature does not exist.
- **[v2] Confidence scores can be misread.** A score of 0.15 means weak
  TF-IDF overlap — it does not mean the answer is 15% correct. Users who
  treat confidence as an accuracy percentage may over-trust or under-trust
  results.
- **Stale docs produce stale answers** with no warning. If `AUTH.md` still
  describes an old token endpoint that has since changed, DocuBot will
  confidently answer from it.

**What instructions would you give real developers who want to use DocuBot safely?**

- Always verify critical answers (auth flows, database config, API
  contracts) against the actual source files. Do not act on DocuBot output
  alone.
- Avoid Mode 1 for project-specific questions. It does not read your docs.
- Treat `"I do not know based on these docs"` as a signal to either improve
  your documentation or try rephrasing your question (Mode 4 does this
  automatically).
- Keep the `docs/` folder up to date. Stale docs produce stale answers
  with no warning.
- Treat confidence scores as a retrieval signal, not an accuracy guarantee.
  A score above 0.30 indicates strong lexical overlap; below 0.15 warrants
  extra scepticism.
