# DocuBot Model Card

This model card is a short reflection on your DocuBot system. Fill it out after you have implemented retrieval and experimented with all three modes:

1. Naive LLM over full docs
2. Retrieval only
3. RAG (retrieval plus LLM)

Use clear, honest descriptions. It is fine if your system is imperfect.

---

## 1. System Overview

**What is DocuBot trying to do?**  
Describe the overall goal in 2 to 3 sentences.

DocuBot answers developer questions about a project by searching documentation files in the `docs/` folder. It finds the most relevant sections and either returns them directly or passes them to Gemini to generate a clean answer. The goal is to give accurate, doc-grounded responses instead of relying on a model's general knowledge.

**What inputs does DocuBot take?**  
For example: user question, docs in folder, environment variables.

- A natural language question typed by the user
- `.md` and `.txt` files inside the `docs/` folder
- A `GEMINI_API_KEY` environment variable (required for Modes 1 and 3)

**What outputs does DocuBot produce?**

- Mode 1: A free-form answer from Gemini based on its general knowledge (docs are not actually used)
- Mode 2: The raw text of the most relevant doc snippets, with filenames
- Mode 3: A Gemini-generated answer grounded in the retrieved snippets

---

## 2. Retrieval Design

**How does your retrieval system work?**  
Describe your choices for indexing and scoring.

- How do you turn documents into an index?
- How do you score relevance for a query?
- How do you choose top snippets?

**Indexing:** `build_index` splits every document's text on whitespace, lowercases and strips punctuation from each token, and maps each word to the list of filenames that contain it — a simple inverted index.

**Scoring:** `score_document` counts how many lowercased query words appear anywhere in the snippet text. Higher count = higher score.

**Choosing snippets:** `retrieve` first uses the index to find candidate documents (any doc containing at least one query word). It then splits each candidate into sections using markdown headings (`###`, `##`, `#`) so each section keeps its header and body together. It scores every section individually, keeps only those scoring `≥ MIN_RETRIEVAL_SCORE` (currently 3), sorts by score descending, and returns the top 3.

**What tradeoffs did you make?**  
For example: speed vs precision, simplicity vs accuracy.

- **Simplicity over accuracy:** word-count scoring ignores word order, synonyms, and context. "reset password" and "password reset" score identically.
- **Speed over precision:** the inverted index quickly narrows candidates, but common words like "the" or "name" can still pull in unrelated docs.
- **Section-level over sentence-level:** splitting on headings gives enough context per snippet but can still return large chunks when a section is long.

---

## 3. Use of the LLM (Gemini)

**When does DocuBot call the LLM and when does it not?**  
Briefly describe how each mode behaves.

- **Naive LLM mode:** sends only the question to Gemini with no doc context. The `all_text` argument is ignored. Gemini answers from its general training data, not from your docs.
- **Retrieval only mode:** never calls the LLM. Returns raw snippets directly to the user.
- **RAG mode:** retrieves the top snippets first, then passes them to Gemini along with strict instructions to answer only from those snippets.

**What instructions do you give the LLM to keep it grounded?**  
Summarize the rules from your prompt. For example: only use snippets, say "I do not know" when needed, cite files.

The RAG prompt in `llm_client.py` tells Gemini to:

- Answer only using the provided snippets — no invented functions, endpoints, or config values
- Reply with exactly `"I do not know based on these docs."` if the snippets are not enough
- Mention which files it relied on when it does answer

---

## 4. Experiments and Comparisons

Run the **same set of queries** in all three modes. Fill in the table with short notes.

You can reuse or adapt the queries from `dataset.py`.

| Query                                      | Naive LLM: helpful or harmful?                          | Retrieval only: helpful or harmful?                          | RAG: helpful or harmful?                     | Notes                                                    |
| ------------------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------- | -------------------------------------------------------- |
| Where is the auth token generated?         | Harmful — answers from general knowledge, not your code | Helpful — returns the correct AUTH.md section                | Helpful — clean answer citing AUTH.md        | Naive LLM fabricates details                             |
| How do I connect to the database?          | Harmful — gives generic advice unrelated to your setup  | Helpful — returns DATABASE.md connection config              | Helpful — accurate answer citing DATABASE.md | RAG and retrieval agree here                             |
| Which endpoint lists all users?            | Harmful — may describe a plausible but wrong endpoint   | Helpful — returns `GET /api/users` section with full context | Helpful — cites API_REFERENCE.md correctly   | Paragraph-level retrieval fixed the missing header issue |
| How does a client refresh an access token? | Harmful — generic OAuth explanation, not your API       | Helpful — returns `POST /api/refresh` section                | Helpful — accurate answer with file citation | All three diverge most here                              |

**What patterns did you notice?**

- **Naive LLM looks impressive but is untrustworthy** when the question sounds generic (e.g. "how do I connect to a database?") — it gives a fluent, confident answer that has nothing to do with your actual project config.
- **Retrieval only is clearly better** when you need to verify the exact text from the docs — no hallucination risk, you see the raw source.
- **RAG is clearly better than both** when you want a readable answer that is still grounded — it combines Gemini's fluency with the accuracy of retrieval. It fails only when retrieval fails first.

---

## 5. Failure Cases and Guardrails

**Describe at least two concrete failure cases you observed.**

**Failure case 1:**

- Question: `"what is my name?"`
- What happened: The system returned API and database snippets because `"name"` matched JSON field names like `"name": "Alpha Project"` in the docs.
- What should have happened: The system should have refused — no snippet contains meaningful evidence about the user's name.

**Failure case 2:**

- Question: `"Which endpoint returns all users?"` (before the paragraph-level fix)
- What happened: The system returned only `"Returns a list of all users. Only accessible to admins."` — the `### GET /api/users` header was in a separate chunk and got dropped.
- What should have happened: The snippet should include the heading so the user knows the actual endpoint path.

**When should DocuBot say "I do not know based on the docs I have"?**  
Give at least two specific situations.

1. When no snippet scores at or above `MIN_RETRIEVAL_SCORE` — the query has no meaningful match in the docs.
2. When the retrieved snippets are tangentially related (e.g. a field named "name" matched "what is my name") but don't actually answer the question.

**What guardrails did you implement?**  
Examples: refusal rules, thresholds, limits on snippets, safe defaults.

- **`MIN_RETRIEVAL_SCORE = 3`** — a snippet must match at least 3 query words to be returned. Single-word coincidental matches are rejected.
- **Empty result refusal** — if `retrieve` returns no snippets, both `answer_retrieval_only` and `answer_rag` return `"I do not know based on these docs."` instead of guessing.
- **RAG prompt refusal rule** — Gemini is explicitly told to say `"I do not know based on these docs."` if the snippets are insufficient.

---

## 6. Limitations and Future Improvements

**Current limitations**

1. **No semantic understanding**: scoring only counts exact word matches. Synonyms, paraphrases, and related concepts get a score of zero.
2. **Common words cause false matches**: words like "name", "is", or "user" appear everywhere and can pull in irrelevant snippets even with a raised threshold.
3. **Section splitting is fragile**: docs without markdown headings fall back to `\n\n` splits, which may cut context at the wrong place.

**Future improvements**

1. **Add stopword filtering**: exclude words like "what", "is", "my" from scoring so only content words count toward the score.
2. **Use embedding-based retrieval**: replace word-count scoring with vector similarity so semantically related content scores highly even without exact word overlap.
3. **Chunk overlap**: when splitting into sections, include the parent heading in every child chunk so context is never lost.

---

## 7. Responsible Use

**Where could this system cause real world harm if used carelessly?**  
Think about wrong answers, missing information, or over trusting the LLM.

- **Naive LLM mode fabricates answers** that sound correct but are based on general training data, not your actual codebase. A developer could follow wrong instructions and break their setup.
- **Retrieval can miss critical docs** if the user phrases the question differently from the doc's wording. The system silently returns nothing instead of flagging the gap.
- **Over-trusting RAG output**: if the retrieved snippet is outdated or incomplete, Gemini will still generate a confident-sounding answer based on it.

**What instructions would you give real developers who want to use DocuBot safely?**

- Always verify critical answers (auth flows, database config, API contracts) against the actual source files. Do not act on DocuBot output alone.
- Avoid using Naive LLM mode (Mode 1) for project-specific questions. It does not read your docs.
- Keep the `docs/` folder up to date. Stale docs produce stale answers with no warning.
- Treat "I do not know based on these docs" as a signal to improve your documentation, not just a dead end.

---
