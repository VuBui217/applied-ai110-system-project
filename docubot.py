"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a retrieval index using TF-IDF (enhanced from Phase 1)
- Retrieving relevant snippets with confidence scores (0.0-1.0)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
- Supporting agentic query refinement (Phase 4 / Mode 4)
"""

import os
import glob
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Minimum TF-IDF cosine similarity a snippet must reach to be returned.
# Replaces the old integer MIN_RETRIEVAL_SCORE word-count threshold.
MIN_CONFIDENCE = 0.10

# Confidence below this triggers a retry in agentic mode.
AGENTIC_CONFIDENCE_THRESHOLD = 0.15

# Maximum number of query-refinement retries in agentic mode.
AGENTIC_MAX_RETRIES = 2


class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client:  optional Gemini client for LLM-based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load raw documents into memory
        self.documents = self.load_documents()          # [(filename, text), ...]

        # Build inverted index for fast candidate filtering
        self.index = self.build_index(self.documents)

        # Split every doc into sections; fit TF-IDF over all sections
        self.paragraphs = self._extract_all_paragraphs()  # [(filename, section), ...]
        self.vectorizer, self.paragraph_vectors = self._fit_tfidf()

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of (filename, text) tuples.
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Inverted Index (fast candidate filtering)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Builds a simple inverted index mapping lowercase words to the
        filenames they appear in.  Used to narrow candidates quickly
        before TF-IDF scoring.

        Example: {"token": ["AUTH.md", "API_REFERENCE.md"], ...}
        """
        index = {}
        for filename, text in documents:
            for word in text.lower().split():
                word = word.strip(".,!?;:\"'()[]{}")
                if word not in index:
                    index[word] = []
                if filename not in index[word]:
                    index[word].append(filename)
        return index

    # -----------------------------------------------------------
    # TF-IDF Setup
    # -----------------------------------------------------------

    def _extract_all_paragraphs(self):
        """
        Splits every document into sections using markdown headings as
        boundaries, keeping the heading attached to its body.
        Returns a flat list of (filename, section_text) tuples.
        """
        all_paragraphs = []
        for filename, text in self.documents:
            sections = re.split(r'(?=^#{1,3} )', text, flags=re.MULTILINE)
            for section in sections:
                section = section.strip()
                if section:
                    all_paragraphs.append((filename, section))
        return all_paragraphs

    def _fit_tfidf(self):
        """
        Fits a TF-IDF vectorizer over all extracted paragraphs.
        Uses unigrams + bigrams and removes English stop words.
        Returns (vectorizer, paragraph_vectors).
        """
        texts = [para for _, para in self.paragraphs]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        paragraph_vectors = vectorizer.fit_transform(texts)
        return vectorizer, paragraph_vectors

    # -----------------------------------------------------------
    # Scoring and Retrieval (TF-IDF enhanced)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Returns a cosine similarity score (0.0-1.0) between the query
        and a piece of text using the fitted TF-IDF vectorizer.
        Higher score = more relevant.
        """
        query_vec = self.vectorizer.transform([query])
        text_vec = self.vectorizer.transform([text])
        score = cosine_similarity(query_vec, text_vec)[0][0]
        return float(score)

    def retrieve(self, query, top_k=3):
        """
        Scores all paragraphs against the query using TF-IDF cosine
        similarity and returns the top_k most relevant results.

        Returns a list of (filename, snippet, confidence) triples
        sorted by confidence descending.
        Only snippets at or above MIN_CONFIDENCE are included.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.paragraph_vectors)[0]

        results = []
        for i, (filename, paragraph) in enumerate(self.paragraphs):
            confidence = float(scores[i])
            if confidence >= MIN_CONFIDENCE:
                results.append((confidence, filename, paragraph))

        results.sort(key=lambda x: x[0], reverse=True)
        return [
            (filename, snippet, confidence)
            for confidence, filename, snippet in results[:top_k]
        ]

    def average_confidence(self, snippets):
        """
        Returns the mean confidence across a list of
        (filename, snippet, confidence) triples.
        Returns 0.0 for an empty list.
        """
        if not snippets:
            return 0.0
        return sum(c for _, _, c in snippets) / len(snippets)

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Mode 2 — retrieval only.
        Returns raw snippets with filenames and confidence scores.
        No LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text, confidence in snippets:
            formatted.append(f"[{filename}] (confidence: {confidence:.2f})\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Mode 3 — RAG.
        Retrieves snippets then passes them to Gemini for a grounded answer.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        # LLM client expects (filename, text) pairs — strip confidence
        snippet_pairs = [(filename, text) for filename, text, _ in snippets]
        return self.llm_client.answer_from_snippets(query, snippet_pairs)

    def answer_agentic(self, query, top_k=3):
        """
        Mode 4 — agentic query refinement.

        Step 1: Retrieve with the original query.
        Step 2: If average confidence is below AGENTIC_CONFIDENCE_THRESHOLD
                and retries remain, ask Gemini to rephrase the query.
        Step 3: Retrieve again with the rephrased query.
        Step 4: Repeat up to AGENTIC_MAX_RETRIES times.
        Step 5: Answer using the best retrieval result found.

        Prints each intermediate step so the reasoning is visible.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "Agentic mode requires an LLM client. Provide a GeminiClient instance."
            )

        current_query = query
        best_snippets = []
        best_confidence = 0.0

        for attempt in range(1, AGENTIC_MAX_RETRIES + 2):  # +2: initial + retries
            print(f"  [Agent attempt {attempt}] Query: \"{current_query}\"")

            snippets = self.retrieve(current_query, top_k=top_k)
            avg_conf = self.average_confidence(snippets)

            print(f"  [Agent attempt {attempt}] Avg confidence: {avg_conf:.2f}")

            if avg_conf > best_confidence:
                best_confidence = avg_conf
                best_snippets = snippets

            # Stop early if confidence is good enough or no retries left
            if avg_conf >= AGENTIC_CONFIDENCE_THRESHOLD:
                print(f"  [Agent] Confidence threshold met. Proceeding to answer.\n")
                break

            if attempt == AGENTIC_MAX_RETRIES + 1:
                print(f"  [Agent] Max retries reached. Using best result found.\n")
                break

            # Ask Gemini to rephrase the query
            print(f"  [Agent] Low confidence — asking Gemini to rephrase query...")
            current_query = self.llm_client.rephrase_query(current_query)
            print(f"  [Agent] Rephrased query: \"{current_query}\"")

        if not best_snippets:
            return "I do not know based on these docs."

        snippet_pairs = [(filename, text) for filename, text, _ in best_snippets]
        return self.llm_client.answer_from_snippets(query, snippet_pairs)

    # -----------------------------------------------------------
    # Helper: full corpus text for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        Used in Mode 1 (naive LLM) as the full context.
        """
        return "\n\n".join(text for _, text in self.documents)
