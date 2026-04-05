"""
Text Preprocessing Module
Handles tokenization, stopword removal, lemmatization using NLTK and spaCy.
Domain-independent — no hardcoded field or skill terms.
"""
import re
import string

# ── Lazy imports ──────────────────────────────────────────────────────────────
_nlp = None
_stop_words = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
        except ImportError:
            _nlp = False
    return _nlp if _nlp is not False else None


def _get_stopwords():
    global _stop_words
    if _stop_words is None:
        try:
            from nltk.corpus import stopwords
            import nltk
            try:
                _stop_words = set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                nltk.download("punkt", quiet=True)
                nltk.download("wordnet", quiet=True)
                _stop_words = set(stopwords.words("english"))
        except ImportError:
            _stop_words = set()
    return _stop_words


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline:
    1. Lowercase
    2. Remove URLs, emails
    3. Remove punctuation (preserve hyphens inside words)
    4. Remove digit-only tokens
    5. Lemmatize with spaCy (fallback: stopword removal)
    Returns a cleaned string.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'(?<!\w)-(?!\w)', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    nlp = _get_nlp()
    stop_words = _get_stopwords()

    if nlp:
        doc = nlp(text)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and len(token.text) > 1
        ]
        text = " ".join(tokens)
    else:
        tokens = [w for w in text.split() if w not in stop_words and len(w) > 1]
        text = " ".join(tokens)

    return text


def extract_noun_phrases(text: str) -> list:
    """Extract noun phrases from text using spaCy."""
    nlp = _get_nlp()
    if not nlp:
        return _fallback_noun_phrases(text)
    doc = nlp(text[:100000])
    return [chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]


def _fallback_noun_phrases(text: str) -> list:
    """Regex-based noun phrase fallback when spaCy is unavailable."""
    stop_words = _get_stopwords()
    words = re.findall(r'\b[a-zA-Z][a-zA-Z\-]{1,}\b', text.lower())
    return [w for w in words if w not in stop_words and len(w) > 2]


def tokenize_sentences(text: str) -> list:
    """Split text into sentences."""
    try:
        import nltk
        return nltk.sent_tokenize(text)
    except Exception:
        return [s.strip() for s in text.split('.') if s.strip()]


def preprocess_for_skills(text: str) -> str:
    """Light cleaning for skill extraction (preserve case-sensitive tokens)."""
    text = text.lower()
    text = re.sub(r'[,\n•|/\\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()