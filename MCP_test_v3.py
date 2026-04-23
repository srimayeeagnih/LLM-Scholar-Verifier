"""
Academic Paper Search Tool
Searches arXiv and Semantic Scholar for papers matching user input,
extracts keywords, filters results, and optionally downloads full text from PDFs.
"""
import re
import time
import requests
import xml.etree.ElementTree as ET
from collections import Counter
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdfminer.high_level import extract_text
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MAX_RESULTS_PER_SOURCE = 100
TRIAGE_TOP_N = 50  # Max papers to keep after abstract-level TF-IDF triage


# ──────────────────────────────────────────────
# Claim Parsing (full query retrieval)
# ──────────────────────────────────────────────
def parse_claims(user_query):
    """
    Parse a user query into individual claims, preserving all words (including stop words).

    Batching logic:
    - If the query contains title/author patterns (e.g., "Title: ...", "Author: ...",
      or quoted titles), split by those markers.
    - Otherwise, split by blank lines (paragraph breaks).
    - If neither, return the entire query as a single claim.

    Returns:
        List of dicts with keys: 'text' (full claim text), 'title', 'author' (if detected).
    """
    claims = []

    # Pattern: lines starting with title/author markers
    title_pattern = re.compile(r'^(?:title|paper|article)\s*[:\-]\s*(.+)', re.IGNORECASE)
    author_pattern = re.compile(r'^(?:author|by|written by)\s*[:\-]\s*(.+)', re.IGNORECASE)

    # Split into paragraphs by blank lines
    paragraphs = re.split(r'\n\s*\n', user_query.strip())

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        claim = {'text': para, 'title': None, 'author': None}

        # Scan lines for title/author markers
        lines = para.split('\n')
        non_meta_lines = []
        for line in lines:
            line_stripped = line.strip()
            t_match = title_pattern.match(line_stripped)
            a_match = author_pattern.match(line_stripped)
            if t_match:
                claim['title'] = t_match.group(1).strip().strip('"\'')
            elif a_match:
                claim['author'] = a_match.group(1).strip()
            else:
                non_meta_lines.append(line_stripped)

        # Also detect quoted titles inline, but only if followed by "by Author Name"
        if not claim['title']:
            by_match = re.search(r'"([^"]{10,})"\s+by\s+(.+?)(?:\.|,|$)', para)
            if by_match:
                claim['title'] = by_match.group(1)
                if not claim['author']:
                    claim['author'] = by_match.group(2).strip()

        # Keep full text including stop words
        if non_meta_lines:
            claim['text'] = ' '.join(non_meta_lines)

        claims.append(claim)

    return claims


# ──────────────────────────────────────────────
# Keyword Extraction
# ──────────────────────────────────────────────
def extract_keywords(text, min_length=4, top_n=10):
    """Extract keywords from user input text, prioritizing capitalized words, acronyms, and numeric terms.

    Priority order:
      1. Acronyms  – all-uppercase tokens (e.g. SEO, URLs), no min-length restriction.
      2. Proper nouns – capitalized words that are NOT at the start of a sentence.
         Only the *first* word of a consecutive capitalized run is kept so that
         "Google Search Console" yields "google" rather than all three words.
      3. Numeric terms – percentages, fractions, and number+unit measurements.
      4. Regular keywords – remaining words ranked by frequency (includes
         alphanumeric identifiers like p53, covid19).
    """
    # --- Detect sentence-start positions (grammatical caps to ignore) ---
    sentence_start_positions = set()
    first_match = re.match(r'\s*([a-zA-Z]+)', text)
    if first_match:
        sentence_start_positions.add(first_match.start(1))
    for m in re.finditer(r'[.!?]\s+([a-zA-Z]+)', text):
        sentence_start_positions.add(m.start(1))

    # --- Pass 1: Collect acronyms & leading proper nouns from original text ---
    acronyms = []
    proper_nouns = []
    prev_was_capitalized = False

    for match in re.finditer(r'\b[a-zA-Z]+\b', text):
        word = match.group()
        pos = match.start()
        lower = word.lower()

        if lower in STOPWORDS:
            prev_was_capitalized = False
            continue

        # Acronym: 2+ uppercase letters, optional trailing 's' (SEO, URLs, API)
        if re.match(r'^[A-Z]{2,}s?$', word):
            if lower not in acronyms:
                acronyms.append(lower)
            prev_was_capitalized = True

        # Proper noun: capitalised, not a sentence opener, first of a cap-run
        elif word[0].isupper() and pos not in sentence_start_positions:
            if not prev_was_capitalized and lower not in proper_nouns and lower not in acronyms:
                proper_nouns.append(lower)
            prev_was_capitalized = True

        else:
            prev_was_capitalized = False

    # --- Pass 2: Numeric terms (percentages, fractions, measurements) ---
    numeric_terms = []
    seen_numeric = set()

    # Percentages: 50%, 3.5%, "50 percent"
    for m in re.finditer(r'\d+(?:\.\d+)?\s*(?:%|percent)', text, re.IGNORECASE):
        token = re.sub(r'\s+', '', m.group().lower())   # normalise "50 %" → "50%"
        if token not in seen_numeric:
            seen_numeric.add(token)
            numeric_terms.append(token)

    # Fractions: 1/3, 2/3
    for m in re.finditer(r'\b\d+/\d+\b', text):
        token = m.group()
        if token not in seen_numeric:
            seen_numeric.add(token)
            numeric_terms.append(token)

    # Number + unit measurements
    _MEAS_RE = re.compile(
        r'\b(\d+(?:\.\d+)?)\s*'
        r'(degrees?\s*(?:celsius|fahrenheit|kelvin|[cfk])'
        r'|°[cfk]'
        r'|mg/dl|g/dl|mmhg|mm\s*hg'
        r'|m[lg]|kg|[gc]m|mm|nm|µm|µg|µl'
        r'|mol(?:/l)?|mmol|µmol'
        r'|hz|khz|mhz|ghz'
        r'|kpa|mpa|atm|pa|bar'
        r'|[mk]?w|kj|mj|ev'
        r'|ppm|ppb)\b',
        re.IGNORECASE,
    )
    for m in _MEAS_RE.finditer(text):
        token = re.sub(r'\s+', ' ', m.group().strip().lower())
        if token not in seen_numeric:
            seen_numeric.add(token)
            numeric_terms.append(token)

    # --- Pass 3: Regular keywords by frequency (includes alphanumeric tokens) ---
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text.lower())
    regular = [w for w in words if len(w) >= min_length and w not in STOPWORDS]
    regular_ranked = [word for word, _ in Counter(regular).most_common(top_n)]

    # --- Combine: acronyms → proper nouns → numeric → regular, deduplicated ---
    seen = set()
    combined = []
    for word in acronyms + proper_nouns + numeric_terms + regular_ranked:
        if word not in seen:
            seen.add(word)
            combined.append(word)

    final_keywords = combined[:top_n]

    # --- POS tagging: build a parallel dict mapping keyword → POS tag ---
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    # Map lowercased tokens to their POS tag (first occurrence wins)
    token_pos = {}
    for tok, tag in tagged:
        low = tok.lower()
        if low not in token_pos:
            token_pos[low] = tag

    pos_tags = {}
    for kw in final_keywords:
        pos_tags[kw] = token_pos.get(kw, 'CD' if any(c.isdigit() for c in kw) else 'NN')

    return final_keywords, pos_tags


def expand_with_synonyms(keywords, max_synonyms=4):
    """Expand each keyword with WordNet synonyms.

    Returns list of sets, one per keyword: {original, syn1, syn2, ...}
    """
    expanded = []
    for kw in keywords:
        synset_group = {kw}
        for syn in wordnet.synsets(kw):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name != kw and len(name) >= 3:
                    synset_group.add(name)
                    if len(synset_group) >= max_synonyms + 1:
                        break
            if len(synset_group) >= max_synonyms + 1:
                break
        expanded.append(synset_group)
    return expanded


# ──────────────────────────────────────────────
# Counterfactual Agreement Analysis
# ──────────────────────────────────────────────
STOPWORDS = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are', 'was',
             'were', 'been', 'have', 'has', 'will', 'would', 'could', 'should',
             'about', 'into', 'through', 'during', 'before', 'after', 'above',
             'below', 'between', 'under', 'again', 'further', 'then', 'once',
             'also', 'more', 'most', 'other', 'some', 'such', 'than', 'too',
             'very', 'can', 'just', 'but', 'not', 'only', 'same', 'our', 'which',
             'their', 'there', 'these', 'those', 'what', 'when', 'where', 'who',
             'how', 'all', 'each', 'every', 'both', 'few', 'many', 'several',
             'her', 'him', 'his', 'its', 'she', 'they', 'you', 'your',
             'is', 'it', 'in', 'on', 'at', 'to', 'of', 'by', 'as', 'or',
             'an', 'a', 'be', 'do', 'no', 'so', 'if', 'up', 'we', 'my', 'me',
             'he', 'us', 'am', 'did', 'had', 'may', 'own', 'nor', 'any',
             'does', 'without', 'never', 'none', 'cannot', 'neither'}


def _negate_sentence(sentence):
    """Negate a single sentence by inserting or removing 'not' at the first auxiliary verb.

    If no auxiliary verb is found, inserts 'do not' / 'does not' before the main verb
    using POS tagging.

    Returns the negated sentence string.
    """
    AUX_VERBS = {'is', 'are', 'was', 'were', 'do', 'does', 'did',
                 'has', 'have', 'had', 'will', 'would', 'could',
                 'should', 'can', 'may', 'might', 'shall'}

    tokens = word_tokenize(sentence)

    for i, tok in enumerate(tokens):
        if tok.lower() in AUX_VERBS:
            # Check if next token is "not" or contraction "n't"
            if i + 1 < len(tokens) and tokens[i + 1].lower() in ('not', "n't"):
                # Remove the negation
                result = tokens[:i + 1] + tokens[i + 2:]
                return ' '.join(result)
            else:
                # Insert "not" after the auxiliary verb
                result = tokens[:i + 1] + ['not'] + tokens[i + 1:]
                return ' '.join(result)

    # Fallback: no auxiliary verb — insert "do not" / "does not" before the main verb
    tagged = pos_tag(tokens)
    for i, (tok, tag) in enumerate(tagged):
        if tag.startswith('VB') and tok.lower() not in AUX_VERBS:
            # Use "does not" for 3rd person singular present (VBZ), "do not" otherwise
            aux = 'does not' if tag == 'VBZ' else 'do not'
            result = tokens[:i] + aux.split() + tokens[i:]
            return ' '.join(result)

    # Last resort: prepend "do not" before the whole sentence
    return f"do not {sentence[0].lower()}{sentence[1:]}"


def generate_counterfactual(claim_text):
    """Generate a negated (counterfactual) version of the claim.

    Splits the claim into individual sentences and negates each one independently.
    For each sentence:
    - Find the first auxiliary/copula verb (is, are, was, were, do, does, did,
      has, have, had, will, would, could, should, can, may).
    - If "not" (or "n't") immediately follows it, remove the negation.
    - Otherwise, insert "not" after the verb.
    - Fallback: prepend "It is not the case that" if no verb is found.

    Returns:
        The counterfactual claim string with all sentences negated.
    """
    sentences = re.split(r'(?<=[.!?])\s+', claim_text.strip())
    negated = [_negate_sentence(s) for s in sentences if s.strip()]
    return ' '.join(negated)


def counterfactual_agreement(claim_text, counterfactual_text, papers):
    """Compare each paper's top sentences to both the original and counterfactual claims.

    Uses TF-IDF cosine similarity to determine whether each sentence is closer
    to the original claim (agree) or the counterfactual (disagree).

    Updates each paper dict in place with:
        'sentence_comparisons': list of dicts with keys:
            'sentence', 'retrieval_score', 'sim_original', 'sim_counterfactual', 'verdict'

    Args:
        claim_text:          The original user claim.
        counterfactual_text: The negated version of the claim.
        papers:              List of paper dicts (must have 'top_sentences').
    """
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

    # Collect all unique sentences across papers
    all_sentences = []
    paper_sentence_indices = {}  # paper_idx -> list of indices into all_sentences

    for idx, paper in enumerate(papers):
        top_sents = paper.get('top_sentences', [])
        indices = []
        for sent_text, _ in top_sents:
            indices.append(len(all_sentences))
            all_sentences.append(sent_text)
        paper_sentence_indices[idx] = indices

    if not all_sentences:
        for paper in papers:
            paper['sentence_comparisons'] = []
        return

    # Build TF-IDF matrix: [original_claim, counterfactual, sent1, sent2, ...]
    # No stop word filtering here — negation words ("not", "do not") are the key signal.
    corpus = [claim_text, counterfactual_text] + all_sentences
    vectorizer = TfidfVectorizer(max_features=20000)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    claim_vec = tfidf_matrix[0:1]
    counter_vec = tfidf_matrix[1:2]
    sentence_vecs = tfidf_matrix[2:]

    # Compute similarities
    sims_original = sklearn_cosine(claim_vec, sentence_vecs)[0]
    sims_counterfactual = sklearn_cosine(counter_vec, sentence_vecs)[0]

    # Assign verdicts back to papers
    total_agree = 0
    total_disagree = 0
    total_neutral = 0

    for idx, paper in enumerate(papers):
        top_sents = paper.get('top_sentences', [])
        comparisons = []

        for j, (sent_text, retrieval_score) in enumerate(top_sents):
            sent_idx = paper_sentence_indices[idx][j]
            sim_orig = float(sims_original[sent_idx])
            sim_counter = float(sims_counterfactual[sent_idx])

            if sim_orig > sim_counter:
                verdict = 'agree'
                total_agree += 1
            elif sim_counter > sim_orig:
                verdict = 'disagree'
                total_disagree += 1
            else:
                verdict = 'neutral'
                total_neutral += 1

            comparisons.append({
                'sentence': sent_text,
                'retrieval_score': retrieval_score,
                'sim_original': sim_orig,
                'sim_counterfactual': sim_counter,
                'verdict': verdict,
            })

        paper['sentence_comparisons'] = comparisons

    print(f"  Counterfactual analysis: {len(all_sentences)} sentences scored")
    print(f"  Agree: {total_agree}  |  Disagree: {total_disagree}  |  Neutral: {total_neutral}")


# ──────────────────────────────────────────────
# Query Builders
# ──────────────────────────────────────────────
def build_arxiv_query(keywords):
    """Build an arXiv API search query from keyword groups (with synonyms).

    Filters out multi-word synonyms (arXiv 'all:' fields only support single words)
    and limits synonyms per group to avoid overly complex queries that cause 503 errors.
    """
    if keywords and isinstance(keywords[0], set):
        groups = []
        for group in keywords:
            # Only keep single-word synonyms (multi-word breaks arXiv all: syntax)
            single_word = [w for w in group if ' ' not in w][:3]
            if single_word:
                groups.append('(' + ' OR '.join(f'all:{w}' for w in single_word) + ')')
        return ' AND '.join(groups)
    terms = [f'all:{kw}' for kw in keywords]
    return ' AND '.join(terms)


def build_semantic_scholar_query(keywords):
    """Build a Semantic Scholar search query from keyword groups (with synonyms)."""
    if keywords and isinstance(keywords[0], set):
        # Semantic Scholar uses plain text, take original + top synonym per group
        terms = []
        for group in keywords:
            terms.extend(list(group)[:2])
        return ' '.join(terms)
    return ' '.join(keywords)


# ──────────────────────────────────────────────
# Database Search Functions
# ──────────────────────────────────────────────
def search_arxiv(query, max_results=10, _retried=False):
    """Search arXiv and return a list of paper dicts.

    On 503 errors, retries once with a simplified query (strips OR groups down
    to the first term in each group) and a reduced result count.
    """
    url = 'http://export.arxiv.org/api/query'
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
    }
    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 503 and not _retried:
        # Simplify: strip synonym OR groups, keep only first term per group
        simple_query = re.sub(r'\(([^)]*?)\s+OR\s+[^)]*\)', r'\1', query)
        print(f"  arXiv 503, retrying with simplified query: {simple_query}")
        time.sleep(5)
        return search_arxiv(simple_query, max_results=min(max_results, 30), _retried=True)
    response.raise_for_status()

    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(response.text)

    papers = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns)
        summary = entry.find('atom:summary', ns)
        authors = entry.findall('atom:author/atom:name', ns)

        # Try to find a PDF link
        pdf_url = None
        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href')
                break

        papers.append({
            'title': title.text.strip() if title is not None else 'No title',
            'abstract': summary.text.strip() if summary is not None else 'No abstract',
            'authors': [a.text for a in authors],
            'pdf_url': pdf_url,
            'full_text': summary.text.strip() if summary is not None else '',
            'source': 'arxiv',
        })
    return papers


def search_semantic_scholar(query, limit=10):
    """Search Semantic Scholar and return raw JSON response."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,abstract,url,openAccessPdf",
    }
    response = requests.get(url, params=params, timeout=30)
    return response.json()


# ──────────────────────────────────────────────
# PDF Text Extraction
# ──────────────────────────────────────────────
def extract_text_from_pdf_url(pdf_url):
    """Download PDF and extract text content."""
    response = requests.get(pdf_url)
    response.raise_for_status()
    pdf_file = BytesIO(response.content)
    text = extract_text(pdf_file)
    return text


# ──────────────────────────────────────────────
# Batched Parallel PDF Download
# ──────────────────────────────────────────────
def _download_one_pdf(paper):
    """Download a single paper's PDF and return (paper, full_text_or_None)."""
    pdf_url = paper.get('pdf_url')
    if not pdf_url:
        return paper, None
    if paper.get('full_text', '') != paper.get('abstract', ''):
        return paper, None  # already has full text
    try:
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
        pdf_bytes = BytesIO(resp.content)
        with ThreadPoolExecutor(max_workers=1) as tex:
            future = tex.submit(extract_text, pdf_bytes)
            try:
                text = future.result(timeout=60)
            except Exception:
                return paper, None
        if text and len(text.strip()) > len(paper.get('abstract', '')):
            return paper, text.strip()
    except Exception:
        pass
    return paper, None


def download_pdfs_batched(papers, batch_size=10, max_workers=5):
    """Download PDFs in parallel batches, updating papers in place.

    Args:
        papers:      List of paper dicts (must have 'pdf_url', 'abstract', 'full_text').
        batch_size:  Number of papers per batch.
        max_workers: Threads per batch.
    """
    seen_urls = set()
    candidates = []
    for p in papers:
        url = p.get('pdf_url')
        if url and p.get('full_text', '') == p.get('abstract', '') and url not in seen_urls:
            seen_urls.add(url)
            candidates.append(p)

    if not candidates:
        print("  No PDFs to download.")
        return

    total = len(candidates)
    downloaded = 0

    for batch_start in range(0, total, batch_size):
        batch = candidates[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        print(f"  Batch {batch_num}: downloading {len(batch)} PDFs in parallel...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_download_one_pdf, p): p for p in batch}
            for future in as_completed(futures):
                paper, text = future.result()
                if text:
                    paper['full_text'] = text
                    downloaded += 1
                    print(f"    OK  {paper['title'][:55]}... ({len(text)} chars)")
                else:
                    print(f"    --  {paper['title'][:55]}... (kept abstract)")

    print(f"  Done: {downloaded}/{total} PDFs extracted successfully.")


# ──────────────────────────────────────────────
# Sentence Extraction
# ──────────────────────────────────────────────
def extract_sentences(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ──────────────────────────────────────────────
# TF-IDF + KNN Sentence Retrieval (Abstract level)
# ──────────────────────────────────────────────
def tfidf_knn_rank_papers(claim_text, papers, top_sentences_per_paper=5, knn_k=20):
    """Rank papers by TF-IDF + KNN sentence similarity to the claim.

    1. Extract sentences from every paper's full text.
    2. Fit a single TF-IDF matrix over all sentences + the claim.
    3. Use KNN (cosine) to find the K nearest sentences to the claim.
    4. Score each paper by its best-matching sentence.

    Updates each paper dict in place with:
        'top_sentences':   [(sentence, score), ...]
        'best_similarity': float

    Args:
        claim_text:             The user's claim string.
        papers:                 List of paper dicts.
        top_sentences_per_paper: Max sentences to keep per paper.
        knn_k:                  Neighbours to retrieve from KNN.
    """
    # ── Build corpus: collect all sentences, track which paper they belong to ──
    all_sentences = []
    sentence_meta = []  # (paper_index, sentence_text)

    for idx, paper in enumerate(papers):
        text = paper.get('full_text') or paper.get('abstract') or ''
        sents = extract_sentences(text)
        for s in sents:
            all_sentences.append(s)
            sentence_meta.append(idx)

    if not all_sentences:
        print("  No sentences extracted from any paper.")
        for paper in papers:
            paper['top_sentences'] = []
            paper['best_similarity'] = 0.0
        return

    # ── TF-IDF: claim is index 0, sentences follow ──
    corpus = [claim_text] + all_sentences
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    claim_vec = tfidf_matrix[0:1]          # shape (1, n_features)
    sentence_vecs = tfidf_matrix[1:]       # shape (n_sentences, n_features)

    # ── KNN: find closest sentences to the claim ──
    effective_k = min(knn_k, sentence_vecs.shape[0])
    knn = NearestNeighbors(n_neighbors=effective_k, metric='cosine', algorithm='brute')
    knn.fit(sentence_vecs)
    distances, indices = knn.kneighbors(claim_vec)

    # cosine distance → cosine similarity
    similarities = 1.0 - distances[0]
    neighbor_indices = indices[0]

    # ── Assign scores back to papers ──
    paper_sentences = {i: [] for i in range(len(papers))}
    for sent_idx, sim in zip(neighbor_indices, similarities):
        paper_idx = sentence_meta[sent_idx]
        sent_text = all_sentences[sent_idx]
        paper_sentences[paper_idx].append((sent_text, float(sim)))

    total_matched = 0
    for idx, paper in enumerate(papers):
        scored = sorted(paper_sentences[idx], key=lambda x: x[1], reverse=True)
        paper['top_sentences'] = scored[:top_sentences_per_paper]
        paper['best_similarity'] = scored[0][1] if scored else 0.0
        if scored:
            total_matched += 1

    print(f"  TF-IDF corpus: {len(all_sentences)} sentences from {len(papers)} papers")
    print(f"  KNN retrieved {effective_k} nearest sentences")
    print(f"  {total_matched}/{len(papers)} papers have at least one matching sentence")


# ──────────────────────────────────────────────
# Doc2Vec Sentence Retrieval
# ──────────────────────────────────────────────
def doc2vec_rank_papers(claim_text, papers, top_sentences_per_paper=5,
                        vector_size=100, epochs=20, window=5):
    """Rank papers by Doc2Vec sentence similarity to the claim.

    1. Extract sentences from every paper's full text.
    2. Train a Doc2Vec model on all sentences.
    3. Infer a vector for the claim.
    4. Rank sentences by cosine similarity to the claim vector.

    Updates each paper dict in place with:
        'd2v_top_sentences':   [(sentence, score), ...]
        'd2v_best_similarity': float

    Args:
        claim_text:             The user's claim string.
        papers:                 List of paper dicts.
        top_sentences_per_paper: Max sentences to keep per paper.
        vector_size:            Dimensionality of Doc2Vec embeddings.
        epochs:                 Training epochs for Doc2Vec.
        window:                 Context window size for Doc2Vec.
    """
    import numpy as np

    # ── Build corpus: collect all sentences, track which paper they belong to ──
    all_sentences = []
    sentence_meta = []  # paper_index per sentence

    for idx, paper in enumerate(papers):
        text = paper.get('full_text') or paper.get('abstract') or ''
        sents = extract_sentences(text)
        for s in sents:
            all_sentences.append(s)
            sentence_meta.append(idx)

    if not all_sentences:
        print("  No sentences extracted from any paper.")
        for paper in papers:
            paper['d2v_top_sentences'] = []
            paper['d2v_best_similarity'] = 0.0
        return

    # ── Prepare tagged documents for Doc2Vec ──
    tagged_docs = [
        TaggedDocument(words=sent.lower().split(), tags=[str(i)])
        for i, sent in enumerate(all_sentences)
    ]

    # ── Train Doc2Vec model ──
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=4,
        epochs=epochs,
    )
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

    # ── Infer claim vector ──
    claim_tokens = claim_text.lower().split()
    claim_vec = model.infer_vector(claim_tokens, epochs=50)

    # ── Compute cosine similarity for each sentence ──
    def cosine_sim(v1, v2):
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    sentence_scores = []
    for i, sent in enumerate(all_sentences):
        sent_vec = model.dv[str(i)]
        sim = cosine_sim(claim_vec, sent_vec)
        sentence_scores.append((i, sent, sim))

    # ── Assign scores back to papers ──
    paper_sentences = {i: [] for i in range(len(papers))}
    for sent_idx, sent_text, sim in sentence_scores:
        paper_idx = sentence_meta[sent_idx]
        paper_sentences[paper_idx].append((sent_text, sim))

    total_matched = 0
    for idx, paper in enumerate(papers):
        scored = sorted(paper_sentences[idx], key=lambda x: x[1], reverse=True)
        paper['d2v_top_sentences'] = scored[:top_sentences_per_paper]
        paper['d2v_best_similarity'] = scored[0][1] if scored else 0.0
        if scored:
            total_matched += 1

    print(f"  Doc2Vec corpus: {len(all_sentences)} sentences from {len(papers)} papers")
    print(f"  Model: vector_size={vector_size}, epochs={epochs}, window={window}")
    print(f"  {total_matched}/{len(papers)} papers have at least one matching sentence")


def fuse_sentence_scores(papers, tfidf_weight=0.5, d2v_weight=0.5):
    """Combine TF-IDF and Doc2Vec sentence scores into a unified ranking.

    For each paper, merges top sentences from both methods. Where a sentence
    appears in both, the scores are blended. Where it appears in only one,
    that score is used with half the missing method's weight.

    Updates each paper in place with:
        'top_sentences':   merged [(sentence, fused_score), ...]
        'best_similarity': best fused score
    """
    for paper in papers:
        tfidf_sents = {s: score for s, score in paper.get('top_sentences', [])}
        d2v_sents = {s: score for s, score in paper.get('d2v_top_sentences', [])}

        all_sents = set(tfidf_sents) | set(d2v_sents)
        fused = []
        for sent in all_sents:
            t_score = tfidf_sents.get(sent)
            d_score = d2v_sents.get(sent)

            if t_score is not None and d_score is not None:
                score = tfidf_weight * t_score + d2v_weight * d_score
            elif t_score is not None:
                score = tfidf_weight * t_score
            else:
                score = d2v_weight * d_score

            fused.append((sent, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        paper['top_sentences'] = fused[:5]
        paper['best_similarity'] = fused[0][1] if fused else 0.0


# ──────────────────────────────────────────────
# Abstract-Level TF-IDF Triage
# ──────────────────────────────────────────────
def tfidf_triage_abstracts(claim_text, papers, top_n=30):
    """Rank papers by TF-IDF cosine similarity of their abstract to the claim.

    Returns the top-N papers sorted by abstract relevance.
    Papers without an abstract are always kept (benefit of the doubt).

    Args:
        claim_text: The user's claim string.
        papers:     List of paper dicts (must have 'abstract').
        top_n:      Number of papers to keep.

    Returns:
        List of top-N paper dicts, sorted by abstract similarity (descending).
    """
    if len(papers) <= top_n:
        print(f"  {len(papers)} papers <= top_n={top_n}, skipping triage.")
        return papers

    # Separate papers with/without abstracts
    has_abstract = []
    no_abstract = []
    for p in papers:
        abstract = (p.get('abstract') or '').strip()
        if abstract and abstract != 'No abstract':
            has_abstract.append(p)
        else:
            no_abstract.append(p)

    if not has_abstract:
        return papers[:top_n]

    # TF-IDF: claim at index 0, abstracts follow
    abstracts = [p.get('abstract') or '' for p in has_abstract]
    corpus = [claim_text] + abstracts
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    claim_vec = tfidf_matrix[0:1]
    abstract_vecs = tfidf_matrix[1:]

    # Cosine similarity via dot product (TF-IDF vectors are already L2-normalised by sklearn)
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    scores = sklearn_cosine(claim_vec, abstract_vecs)[0]

    for paper, score in zip(has_abstract, scores):
        paper['abstract_triage_score'] = float(score)

    # Sort by score, keep top_n (minus slots reserved for no-abstract papers)
    has_abstract.sort(key=lambda p: p['abstract_triage_score'], reverse=True)
    slots_for_abstracts = max(0, top_n - len(no_abstract))
    kept = has_abstract[:slots_for_abstracts] + no_abstract

    print(f"  Scored {len(has_abstract)} abstracts via TF-IDF")
    if has_abstract:
        print(f"  Score range: {has_abstract[-1]['abstract_triage_score']:.3f} "
              f"– {has_abstract[0]['abstract_triage_score']:.3f}")
    dropped = len(papers) - len(kept)
    print(f"  Kept {len(kept)} papers, dropped {dropped}")

    return kept


# ──────────────────────────────────────────────
# Unified Search
# ──────────────────────────────────────────────
def search_all_databases(user_input, max_results_per_source=100):
    """
    Search arXiv and Semantic Scholar, then rank papers by claim similarity.

    All retrieved papers go through BoW similarity scoring and sentence-level
    analysis — no keyword triage gate.

    Args:
        user_input: Raw text input from user (e.g., "diabetes is harmful")
        max_results_per_source: Max papers to fetch from each database

    Returns:
        Dictionary with results from each source, combined results, and counterfactual
    """
    # Use top 3 keywords for database queries, top 5 for triage matching
    search_keywords, search_pos_tags = extract_keywords(user_input, top_n=3)
    triage_keywords, triage_pos_tags = extract_keywords(user_input, top_n=5)
    print(f"Search keywords: {search_keywords}")
    print(f"  POS tags: {search_pos_tags}")
    print(f"Triage keywords: {triage_keywords}")
    print(f"  POS tags: {triage_pos_tags}")

    if not search_keywords:
        print("No keywords found in input.")
        return {'arxiv': [], 'semantic_scholar': [],
                'combined': []}

    # Expand search keywords with WordNet synonyms
    expanded = expand_with_synonyms(search_keywords, max_synonyms=4)
    for kw, group in zip(search_keywords, expanded):
        syns = group - {kw}
        if syns:
            print(f"  {kw} -> synonyms: {syns}")
    print()

    arxiv_query = build_arxiv_query(expanded)
    semantic_query = build_semantic_scholar_query(expanded)

    print(f"arXiv query: {arxiv_query}")
    print(f"Semantic Scholar query: {semantic_query}\n")

    all_results = {
        'arxiv': [],
        'semantic_scholar': [],
        'combined': [],
    }

    # Search arXiv
    print("Searching arXiv...")
    try:
        arxiv_papers = search_arxiv(arxiv_query, max_results=max_results_per_source)
        all_results['arxiv'] = arxiv_papers
        print(f"  Found {len(arxiv_papers)} papers from arXiv")
    except Exception as e:
        print(f"  arXiv search failed: {e}")

    time.sleep(3)  # Respect arXiv rate limit (max 1 request per 3 seconds)

    # Search Semantic Scholar
    print("Searching Semantic Scholar...")
    try:
        ss_results = search_semantic_scholar(semantic_query, limit=max_results_per_source)
        ss_papers = []
        for paper in ss_results.get('data', []):
            ss_papers.append({
                'title': paper.get('title', 'No title'),
                'abstract': paper.get('abstract', 'No abstract'),
                'authors': [a.get('name', '') for a in paper.get('authors', [])],
                'pdf_url': (paper.get('openAccessPdf', {}) or {}).get('url') or paper.get('url'),
                'full_text': paper.get('abstract', ''),
                'source': 'semantic_scholar',
            })
        all_results['semantic_scholar'] = ss_papers
        print(f"  Found {len(ss_papers)} papers from Semantic Scholar")
    except Exception as e:
        print(f"  Semantic Scholar search failed: {e}")

    combined = all_results['arxiv'] + all_results['semantic_scholar']
    print(f"\nTotal retrieved: {len(combined)} papers")

    # ── Stage 1: Abstract-level TF-IDF triage ──
    print(f"\n--- Stage 1: TF-IDF triage on abstracts (top {TRIAGE_TOP_N}) ---")
    combined = tfidf_triage_abstracts(user_input, combined, top_n=TRIAGE_TOP_N)

    # ── Stage 2: Batched parallel PDF download ──
    print(f"\n--- Stage 2: Downloading PDFs (batched, parallel) ---")
    download_pdfs_batched(combined, batch_size=10, max_workers=5)

    # ── Stage 3a: TF-IDF + KNN sentence retrieval ──
    print(f"\n--- Stage 3a: TF-IDF + KNN sentence retrieval ---")
    tfidf_knn_rank_papers(user_input, combined, top_sentences_per_paper=5, knn_k=50)

    # ── Stage 3b: Doc2Vec sentence retrieval ──
    print(f"\n--- Stage 3b: Doc2Vec sentence retrieval ---")
    doc2vec_rank_papers(user_input, combined, top_sentences_per_paper=5)

    # ── Stage 3c: Fuse TF-IDF + Doc2Vec scores ──
    print(f"\n--- Stage 3c: Fusing TF-IDF + Doc2Vec scores (50/50) ---")
    fuse_sentence_scores(combined, tfidf_weight=0.5, d2v_weight=0.5)

    # ── Stage 4: Counterfactual agreement analysis ──
    print(f"\n--- Stage 4: Counterfactual agreement analysis ---")
    counterfactual = generate_counterfactual(user_input)
    print(f"  Original claim:      {user_input}")
    print(f"  Counterfactual:      {counterfactual}")
    counterfactual_agreement(user_input, counterfactual, combined)
    all_results['counterfactual'] = counterfactual

    # Sort papers by best sentence similarity (most relevant first)
    combined.sort(key=lambda p: p.get('best_similarity', 0), reverse=True)

    all_results['combined'] = combined

    print(f"\nPipeline complete: {len(combined)} papers ranked by similarity")

    return all_results


# ──────────────────────────────────────────────
# Claim-Based Search
# ──────────────────────────────────────────────
def search_by_claims(user_input, max_results_per_source=10):
    """
    Parse user input into individual claims and search each one independently.

    If the input contains multiple claims (separated by blank lines), each claim
    is searched with its own keywords. A paper only needs to match keywords from
    the claim it was found for (not across claims).

    Returns:
        List of dicts, one per claim, each with keys:
            'claim_text', 'keywords', 'results' (the search_all_databases output)
    """
    claims = parse_claims(user_input)
    print(f"Identified {len(claims)} claim(s)\n")

    all_claim_results = []
    for i, claim in enumerate(claims):
        claim_text = claim['text']
        print(f"{'='*80}")
        print(f"CLAIM {i+1}: {claim_text[:100]}...")
        print(f"{'='*80}")

        keywords, kw_pos_tags = extract_keywords(claim_text, top_n=5)
        print(f"Keywords: {keywords}")
        print(f"  POS tags: {kw_pos_tags}\n")

        if not keywords:
            print("No keywords found for this claim, skipping.\n")
            all_claim_results.append({
                'claim_text': claim_text,
                'keywords': [],
                'results': {'arxiv': [], 'semantic_scholar': [], 'combined': []},
            })
            continue

        results = search_all_databases(claim_text, max_results_per_source)
        all_claim_results.append({
            'claim_text': claim_text,
            'keywords': keywords,
            'results': results,
        })
        print()

    return all_claim_results


# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────
def display_claim_results(claim_results, max_display_per_claim=5):
    """Display search results grouped by claim, with counterfactual agreement scores."""
    total_papers = 0

    for i, claim_data in enumerate(claim_results, 1):
        claim_text = claim_data['claim_text']
        keywords = claim_data['keywords']
        results = claim_data['results']
        papers = results.get('combined', [])
        counterfactual = results.get('counterfactual', '')
        total_papers += len(papers)

        # Compute per-paper and aggregate claim-level scores
        verdict = compute_claim_verdict(claim_data)
        ct = verdict['claim_totals']

        print(f"\n{'='*80}")
        print(f"CLAIM {i}: {claim_text[:100]}...")
        if counterfactual:
            print(f"COUNTERFACTUAL: {counterfactual[:100]}...")
        print(f"Keywords: {keywords}")
        print(f"Papers found: {len(papers)}")
        print(f"{'='*80}")

        if not papers:
            print("  No matching papers found for this claim.")
            continue

        # ── Claim-level verdict ──
        print(f"\n  --- Claim Verdict ---")
        print(f"  Agree: {ct['agree']}  |  Disagree: {ct['disagree']}  |  "
              f"Neutral: {ct['neutral']}  |  Total: {ct['total']}")
        print(f"  Agreement rate:    {ct['agreement_rate']:.1%}")
        print(f"  Disagreement rate: {ct['disagreement_rate']:.1%}")
        print(f"  Validity score:    {ct['validity_score']:.1%}")
        print()

        for j, paper in enumerate(papers[:max_display_per_claim], 1):
            source_label = paper.get('source', 'unknown').upper()
            similarity = paper.get('best_similarity', 0.0)
            ns = paper.get('negation_scores', {})
            print(f"\n  [{j}] [{source_label}] {paper['title']}")
            print(f"      Similarity: {similarity:.3f}")
            if paper.get('authors'):
                authors_str = ', '.join(paper['authors'][:3])
                if len(paper['authors']) > 3:
                    authors_str += '...'
                print(f"      Authors: {authors_str}")

            # Show sentence-level counterfactual comparisons
            comparisons = paper.get('sentence_comparisons', [])
            if comparisons:
                print(f"      Sentence verdicts:")
                for k, comp in enumerate(comparisons[:3], 1):
                    v = comp['verdict'].upper()
                    so = comp['sim_original']
                    sc = comp['sim_counterfactual']
                    print(f"        {k}. [{v}] orig={so:.3f} counter={sc:.3f} "
                          f"| {comp['sentence'][:120]}...")

            # Per-paper agreement summary
            if ns and ns.get('total', 0) > 0:
                print(f"      Verdict: agree={ns['agree']} disagree={ns['disagree']} "
                      f"neutral={ns['neutral']} | "
                      f"agreement={ns['agreement_rate']:.0%} "
                      f"validity={ns['validity_score']:.0%}")

            if paper.get('pdf_url'):
                print(f"      Link: {paper['pdf_url']}")
            print(f"      {'-'*74}")

    # ── Final summary with per-claim verdicts ──
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total papers: {total_papers} across {len(claim_results)} claim(s)\n")

    for i, claim_data in enumerate(claim_results, 1):
        verdict = compute_claim_verdict(claim_data)
        ct = verdict['claim_totals']
        papers = claim_data['results'].get('combined', [])
        print(f"  Claim {i}: {claim_data['claim_text'][:80]}...")
        print(f"    Papers: {len(papers)}  |  Agree: {ct['agree']}  |  "
              f"Disagree: {ct['disagree']}  |  Neutral: {ct['neutral']}")
        print(f"    Agreement: {ct['agreement_rate']:.1%}  |  "
              f"Disagreement: {ct['disagreement_rate']:.1%}")
        print(f"    >>> Validity Score: {ct['validity_score']:.1%}")
        print()

    print(f"{'='*80}")

def compute_negation_scores(paper):
    """Compute agreement scores for a single paper using counterfactual comparison.

    Counts sentences where the verdict is 'agree', 'disagree', or 'neutral'
    based on TF-IDF similarity to the original claim vs the counterfactual.

    Returns a dict with:
        agree, disagree, neutral    (raw counts)
        total              – total sentences compared
        agreement_rate     – agree / total
        disagreement_rate  – disagree / total
        validity_score     – agreement_rate (simplified)
    """
    agree = 0
    disagree = 0
    neutral = 0

    for comp in paper.get('sentence_comparisons', []):
        verdict = comp.get('verdict', 'neutral')
        if verdict == 'agree':
            agree += 1
        elif verdict == 'disagree':
            disagree += 1
        else:
            neutral += 1

    total = agree + disagree + neutral
    agreement_rate = agree / total if total else 0.0
    disagreement_rate = disagree / total if total else 0.0

    return {
        'agree': agree,
        'disagree': disagree,
        'neutral': neutral,
        'total': total,
        'agreement_rate': agreement_rate,
        'disagreement_rate': disagreement_rate,
        'validity_score': agreement_rate,
    }


def compute_claim_verdict(claim_data):
    """Compute an aggregate verdict for a claim across all its papers.

    Pools the per-paper sentence verdicts (agree, disagree, neutral) and
    computes overall rates.

    Also attaches per-paper scores to each paper dict under 'negation_scores'.

    Returns a dict with:
        paper_scores  – list of (paper_title, scores_dict) tuples
        claim_totals  – pooled counts and rates
    """
    papers = claim_data['results'].get('combined', [])
    paper_scores = []
    pooled = {'agree': 0, 'disagree': 0, 'neutral': 0}

    for paper in papers:
        scores = compute_negation_scores(paper)
        paper['negation_scores'] = scores
        paper_scores.append((paper['title'], scores))
        for key in pooled:
            pooled[key] += scores[key]

    total = pooled['agree'] + pooled['disagree'] + pooled['neutral']

    claim_totals = dict(pooled)
    claim_totals['total'] = total
    claim_totals['agreement_rate'] = pooled['agree'] / total if total else 0.0
    claim_totals['disagreement_rate'] = pooled['disagree'] / total if total else 0.0
    claim_totals['validity_score'] = claim_totals['agreement_rate']

    return {
        'paper_scores': paper_scores,
        'claim_totals': claim_totals,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    user_input = input("Enter your search query: ").strip()
    if not user_input:
        user_input = 'Diabetes is harmful'
        print(f"Using default query: {user_input}")

    claim_results = search_by_claims(user_input, MAX_RESULTS_PER_SOURCE)
    display_claim_results(claim_results, max_display_per_claim=5)
