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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MAX_RESULTS_PER_SOURCE = 100
TRIAGE_TOP_N = 15  # Max papers to keep after abstract-level TF-IDF triage


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

    # Number + unit measurements (mirrors the patterns in build_bow)
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

    return combined[:top_n]


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
# Claim Word Retrieval & Stop-Word Proximity
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


def retrieve_all_claim_words(text):
    """Retrieve every word from the claim, stop words included.

    Returns a list of (word, position) tuples in original order.
    Nothing is filtered — all tokens are preserved so downstream
    proximity analysis can reason about the full context.
    """
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return [(word, i) for i, word in enumerate(tokens)]


def compute_stopword_proximity(claim_words, keywords, window=5, synonym_map=None):
    """Map each keyword to nearby stop words within ±window tokens.

    For every occurrence of a keyword (matched by stem, or by synonym if
    synonym_map is provided) in the claim, collects all stop words that
    fall within the surrounding window.

    Args:
        claim_words: List of (word, position) from retrieve_all_claim_words.
        keywords:    List of keyword strings (from extract_keywords).
        window:      Tokens before/after the keyword to scan (default 5).
        synonym_map: Optional dict mapping keyword -> set of synonyms.
                     When provided, a word also matches a keyword if its
                     stem matches any synonym's stem.

    Returns:
        Dict keyed by keyword, each value a list of dicts:
            {'stopword': str, 'distance': int, 'position': int}
        distance is signed: negative = before keyword, positive = after.
    """
    keyword_stems = {kw: simple_stem(kw) for kw in keywords}

    # Pre-compute synonym stems per keyword
    synonym_stems = {}
    if synonym_map:
        for kw, synonyms in synonym_map.items():
            synonym_stems[kw] = {simple_stem(syn) for syn in synonyms}

    # Locate every position of each keyword in the claim
    keyword_positions = {}
    for word, pos in claim_words:
        stem = simple_stem(word)
        for kw, kw_stem in keyword_stems.items():
            if stem == kw_stem or word == kw:
                keyword_positions.setdefault(kw, []).append(pos)
            elif kw in synonym_stems and stem in synonym_stems[kw]:
                keyword_positions.setdefault(kw, []).append(pos)

    # Collect stop words neighbouring each keyword occurrence
    proximity_map = {}
    for kw, positions in keyword_positions.items():
        neighbours = []
        for kw_pos in positions:
            for word, pos in claim_words:
                distance = pos - kw_pos
                if distance != 0 and abs(distance) <= window and word in STOPWORDS:
                    neighbours.append({
                        'stopword': word,
                        'distance': distance,
                        'position': pos,
                    })
        proximity_map[kw] = neighbours

    return proximity_map


NEGATORS = {'no', 'not', 'never', 'none', 'nor', 'neither', 'cannot', 'without'}


def compare_claim_to_sentences(proximity_map, sentences, keywords, window=5,
                               keyword_synonym_groups=None):
    """Compare stop-word context around keywords in claim vs paper sentences.

    For each top-matching sentence from a paper, tokenises it, builds its own
    stop-word proximity map, then checks — keyword by keyword — whether the
    claim and the sentence share the same negation context.

    Args:
        proximity_map: Claim-side output of compute_stopword_proximity.
        sentences:     List of (sentence_text, similarity_score) tuples
                       (from rank_sentences_by_claim).
        keywords:      List of keyword strings.
        window:        Token window for sentence-side proximity scan.
        keyword_synonym_groups: Optional list of sets (one per keyword) from
                       expand_with_synonyms.  When provided, a keyword is
                       considered present in a sentence if any of its synonyms
                       appear there.

    Returns:
        List of dicts, one per sentence::

            {'sentence':            str,
             'similarity':          float,
             'keyword_comparisons': [
                 {'keyword':            str,
                  'in_sentence':        bool,   # keyword found in sentence?
                  'claim_negated':      bool,
                  'sentence_negated':   bool,
                  'agreement':          bool,   # True when both sides match
                  'claim_stopwords':    [str],
                  'sentence_stopwords': [str]},
                 ...
             ]}
    """
    # Build synonym_map: keyword -> set of synonyms (for proximity matching)
    synonym_map = None
    if keyword_synonym_groups:
        synonym_map = {}
        for kw, group in zip(keywords, keyword_synonym_groups):
            synonym_map[kw] = group

    results = []
    for sent_text, sim_score in sentences:
        sent_words = retrieve_all_claim_words(sent_text)
        sent_proximity = compute_stopword_proximity(
            sent_words, keywords, window, synonym_map=synonym_map
        )
        sent_lower = sent_text.lower()

        comparisons = []
        for kw in keywords:
            claim_neighbours = proximity_map.get(kw, [])
            sent_neighbours = sent_proximity.get(kw, [])

            claim_negated = any(n['stopword'] in NEGATORS for n in claim_neighbours)
            sent_negated = any(n['stopword'] in NEGATORS for n in sent_neighbours)

            # Check if keyword or any of its synonyms appear in the sentence
            in_sent = len(sent_neighbours) > 0 or kw in sent_lower
            if not in_sent and synonym_map and kw in synonym_map:
                for syn in synonym_map[kw]:
                    if syn in sent_lower:
                        in_sent = True
                        break

            comparisons.append({
                'keyword': kw,
                'in_sentence': in_sent,
                'claim_negated': claim_negated,
                'sentence_negated': sent_negated,
                'agreement': claim_negated == sent_negated,
                'claim_stopwords': [n['stopword'] for n in claim_neighbours],
                'sentence_stopwords': [n['stopword'] for n in sent_neighbours],
            })

        results.append({
            'sentence': sent_text,
            'similarity': sim_score,
            'keyword_comparisons': comparisons,
        })
    return results


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
    response = requests.get(url, params=params)
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
    response = requests.get(url, params=params)
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
        text = extract_text(BytesIO(resp.content))
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
    candidates = [p for p in papers
                  if p.get('pdf_url') and p.get('full_text', '') == p.get('abstract', '')]

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
# Stemming (used by negation proximity analysis)
# ──────────────────────────────────────────────
def simple_stem(word):
    """Strip common suffixes to get a rough word stem."""
    w = word.lower()
    for suffix in ['ing', 'tion', 'sion', 'ment', 'ness', 'ies', 'ied', 'ous',
                    'ive', 'able', 'ible', 'ful', 'less', 'ally', 'ical',
                    'ed', 'er', 'es', 'ly', 'al', 's']:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def extract_sentences(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ──────────────────────────────────────────────
# TF-IDF + KNN Sentence Retrieval
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
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=2,
        max_features=30000,
    )
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
        Dictionary with results from each source, combined results, and claim BoW
    """
    # Use top 3 keywords for database queries, top 5 for triage matching
    search_keywords = extract_keywords(user_input, top_n=3)
    triage_keywords = extract_keywords(user_input, top_n=5)
    print(f"Search keywords: {search_keywords}")
    print(f"Triage keywords: {triage_keywords}")

    if not search_keywords:
        print("No keywords found in input.")
        return {'arxiv': [], 'semantic_scholar': [],
                'combined': [], 'claim_bow': Counter()}

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
        'claim_words': [],
        'proximity_map': {},
        'claim_bow': Counter(),
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
    download_pdfs_batched(combined, batch_size=8, max_workers=8)

    # ── Stage 3: TF-IDF + KNN sentence retrieval ──
    print(f"\n--- Stage 3: TF-IDF + KNN sentence retrieval ---")
    tfidf_knn_rank_papers(user_input, combined, top_sentences_per_paper=5, knn_k=50)

    # ── Stage 3.5: Compute TF-IDF importance weights for keywords ──
    print(f"\n--- Computing keyword importance weights ---")
    kw_abstracts = [(p.get('abstract') or '') for p in combined]
    kw_corpus = [user_input] + kw_abstracts
    kw_vectorizer = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 2),
        max_df=0.85, min_df=1, max_features=30000,
    )
    kw_matrix = kw_vectorizer.fit_transform(kw_corpus)
    kw_vocab = kw_vectorizer.vocabulary_
    claim_tfidf_vec = kw_matrix[0]

    keyword_weights = {}
    for kw in triage_keywords:
        if kw in kw_vocab:
            keyword_weights[kw] = float(claim_tfidf_vec[0, kw_vocab[kw]])
        else:
            keyword_weights[kw] = 0.0

    # Normalize: scale to [0.1, 1.0] so no keyword is completely ignored
    max_w = max(keyword_weights.values()) if keyword_weights.values() else 1.0
    if max_w > 0:
        keyword_weights = {kw: max(w / max_w, 0.1) for kw, w in keyword_weights.items()}
    else:
        keyword_weights = {kw: 1.0 for kw in triage_keywords}

    for kw, w in keyword_weights.items():
        print(f"  {kw}: weight = {w:.3f}")

    all_results['keyword_weights'] = keyword_weights

    # Attach weights to each paper for downstream scoring
    for paper in combined:
        paper['keyword_weights'] = keyword_weights

    # ── Stage 4: Claim word retrieval & negation analysis ──
    print(f"\n--- Stage 4: Claim word retrieval & negation context ---")
    triage_expanded = expand_with_synonyms(triage_keywords, max_synonyms=4)
    claim_words = retrieve_all_claim_words(user_input)
    proximity_map = compute_stopword_proximity(claim_words, triage_keywords)
    all_results['claim_words'] = claim_words
    all_results['proximity_map'] = proximity_map
    all_results['triage_expanded'] = triage_expanded
    print(f"  Claim tokens: {len(claim_words)} words")
    for kw, neighbours in proximity_map.items():
        if neighbours:
            nearby = ', '.join(f"{n['stopword']}({n['distance']:+d})" for n in neighbours)
            print(f"    [{kw}] nearby stop words: {nearby}")

    for paper in combined:
        top_sentences = paper.get('top_sentences', [])
        if not top_sentences:
            paper['sentence_comparisons'] = []
            continue
        comparisons = compare_claim_to_sentences(
            proximity_map, top_sentences, triage_keywords,
            keyword_synonym_groups=triage_expanded
        )
        paper['sentence_comparisons'] = comparisons

        for comp in comparisons:
            for kc in comp['keyword_comparisons']:
                if kc['in_sentence'] and not kc['agreement']:
                    print(f"  {paper['title'][:50]}...")
                    print(f"    [DISAGREE] keyword '{kc['keyword']}': "
                          f"claim_negated={kc['claim_negated']}, "
                          f"sentence_negated={kc['sentence_negated']}")
                    print(f"    sentence: {comp['sentence'][:120]}...")

    # Sort papers by best sentence similarity (most relevant first)
    combined.sort(key=lambda p: p.get('best_similarity', 0), reverse=True)

    all_results['combined'] = combined

    print(f"\nPipeline complete: {len(combined)} papers ranked by TF-IDF + KNN similarity")

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

        keywords = extract_keywords(claim_text, top_n=5)
        print(f"Keywords: {keywords}\n")

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
    """Display search results grouped by claim, with aggregate negation scores."""
    total_papers = 0

    for i, claim_data in enumerate(claim_results, 1):
        claim_text = claim_data['claim_text']
        keywords = claim_data['keywords']
        papers = claim_data['results'].get('combined', [])
        total_papers += len(papers)

        # Compute per-paper and aggregate claim-level negation scores
        verdict = compute_claim_verdict(claim_data)
        ct = verdict['claim_totals']

        print(f"\n{'='*80}")
        print(f"CLAIM {i}: {claim_text[:100]}...")
        print(f"Keywords: {keywords}")
        print(f"Papers found: {len(papers)}")
        print(f"{'='*80}")

        if not papers:
            print("  No matching papers found for this claim.")
            continue

        # ── Claim-level verdict ──
        print(f"\n  --- Claim Verdict ---")
        print(f"  Agree: {ct['agree']}  |  Disagree: {ct['disagree']}  |  "
              f"Missing: {ct['missing']}  |  Total found: {ct['total_found']}")
        print(f"  Agreement rate:    {ct['agreement_rate']:.1%}")
        print(f"  Disagreement rate: {ct['disagreement_rate']:.1%}")
        print(f"  Coverage rate:     {ct['coverage_rate']:.1%}")
        print(f"  Consistency rate:  {ct['consistency_rate']:.1%}")
        print(f"  Validity score:    {ct['validity_score']:.1%}")
        print()

        for j, paper in enumerate(papers[:max_display_per_claim], 1):
            source_label = paper.get('source', 'unknown').upper()
            similarity = paper.get('best_similarity', 0.0)
            ns = paper.get('negation_scores', {})
            print(f"\n  [{j}] [{source_label}] {paper['title']}")
            print(f"      TF-IDF similarity: {similarity:.3f}")
            if paper.get('authors'):
                authors_str = ', '.join(paper['authors'][:3])
                if len(paper['authors']) > 3:
                    authors_str += '...'
                print(f"      Authors: {authors_str}")
            top_sentences = paper.get('top_sentences', [])
            if top_sentences:
                print(f"      Top matching sentences:")
                for k, (sent, score) in enumerate(top_sentences[:3], 1):
                    print(f"        {k}. [{score:.3f}] {sent[:150]}...")

            # Per-paper negation scores
            if ns and ns.get('total_comparisons', 0) > 0:
                print(f"      Negation: agree={ns['agree']} disagree={ns['disagree']} "
                      f"missing={ns['missing']} | "
                      f"agreement={ns['agreement_rate']:.0%} "
                      f"coverage={ns['coverage_rate']:.0%} "
                      f"consistency={ns['consistency_rate']:.0%} "
                      f"validity={ns['validity_score']:.0%}")
                if ns['disagree'] > 0:
                    # List which keywords disagree
                    disag_kws = set()
                    for comp in paper.get('sentence_comparisons', []):
                        for kc in comp['keyword_comparisons']:
                            if kc['in_sentence'] and not kc['agreement']:
                                disag_kws.add(kc['keyword'])
                    print(f"      ** NEGATION MISMATCH on: {', '.join(disag_kws)} **")
            elif paper.get('sentence_comparisons'):
                print(f"      Negation context: no keywords found in sentences")

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
              f"Disagree: {ct['disagree']}  |  Missing: {ct['missing']}")
        print(f"    Agreement: {ct['agreement_rate']:.1%}  |  "
              f"Coverage: {ct['coverage_rate']:.1%}  |  "
              f"Consistency: {ct['consistency_rate']:.1%}")
        print(f"    >>> Validity Score: {ct['validity_score']:.1%}")
        print()

    print(f"{'='*80}")

def compute_negation_scores(paper):
    """Compute aggregate negation scores for a single paper.

    Examines every keyword comparison across all sentence comparisons and
    classifies each into one of three buckets:

        agree    – keyword found in sentence, negation status matches
        disagree – keyword found in sentence, negation status differs
        missing  – keyword not found in the sentence at all

    Also computes a per-keyword consistency rate: for each keyword that
    appears in multiple sentences, how consistently do those sentences
    agree or disagree?  A keyword that agrees in 3 sentences and disagrees
    in 1 has consistency 3/4 = 0.75.  The paper-level consistency_rate is
    the average across all keywords found.

    Returns a dict with:
        agree, disagree, missing          (raw counts)
        total_found       – keyword×sentence pairs where keyword was found
        total_comparisons – all keyword×sentence pairs
        keywords_covered  – number of unique keywords found in >= 1 sentence
        keywords_total    – total unique keywords checked
        agreement_rate    – agree / total_found
        disagreement_rate – disagree / total_found
        coverage_rate     – keywords_covered / keywords_total (per-keyword)
        consistency_rate  – avg per-keyword consistency (0-1)
        validity_score    – agreement_rate * coverage_rate * consistency_rate
    """
    agree = 0
    disagree = 0
    missing = 0

    # Track per-keyword agree/disagree counts across sentences
    keyword_agree = {}    # keyword -> count of agreeing comparisons
    keyword_disagree = {} # keyword -> count of disagreeing comparisons
    all_checked_keywords = set()  # every keyword that was compared

    for comp in paper.get('sentence_comparisons', []):
        for kc in comp['keyword_comparisons']:
            kw = kc['keyword']
            all_checked_keywords.add(kw)
            if not kc['in_sentence']:
                missing += 1
            elif kc['agreement']:
                agree += 1
                keyword_agree[kw] = keyword_agree.get(kw, 0) + 1
            else:
                disagree += 1
                keyword_disagree[kw] = keyword_disagree.get(kw, 0) + 1

    total_found = agree + disagree
    total_comparisons = total_found + missing

    # Per-keyword coverage, weighted by TF-IDF importance
    keywords_found = set(keyword_agree) | set(keyword_disagree)
    keywords_covered = len(keywords_found)
    keywords_total = len(all_checked_keywords)
    kw_weights = paper.get('keyword_weights', {})

    agreement_rate = agree / total_found if total_found else 0.0
    disagreement_rate = disagree / total_found if total_found else 0.0

    if kw_weights and all_checked_keywords:
        # Weighted coverage: sum of weights for found keywords / sum of weights for all keywords
        found_weight = sum(kw_weights.get(kw, 1.0) for kw in keywords_found)
        total_weight = sum(kw_weights.get(kw, 1.0) for kw in all_checked_keywords)
        coverage_rate = found_weight / total_weight if total_weight else 0.0
    else:
        # Fallback to binary coverage
        coverage_rate = keywords_covered / keywords_total if keywords_total else 0.0

    # Per-keyword consistency: max(agree, disagree) / total for that keyword
    all_keywords = set(keyword_agree) | set(keyword_disagree)
    if all_keywords:
        consistencies = []
        for kw in all_keywords:
            kw_agree = keyword_agree.get(kw, 0)
            kw_disagree = keyword_disagree.get(kw, 0)
            kw_total = kw_agree + kw_disagree
            consistencies.append(max(kw_agree, kw_disagree) / kw_total)
        consistency_rate = sum(consistencies) / len(consistencies)
    else:
        consistency_rate = 0.0

    validity_score = agreement_rate * coverage_rate * consistency_rate

    return {
        'agree': agree,
        'disagree': disagree,
        'missing': missing,
        'total_found': total_found,
        'total_comparisons': total_comparisons,
        'keywords_covered': keywords_covered,
        'keywords_total': keywords_total,
        'agreement_rate': agreement_rate,
        'disagreement_rate': disagreement_rate,
        'coverage_rate': coverage_rate,
        'consistency_rate': consistency_rate,
        'validity_score': validity_score,
    }


def compute_claim_verdict(claim_data):
    """Compute an aggregate verdict for a claim across all its papers.

    Pools the per-paper negation buckets (agree, disagree, missing) and
    produces the same rate metrics as compute_negation_scores, but across
    every paper for the claim.

    Also attaches per-paper scores to each paper dict under 'negation_scores'.

    Returns a dict with:
        paper_scores  – list of (paper_title, scores_dict) tuples
        claim_totals  – pooled counts and rates (same keys as per-paper)
    """
    papers = claim_data['results'].get('combined', [])
    paper_scores = []
    pooled = {'agree': 0, 'disagree': 0, 'missing': 0}

    for paper in papers:
        scores = compute_negation_scores(paper)
        paper['negation_scores'] = scores
        paper_scores.append((paper['title'], scores))
        for key in pooled:
            pooled[key] += scores[key]

    total_found = pooled['agree'] + pooled['disagree']
    total_comparisons = total_found + pooled['missing']

    claim_totals = dict(pooled)
    claim_totals['total_found'] = total_found
    claim_totals['total_comparisons'] = total_comparisons
    claim_totals['agreement_rate'] = (
        pooled['agree'] / total_found if total_found else 0.0
    )
    claim_totals['disagreement_rate'] = pooled['disagree'] / total_found if total_found else 0.0

    # Claim-level coverage: average of per-paper keyword coverage rates
    paper_coverages = [s['coverage_rate'] for _, s in paper_scores if s['keywords_total'] > 0]
    claim_totals['coverage_rate'] = (
        sum(paper_coverages) / len(paper_coverages) if paper_coverages else 0.0
    )

    # Claim-level consistency: average of per-paper consistency rates (papers with scores only)
    paper_consistencies = [s['consistency_rate'] for _, s in paper_scores if s['total_found'] > 0]
    claim_totals['consistency_rate'] = (
        sum(paper_consistencies) / len(paper_consistencies) if paper_consistencies else 0.0
    )
    claim_totals['validity_score'] = (
        claim_totals['agreement_rate'] * claim_totals['coverage_rate'] * claim_totals['consistency_rate']
    )

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
