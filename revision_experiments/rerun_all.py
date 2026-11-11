#!/usr/bin/env python3
"""
Re-run of all experiments for the SN Computer Science revision
("Extractive Text Summarization Using Artificial Intelligence: Formality as a
Catalyst in Low-Resource Language Processing").

WHY THIS SCRIPT EXISTS
----------------------
The original notebooks called `rouge_n_summary_level(tokens, ref_tokens, n)`
with a *flat list of word strings*.  The `rouge` (easy-rouge) package flattens
its input once more, so every word was split into its characters: all
previously reported ROUGE values were CHARACTER n-gram F1 scores, not word
n-gram scores.  This script recomputes everything with WORD-level ROUGE
(precision, recall, F1) under one common pipeline for every method.

WHAT IT PRODUCES (in --out)
---------------------------
  summaries/<dataset>/<method>/<ratio>/<category>/<file>   system summaries
  scores_<dataset>.csv            per-document P/R/F of ROUGE-1/2/L and BLEU
  stats_<dataset>.csv             formality vs every baseline: paired t, Wilcoxon, CI, Holm
  tables/*.tex                    LaTeX tables, ready to paste into sn-article.tex
  macros.tex                      \def lines with the headline numbers used in the paper
  report.txt                      human-readable log of every number

USAGE (see README.md for the full recipe)
-----------------------------------------------
  python rerun_all.py \
      --corpus  "path/to/Gujarati-Textsummarization-corpus/gujarati Text summarization dataset" \
      --gu-pos  "path/to/gu.pos" \
      --stopwords "path/to/Gujarati-Stop-words/stop-words.txt" \
      --legal-pos "path/to/Court-cases-POS-tagging" \
      --informal-pos "path/to/informal_text_pos_tagging/code.pos" \
      --bbc "path/to/BBC News Summary" \
      --out results

Pure-python parts (tokeniser, ROUGE, BLEU, TF-IDF, TextRank, Lead, kappa) have
no dependencies.  Optional dependencies are imported only when needed:
  nltk           formality (POS tagger), English sentence split / POS tags
  indic-nlp-library   Gujarati sentence splitter (falls back to a regex)
  scikit-learn   LSA, LDA
  sentence-transformers  LaBSE neural baselines
  scikit-fuzzy   fuzzy-logic baseline (imports matplotlib)
  scipy          statistical tests
"""
import argparse, csv, json, math, os, random, re, statistics as st, string, sys, unicodedata
from collections import Counter, defaultdict

RATIOS = (20, 30, 40, 50)
random.seed(0)

# --------------------------------------------------------------------------
# Tokenisation (shared by every method AND by the evaluation)
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"[^\s" + re.escape(string.punctuation) + r"।॥‘’“”–—…]+")


def tokenize(text, lower=False):
    toks = _TOKEN.findall(text)
    return [t.lower() for t in toks] if lower else toks


_SENT_FALLBACK = re.compile(r"(?<=[.?!।|])\s+")


def split_gu(text):
    text = " ".join(text.split())
    try:
        from indicnlp.tokenize import sentence_tokenize
        sents = sentence_tokenize.sentence_split(text, lang="gu")
    except ImportError:
        sents = _SENT_FALLBACK.split(text)
    return [s.strip() for s in sents if tokenize(s)]


def split_en(text):
    try:
        import nltk
        sents = nltk.sent_tokenize(" ".join(text.split()))
    except Exception:
        sents = _SENT_FALLBACK.split(" ".join(text.split()))
    return [s.strip() for s in sents if tokenize(s)]


def split_en_reference(text):
    # BBC reference summaries are sentences glued together without spaces.
    text = re.sub(r"(?<=[.!?])(?=[A-Z\"'])", " ", text.replace("\n", " "))
    return split_en(text)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
class Doc:
    __slots__ = ("dataset", "category", "name", "title", "sents", "refs")

    def __init__(self, dataset, category, name, title, sents, refs):
        self.dataset, self.category, self.name = dataset, category, name
        self.title, self.sents, self.refs = title, sents, refs  # refs: {annotator: [sentences]}


def _read(path):
    for enc in ("utf-8", "utf-16"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeError:
            continue
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def load_gujarati(corpus):
    docs = []
    for cat in sorted(os.listdir(corpus)):
        cdir = os.path.join(corpus, cat)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted((f for f in os.listdir(cdir) if f.endswith(".txt")), key=lambda x: int(re.sub(r"\D", "", x) or 0)):
            raw = _read(os.path.join(cdir, fn))
            m = re.search(r"<text>(.*?)</text>", raw, re.S)
            t = re.search(r"<title>(.*?)</title>", raw, re.S)
            sents = split_gu(m.group(1) if m else raw)
            refs = {}
            for a in ("1", "2"):
                p = os.path.join(cdir, "annotator" + a, "a%s_%s" % (a, fn))
                if os.path.exists(p):
                    refs[a] = [l.strip() for l in _read(p).splitlines() if tokenize(l)]
            if sents and refs:
                docs.append(Doc("gujarati", cat, fn, t.group(1).strip() if t else "", sents, refs))
    return docs


def load_bbc(root):
    art_root = os.path.join(root, "News Articles")
    sum_root = os.path.join(root, "Summaries")
    if not os.path.isdir(art_root):  # Kaggle zip sometimes nests one more level
        for sub in os.listdir(root):
            if os.path.isdir(os.path.join(root, sub, "News Articles")):
                art_root, sum_root = os.path.join(root, sub, "News Articles"), os.path.join(root, sub, "Summaries")
    docs = []
    for cat in sorted(os.listdir(art_root)):
        for fn in sorted(os.listdir(os.path.join(art_root, cat))):
            if not fn.endswith(".txt"):
                continue
            lines = _read(os.path.join(art_root, cat, fn)).split("\n", 1)
            sp = os.path.join(sum_root, cat, fn)
            if not os.path.exists(sp) or len(lines) < 2:
                continue
            sents = split_en(lines[1])
            ref = split_en_reference(_read(sp))
            if sents and ref:
                docs.append(Doc("english", cat, fn, lines[0].strip(), sents, {"1": ref}))
    return docs


def load_bbc_csv(path):
    """BBC News Summary as the single CSV mirrored at huggingface.co/datasets/gopalkalpande/bbc-news-summary
    (columns File_path=category, Articles='Title..Body', Summaries)."""
    csv.field_size_limit(10 ** 8)
    docs, counter = [], Counter()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row["File_path"].strip()
            counter[cat] += 1
            title, _, body = row["Articles"].partition("..")
            sents = split_en(body)
            ref = split_en_reference(row["Summaries"])
            if sents and ref:
                docs.append(Doc("english", cat, "%03d.txt" % counter[cat], title.strip(), sents, {"1": ref}))
    return docs


def load_stopwords(path, english=False):
    if english:
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except Exception:
            return set()
    if not path:
        return set()
    return set(_read(path).split())


# --------------------------------------------------------------------------
# Evaluation: word-level ROUGE-1/2/L (P, R, F1) and BLEU-4
# --------------------------------------------------------------------------
def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _prf(hits, n_ref, n_sys):
    r = hits / n_ref if n_ref else 0.0
    p = hits / n_sys if n_sys else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def rouge_n(sys_tokens, ref_tokens, n):
    s, r = _ngrams(sys_tokens, n), _ngrams(ref_tokens, n)
    return _prf(sum((s & r).values()), sum(r.values()), sum(s.values()))


def _lcs_positions(x, y):
    """Indices in y of one LCS of x and y."""
    m, n = len(x), len(y)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            table[i + 1][j + 1] = table[i][j] + 1 if x[i] == y[j] else max(table[i][j + 1], table[i + 1][j])
    i, j, pos = m, n, []
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            pos.append(j - 1); i -= 1; j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return pos


def rouge_l_summary(sys_sents, ref_sents):
    """Summary-level ROUGE-L with union LCS (Lin, 2004, Sec. 3.2)."""
    sys_uni = Counter(t for s in sys_sents for t in s)
    ref_uni = Counter(t for s in ref_sents for t in s)
    hits = 0
    for ref in ref_sents:
        union = set()
        for s in sys_sents:
            union.update(_lcs_positions(s, ref))
        for k in sorted(union):
            w = ref[k]
            if sys_uni[w] > 0 and ref_uni[w] > 0:
                sys_uni[w] -= 1; ref_uni[w] -= 1; hits += 1
    return _prf(hits, sum(len(s) for s in ref_sents), sum(len(s) for s in sys_sents))


def bleu4(sys_tokens, ref_tokens, eps=0.1):
    """Sentence-level BLEU-4, uniform weights, brevity penalty, Chen & Cherry (2014) smoothing method 1."""
    if not sys_tokens:
        return 0.0
    logs = 0.0
    for n in range(1, 5):
        s, r = _ngrams(sys_tokens, n), _ngrams(ref_tokens, n)
        total = sum(s.values())
        if total == 0:
            return 0.0
        match = sum((s & r).values())
        logs += math.log((match if match else eps) / total) / 4
    c, rl = len(sys_tokens), len(ref_tokens)
    bp = 1.0 if c > rl else math.exp(1 - rl / c)
    return bp * math.exp(logs)


def evaluate(sys_sents, ref_sents, lower):
    S = [tokenize(s, lower) for s in sys_sents]
    R = [tokenize(s, lower) for s in ref_sents]
    sf, rf = [t for s in S for t in s], [t for s in R for t in s]
    out = {}
    for name, (p, r, f) in (("R1", rouge_n(sf, rf, 1)), ("R2", rouge_n(sf, rf, 2)), ("RL", rouge_l_summary(S, R))):
        out[name + "_P"], out[name + "_R"], out[name + "_F"] = p, r, f
    out["BLEU"] = bleu4(sf, rf)
    return out


# --------------------------------------------------------------------------
# Sentence selection helper
# --------------------------------------------------------------------------
def select(scores, ratio):
    """Top-k sentence indices (k = round(n*ratio/100), >= 1), returned in document order."""
    n = len(scores)
    k = max(1, min(n, int(round(n * ratio / 100.0))))
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    return sorted(order[:k])


def content_tokens(sent, stop, lower):
    return [t for t in tokenize(sent, lower) if t not in stop]


# --------------------------------------------------------------------------
# Baselines (all share the same tokeniser and stop-word list -> fair comparison)
# --------------------------------------------------------------------------
def score_lead(doc, ctx):
    n = len(doc.sents)
    return [n - i for i in range(n)]


def score_tfidf(doc, ctx):
    toks = [content_tokens(s, ctx["stop"], ctx["lower"]) for s in doc.sents]
    n = len(toks)
    df = Counter(t for s in toks for t in set(s))
    tf = Counter(t for s in toks for t in s)
    total = sum(tf.values()) or 1
    w = {t: (tf[t] / total) * math.log(1 + n / df[t]) for t in tf}
    return [sum(w[t] for t in set(s)) for s in toks]


def _pagerank(sim, d=0.85, iters=200, tol=1e-8):
    n = len(sim)
    out = [sum(row) for row in sim]
    pr = [1.0 / n] * n
    for _ in range(iters):
        new = [(1 - d) / n + d * sum(pr[j] * sim[j][i] / out[j] for j in range(n) if out[j] > 0) for i in range(n)]
        if sum(abs(a - b) for a, b in zip(new, pr)) < tol:
            return new
        pr = new
    return pr


def score_textrank(doc, ctx):
    """TextRank with the original sentence-overlap similarity of Mihalcea & Tarau (2004)."""
    toks = [set(content_tokens(s, ctx["stop"], ctx["lower"])) for s in doc.sents]
    n = len(toks)
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and toks[i] and toks[j]:
                denom = math.log(len(toks[i]) + 1) + math.log(len(toks[j]) + 1)
                sim[i][j] = len(toks[i] & toks[j]) / denom if denom > 0 else 0.0
    return _pagerank(sim)


def _topic_scores(doc, ctx, model):
    """LSA / LDA scoring as in the original notebooks: 2 topics, values below the topic mean set to 0, summed."""
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    texts = [" ".join(content_tokens(s, ctx["stop"], ctx["lower"])) for s in doc.sents]
    if ctx.get("default_token_pattern"):
        vec = CountVectorizer()  # sklearn default: (?u)\b\w\w+\b -- breaks words at Gujarati vowel signs (AEM)
    else:
        vec = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
    try:
        X = vec.fit_transform(texts)
    except ValueError:  # empty vocabulary
        return score_lead(doc, ctx)
    if X.shape[0] < 3 or X.shape[1] < 3:
        return score_lead(doc, ctx)
    if model == "lsa":
        from sklearn.decomposition import TruncatedSVD
        T = TruncatedSVD(n_components=2, random_state=0).fit_transform(X)
    else:
        from sklearn.decomposition import LatentDirichletAllocation
        T = LatentDirichletAllocation(n_components=2, random_state=0).fit_transform(X)
    T = np.where(T < T.mean(axis=0), 0, T)
    return list(T.sum(axis=1))


def score_lsa(doc, ctx):
    return _topic_scores(doc, ctx, "lsa")


def score_lda(doc, ctx):
    return _topic_scores(doc, ctx, "lda")


_LABSE = {}


_EMB_CACHE = {}


def _embed(sents, model_name):
    key = (model_name, tuple(sents))
    if key not in _EMB_CACHE:
        if model_name not in _LABSE:
            from sentence_transformers import SentenceTransformer
            _LABSE[model_name] = SentenceTransformer(model_name)
        _EMB_CACHE[key] = _LABSE[model_name].encode(sents, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return _EMB_CACHE[key]


def score_labse_centroid(doc, ctx):
    """Unsupervised neural baseline: cosine similarity to the document centroid (Radev et al., 2004) in LaBSE space."""
    import numpy as np
    E = _embed(doc.sents, ctx["labse"])
    c = E.mean(axis=0)
    c = c / (np.linalg.norm(c) or 1.0)
    return list(E @ c)


def score_labse_textrank(doc, ctx):
    """Unsupervised neural baseline: PageRank over a LaBSE cosine-similarity graph."""
    import numpy as np
    E = _embed(doc.sents, ctx["labse"])
    S = np.clip(E @ E.T, 0, None)
    np.fill_diagonal(S, 0)
    return _pagerank(S.tolist())


# --------------------------------------------------------------------------
# Fuzzy-logic baseline (port of the public notebook "textsummarizationwith_fuzzy")
# --------------------------------------------------------------------------
# Mamdani system with six inputs, each fuzzified by scikit-fuzzy's automf(3) ("poor", "average", "good"),
# three rules (all-poor -> low, all-average -> medium, all-good -> high), built with skfuzzy.control exactly as in
# the notebook.  If no rule fires (skfuzzy raises an error; this happens e.g. for sentences without any noun),
# the sentence gets the lowest rank 0.
# Corrections w.r.t. the notebook: (1) its "TF-IDF" input was computed over whole sentences instead of words and
# trimmed the first/last character of every sentence, so it was ~1 for every sentence; here it is the sentence's
# summed word TF-IDF, normalised by the document maximum; (2) noun/pronoun tags come from the corrected tagger.
_FIS = None


def fuzzy_rank(x):
    """x: dict of the six crisp inputs -> defuzzified rank (skfuzzy centroid), or 0.0 if no rule fires."""
    global _FIS
    if _FIS is None:
        import numpy as np
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl
        A = {"similarity": np.arange(0, 1.25, .1), "senlength": np.arange(0, 1.25, .1), "senpos": np.arange(0, 2, .1),
             "nounplace": np.arange(0, 1.2, .1), "pronounplace": np.arange(0, 1.2, .1), "tfidfscore": np.arange(0, 1.25, .1)}
        ants = {k: ctrl.Antecedent(u, k) for k, u in A.items()}
        for a in ants.values():
            a.automf(3)
        rank = ctrl.Consequent(np.arange(0, 24, 1), "rank")
        rank["low"] = fuzz.trimf(rank.universe, [0, 0, 10])
        rank["medium"] = fuzz.trimf(rank.universe, [0, 10, 24])
        rank["high"] = fuzz.trimf(rank.universe, [10, 24, 24])
        rules = []
        for lvl, outl in (("poor", "low"), ("average", "medium"), ("good", "high")):
            ante = None
            for k in A:
                ante = ants[k][lvl] if ante is None else ante & ants[k][lvl]
            rules.append(ctrl.Rule(ante, rank[outl]))
        _FIS = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
    for k, v in x.items():
        _FIS.input[k] = v
    try:
        _FIS.compute()
        return float(_FIS.output["rank"])
    except (ValueError, KeyError, AssertionError):
        return 0.0


def score_fuzzy(doc, ctx):
    from difflib import SequenceMatcher
    import nltk
    sents = [s for s in doc.sents]
    plain = ["".join(c for c in s if c not in string.punctuation) for s in sents]
    longest = max(len(s) for s in sents) or 1
    tfidf = score_tfidf(doc, ctx)
    tmax = max(tfidf) or 1.0
    n = len(sents)
    scores = []
    for i, t in enumerate(plain):
        nostop = " ".join(w for w in t.split() if (w.lower() if ctx["lower"] else w) not in ctx["stop"])
        others = [SequenceMatcher(None, nostop, plain[j]).ratio() for j in range(n) if j != i]
        sim = st.mean(others) if others else 0.0
        pos = 0.1 if i == 0 else (1.0 if i == n - 1 else i / n)
        if doc.dataset == "gujarati":
            tags = [tg for _, tg in ctx["gu_tagger"].tag(tokenize(sents[i]) or ["hi"])]
            noun = any(tg == "NN" or tg.startswith(("N_", "N-")) for tg in tags)
            pron = any(tg.startswith("PR_") for tg in tags)
        else:
            tags = [tg for _, tg in nltk.pos_tag(tokenize(sents[i]) or ["hi"])]
            noun = any(tg.startswith("NN") for tg in tags)
            pron = any(tg in ("PRP", "PRP$", "WP", "WP$") for tg in tags)
        scores.append(fuzzy_rank(dict(similarity=sim, senlength=len(sents[i]) / longest, senpos=pos,
                                      nounplace=1.0 if noun else 0.0, pronounplace=(1.0 if pron else 0.0) + 0.1,
                                      tfidfscore=tfidf[i] / tmax)))
    return scores


# --------------------------------------------------------------------------
# Formality method (faithful to the public notebook, with documented switches)
# --------------------------------------------------------------------------
# Word-class sides of Eq. (2) / Table 2, decided by BIS tag prefix so that every sub-tag present in gu.pos
# (e.g. V_VAUX_VNP, DM_DMI, CC_CCS_UT) is covered.  Proper nouns are written "N-NNP" in gu.pos and "N_NNP"
# in the manually tagged validation files; both are formal.  'NN' is the tagger's default tag for unknown words.
GU_FORMAL_PREFIX = ("N_", "N-", "JJ", "PSP")
GU_DEICTIC_PREFIX = ("PR_", "V_", "VM_", "RB", "RP_", "DM_", "CC_")


def gu_class(tag):
    if tag == "NN" or tag.startswith(GU_FORMAL_PREFIX):
        return "F"
    if tag.startswith(GU_DEICTIC_PREFIX):
        return "D"
    return "N"
EN_DEMONSTRATIVES = {"this", "that", "these", "those"}


def en_class(word, tag):
    if tag.startswith("NN") or tag.startswith("JJ") or tag == "IN":
        return "F"
    if tag in ("PRP", "PRP$", "WP", "WP$", "MD", "RP", "UH", "CC", "WRB") or tag.startswith("VB") or tag.startswith("RB"):
        return "D"
    if tag in ("DT", "WDT") and word.lower() in EN_DEMONSTRATIVES:
        return "D"
    return "N"  # articles, numerals, symbols, ... are neutral (not counted)


def shannon_chars(s):
    c = Counter(s)
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values()) if n else 0.0


class GuTagger:
    """Unigram->Bigram->Trigram back-off tagger with default 'NN', trained on 90% of gu.pos (as in the notebooks)."""

    # Hierarchical BIS tag at the END of a token, e.g. "દાખલ_N_NN" -> ("દાખલ", "N_NN").
    # NOTE: nltk's IndianCorpusReader (used by the original notebooks) splits at the LAST underscore and would
    # return ("દાખલ_N", "NN") for such tokens, so tags like N_NN / V_VM / PR_PRP never appear and the training
    # words keep a tag fragment.  We parse the file ourselves and log the tag inventory so this can be checked.
    _TOKEN_TAG = re.compile(r"^(.+?)_((?:N|PR|DM|V|VM|RP|CC|QT|RD)[_-][A-Z]+(?:_[A-Z]+)*|JJ|RB|PSP|NN|NNP|NST|[A-Z]+)$")

    @classmethod
    def read_pos(cls, path):
        sents = []
        for line in _read(path).splitlines():
            line = re.sub(r"</?Sentence[^>]*>", " ", line).strip()
            if not line or line.startswith("<"):
                continue
            sent = []
            for tok in line.split():
                m = cls._TOKEN_TAG.match(tok)
                if m:
                    sent.append((m.group(1), m.group(2)))
            if sent:
                sents.append(sent)
        return sents

    def __init__(self, gu_pos, log=print):
        import nltk
        data = self.read_pos(gu_pos)
        inv = Counter(t for s in data for _, t in s)
        log("gu.pos: %d sentences, %d tokens; most frequent tags: %s" % (len(data), sum(inv.values()), inv.most_common(25)))
        hier = sum(v for t, v in inv.items() if "_" in t)
        log("gu.pos: %.1f%% of tokens carry hierarchical BIS tags (e.g. N_NN)" % (100 * hier / max(1, sum(inv.values()))))
        size = int(len(data) * 0.9)
        self.train, self.test = data[:size], data[size:]
        back = nltk.DefaultTagger("NN")
        self.no_default = None
        for cls in (nltk.UnigramTagger, nltk.BigramTagger, nltk.TrigramTagger):
            back = cls(self.train, backoff=back)
        self.tagger = back
        nd = None
        for cls in (nltk.UnigramTagger, nltk.BigramTagger, nltk.TrigramTagger):
            nd = cls(self.train, backoff=nd)
        self.no_default = nd

    def tag(self, tokens):
        return self.tagger.tag(tokens)


def make_formality_scorer(variant):
    """variant: set of switches. 'w1' -> formal weight 1, '-ent', '-num', '-short', '-lenbonus', 'core' (formula only)."""

    def scorer(doc, ctx):
        lower = ctx["lower"]
        w_formal = 1 if "w1" in variant else 10
        senlength, lensen, scores = 8, 0, []
        for s in doc.sents:
            toks = content_tokens(s, ctx["stop"], lower) or ["hi"]  # notebook uses a dummy token for empty sentences
            if doc.dataset == "gujarati":
                classes = [gu_class(t) for _, t in ctx["gu_tagger"].tag(toks)]
                formal, deictic = classes.count("F"), classes.count("D")
            else:
                import nltk
                classes = [en_class(w, t) for w, t in nltk.pos_tag(tokenize(s))]
                formal, deictic = classes.count("F"), classes.count("D")
            f = ((1 + w_formal * formal) - (1 + deictic) + 100) / 2.0
            if "core" not in variant:
                plain = "".join(c for c in s if c not in string.punctuation)
                if "-short" not in variant and len(plain.split()) <= 8:
                    f -= 1
                if "-lenbonus" not in variant and len(plain) > senlength:
                    senlength = len(plain); lensen += 1; f += lensen
                if "-num" not in variant and any(c.isdigit() for c in plain):
                    f += 1
                if "-ent" not in variant:
                    f += shannon_chars(plain)
            scores.append(f)
        return scores

    return scorer


ABLATIONS = {
    "formality": set(),
    "abl:core-formula-only": {"core"},
    "abl:no-entropy": {"-ent"},
    "abl:no-numeric": {"-num"},
    "abl:no-short-penalty": {"-short"},
    "abl:no-length-bonus": {"-lenbonus"},
    "abl:formal-weight-1": {"w1"},
}

BASELINES = {
    "lead": score_lead,
    "tfidf": score_tfidf,
    "textrank": score_textrank,
    "lsa": score_lsa,
    "lda": score_lda,
    "labse-centroid": score_labse_centroid,
    "labse-textrank": score_labse_textrank,
    "fuzzy": score_fuzzy,
}

PRETTY = {"formality": "Formality (ours)", "lead": "Lead", "tfidf": "TF-IDF", "textrank": "TextRank", "lsa": "LSA",
          "lda": "LDA", "lsa-defaulttok": "LSA (default tokeniser)", "lda-defaulttok": "LDA (default tokeniser)",
          "labse-centroid": "LaBSE-Centroid", "labse-textrank": "LaBSE-TextRank", "fuzzy": "Fuzzy logic"}


# --------------------------------------------------------------------------
# Running methods
# --------------------------------------------------------------------------
def run_method(name, scorer, docs, ctx, out_dir, log):
    rows = []
    for i, d in enumerate(docs):
        try:
            scores = scorer(d, ctx)
        except Exception as e:  # never silently drop a document: fall back and log it
            log("  [%s] %s/%s failed (%s) -> lead fallback" % (name, d.category, d.name, e))
            scores = score_lead(d, ctx)
        for r in RATIOS:
            idx = select(scores, r)
            summ = [d.sents[k] for k in idx]
            p = os.path.join(out_dir, "summaries", d.dataset, name, str(r), d.category)
            os.makedirs(p, exist_ok=True)
            with open(os.path.join(p, d.name), "w", encoding="utf-8") as f:
                f.write("\n".join(summ))
            for a, ref in d.refs.items():
                m = evaluate(summ, ref, ctx["lower"])
                rows.append(dict(method=name, ratio=r, annotator=a, category=d.category, doc=d.name, **m))
        if (i + 1) % 100 == 0:
            log("  [%s] %d/%d documents" % (name, i + 1, len(docs)))
    return rows


def evaluate_external(name, summ_root, docs, ctx):
    """Score summaries produced elsewhere (e.g. the fuzzy-logic notebook), laid out as <root>/<ratio>/<category>/<file>."""
    rows = []
    for d in docs:
        for r in RATIOS:
            p = os.path.join(summ_root, str(r), d.category, d.name)
            if not os.path.exists(p):
                continue
            text = _read(p)
            summ = split_gu(text) if d.dataset == "gujarati" else split_en(text)
            for a, ref in d.refs.items():
                rows.append(dict(method=name, ratio=r, annotator=a, category=d.category, doc=d.name,
                                 **evaluate(summ, ref, ctx["lower"])))
    return rows


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
METRICS = ("R1_F", "R2_F", "RL_F", "BLEU")


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired_stats(rows, reference="formality"):
    from scipy import stats as ss
    by = defaultdict(dict)
    for r in rows:
        by[(r["method"], r["ratio"], r["annotator"])][(r["category"], r["doc"])] = r
    methods = sorted({r["method"] for r in rows if r["method"] != reference and not r["method"].startswith("abl:")})
    out = []
    for ratio in RATIOS:
        for ann in sorted({r["annotator"] for r in rows}):
            for metric in METRICS:
                block = []
                for m in methods:
                    A, B = by.get((reference, ratio, ann), {}), by.get((m, ratio, ann), {})
                    keys = sorted(set(A) & set(B))
                    if len(keys) < 3:
                        continue
                    a = [A[k][metric] for k in keys]
                    b = [B[k][metric] for k in keys]
                    diff = [x - y for x, y in zip(a, b)]
                    md, sd = st.mean(diff), st.stdev(diff)
                    se = sd / math.sqrt(len(diff))
                    tcrit = ss.t.ppf(0.975, len(diff) - 1)
                    t_res = ss.ttest_rel(a, b)
                    p_greater = ss.ttest_rel(a, b, alternative="greater").pvalue
                    try:
                        w_p = ss.wilcoxon(a, b).pvalue
                    except ValueError:
                        w_p = 1.0
                    block.append(dict(ratio=ratio, annotator=ann, metric=metric, baseline=m, n=len(diff),
                                      mean_ref=st.mean(a), mean_base=st.mean(b), mean_diff=md,
                                      rel_gain_pct=100 * md / st.mean(b) if st.mean(b) else float("nan"),
                                      ci_low=md - tcrit * se, ci_high=md + tcrit * se,
                                      t=t_res.statistic, df=len(diff) - 1, p_two_sided=t_res.pvalue,
                                      p_one_sided_greater=p_greater, p_wilcoxon=w_p,
                                      cohens_dz=md / sd if sd else float("nan")))
                for rec, adj_t, adj_w in zip(block, holm([b["p_two_sided"] for b in block]), holm([b["p_wilcoxon"] for b in block])):
                    rec["p_two_sided_holm"], rec["p_wilcoxon_holm"] = adj_t, adj_w
                    out.append(rec)
    return out


# --------------------------------------------------------------------------
# Auxiliary analyses requested by the reviewers
# --------------------------------------------------------------------------
def selected_indices(doc):
    """Map every reference line to the source sentence with the highest token Jaccard (>= 0.6)."""
    src = [set(tokenize(s)) for s in doc.sents]
    chosen = {}
    for a, lines in doc.refs.items():
        sel = set()
        for line in lines:
            lt = set(tokenize(line))
            best = max(range(len(src)), key=lambda i: len(lt & src[i]) / (len(lt | src[i]) or 1))
            if len(lt & src[best]) / (len(lt | src[best]) or 1) >= 0.6:
                sel.add(best)
        chosen[a] = sel
    return chosen


def _auc(pos, neg):
    """P(score of a selected sentence > score of a non-selected one), ties count 1/2 (= Mann-Whitney U / n1 n2)."""
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = [0.0] * len(allv), 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, allv) if g == 1)
    return (r1 - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def selection_analysis(docs, ctx, out_dir, log):
    """Do human annotators prefer formal sentences?  Compares formality scores of selected vs. non-selected sentences."""
    core, full = make_formality_scorer({"core"}), make_formality_scorer(set())
    sel_core, uns_core, sel_full, uns_full = [], [], [], []
    ex = open(os.path.join(out_dir, "formality_examples_gu.tsv"), "w", encoding="utf-8")
    ex.write("category\tdoc\tidx\tcore_formality\tfull_score\tselected_A1\tselected_A2\tsentence\n")
    for d in docs:
        c, f = core(d, ctx), full(d, ctx)
        chosen = selected_indices(d)
        anyset = set().union(*chosen.values())
        for i, s in enumerate(d.sents):
            (sel_core if i in anyset else uns_core).append(c[i])
            (sel_full if i in anyset else uns_full).append(f[i])
            if d.name in ("1.txt", "2.txt", "3.txt"):
                ex.write("%s\t%s\t%d\t%.2f\t%.2f\t%d\t%d\t%s\n" % (d.category, d.name, i, c[i], f[i],
                         i in chosen.get("1", ()), i in chosen.get("2", ()), s))
    ex.close()
    res = dict(n_sel=len(sel_core), n_unsel=len(uns_core),
               core_sel=st.mean(sel_core), core_unsel=st.mean(uns_core), core_auc=_auc(sel_core, uns_core),
               full_sel=st.mean(sel_full), full_unsel=st.mean(uns_full), full_auc=_auc(sel_full, uns_full))
    try:
        from scipy import stats as ss
        res["core_p"] = ss.mannwhitneyu(sel_core, uns_core, alternative="greater").pvalue
        res["full_p"] = ss.mannwhitneyu(sel_full, uns_full, alternative="greater").pvalue
    except ImportError:
        pass
    log("Selection analysis (formality of annotator-selected vs non-selected sentences): " + json.dumps(res))
    return res


def sentence_kappa(docs):
    """Sentence-level Cohen's kappa between the two annotators (selected / not selected)."""
    tab = Counter()
    for d in docs:
        if not {"1", "2"} <= set(d.refs):
            continue
        src = d.sents
        chosen = selected_indices(d)
        for i in range(len(src)):
            tab[(i in chosen["1"], i in chosen["2"])] += 1
    n = sum(tab.values())
    po = (tab[(True, True)] + tab[(False, False)]) / n
    p1 = (tab[(True, True)] + tab[(True, False)]) / n
    p2 = (tab[(True, True)] + tab[(False, True)]) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return dict(kappa=(po - pe) / (1 - pe), observed=po, sentences=n, both=tab[(True, True)])


def reference_ratios(docs):
    out = {}
    for a in ("1", "2"):
        vals = [len(d.refs[a]) / len(d.sents) for d in docs if a in d.refs]
        out[a] = (st.mean(vals), st.stdev(vals))
    return out


def pos_analysis(tagger, docs, stop):
    """Held-out accuracy, formality-class confusion, sentence-level F error, OOV/default-tag rate on the corpus."""
    cls = gu_class
    tok_total = tok_ok = 0
    conf = Counter()
    f_gold, f_pred = [], []
    for sent in tagger.test:
        words = [w for w, _ in sent]
        gold = [t for _, t in sent]
        pred = [t for _, t in tagger.tag(words)]
        tok_total += len(gold); tok_ok += sum(g == p for g, p in zip(gold, pred))
        for g, p in zip(gold, pred):
            conf[(cls(g), cls(p))] += 1
        fg = (sum(cls(t) == "F" for t in gold) - sum(cls(t) == "D" for t in gold) + 100) / 2
        fp = (sum(cls(t) == "F" for t in pred) - sum(cls(t) == "D" for t in pred) + 100) / 2
        f_gold.append(fg); f_pred.append(fp)
    n_cls = sum(conf.values())
    class_acc = sum(v for (g, p), v in conf.items() if g == p) / n_cls
    recall = {c: conf[(c, c)] / max(1, sum(v for (g, _), v in conf.items() if g == c)) for c in "FDN"}
    precision = {c: conf[(c, c)] / max(1, sum(v for (_, p), v in conf.items() if p == c)) for c in "FDN"}
    mae = st.mean(abs(a - b) for a, b in zip(f_gold, f_pred))
    r = _pearson(f_gold, f_pred)
    # Words in the summarisation corpus that the n-gram taggers cannot tag (they receive the default 'NN')
    oov = tot = 0
    for d in docs:
        for s in d.sents:
            toks = content_tokens(s, stop, False)
            tags = tagger.no_default.tag(toks)
            tot += len(tags); oov += sum(t is None for _, t in tags)
    return dict(train_sents=len(tagger.train), test_sents=len(tagger.test), token_acc=tok_ok / tok_total,
                class_acc=class_acc, recall=recall, precision=precision, confusion=dict((("%s->%s" % k), v) for k, v in conf.items()),
                f_mae=mae, f_pearson=r, oov_rate=oov / max(1, tot), corpus_tokens=tot)


def _pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


_TAGRE = re.compile(r"_((?:N|PR|DM|V|RP|CC|QT|RD)[_-][A-Z]+(?:_[A-Z]+)*|JJ|RB|PSP)(?=[^A-Z_-]|$)")


def gold_formality(paths):
    """Sentence-level formality from manually POS-tagged ILCI-style files (percentage form of the formula)."""
    vals, dist, ntok = [], Counter(), 0
    for p in paths:
        for m in re.finditer(r"<Sentence[^>]*>(.*?)</Sentence>", _read(p), re.S):
            tags = [t for t in _TAGRE.findall(m.group(1)) if not t.startswith("RD_")]
            if not tags:
                continue
            n = len(tags)
            fpct = 100 * sum(gu_class(t) == "F" for t in tags) / n
            dpct = 100 * sum(gu_class(t) == "D" for t in tags) / n
            vals.append((fpct - dpct + 100) / 2)
            ntok += n
            for t in tags:
                for name, pre in (("noun", "N_"), ("adjective", "JJ"), ("postposition", "PSP"), ("pronoun", "PR_"),
                                  ("verb", "V_"), ("adverb", "RB"), ("particle", "RP_"), ("demonstrative", "DM_"),
                                  ("conjunction", "CC_"), ("quantifier", "QT_")):
                    if t.startswith(pre):
                        dist[name] += 1
    return vals, {k: 100 * v / ntok for k, v in dist.items()}, ntok


def pos_files(path):
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".pos")]
    return [path]


# --------------------------------------------------------------------------
# LaTeX output
# --------------------------------------------------------------------------
def means(rows):
    acc = defaultdict(list)
    for r in rows:
        for m in METRICS + ("R1_P", "R1_R", "R2_P", "R2_R", "RL_P", "RL_R"):
            acc[(r["method"], r["ratio"], r["annotator"], m)].append(r[m])
    return {k: st.mean(v) for k, v in acc.items()}


def latex_main_table(M, methods, annotators, caption, label):
    cols = len(annotators) * 4
    head1 = " & ".join(r"\multicolumn{4}{c}{%s}" % ("Annotator-%s" % a if len(annotators) > 1 else "Reference") for a in annotators)
    head2 = " & ".join(["R-1 & R-2 & R-L & BLEU"] * len(annotators))
    lines = [r"\begin{table*}[!t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{%s}\label{%s}" % (caption, label),
             r"\begin{tabular}{ll" + "c" * cols + "}", r"\toprule",
             r"CR & Method & " + head1 + r" \\", r"\midrule" if len(annotators) == 1 else
             "".join(r"\cmidrule(lr){%d-%d}" % (3 + 4 * i, 6 + 4 * i) for i in range(len(annotators))),
             r" & & " + head2 + r" \\", r"\midrule"]
    for ratio in RATIOS:
        best = {}
        for a in annotators:
            for m in METRICS:
                best[(a, m)] = max(M.get((meth, ratio, a, m), -1) for meth in methods)
        for i, meth in enumerate(methods):
            cells = []
            for a in annotators:
                for m in METRICS:
                    v = M.get((meth, ratio, a, m))
                    s = "--" if v is None else "%.4f" % v
                    if v is not None and abs(v - best[(a, m)]) < 1e-12:
                        s = r"\textbf{%s}" % s
                    cells.append(s)
            lines.append(("%d\\%%" % ratio if i == 0 else "") + " & " + PRETTY.get(meth, meth) + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if ratio != RATIOS[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def latex_pr_table(M, methods, annotator, caption, label):
    lines = [r"\begin{table*}[!t]", r"\centering", r"\footnotesize", r"\caption{%s}\label{%s}" % (caption, label),
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r"CR & Method & \multicolumn{2}{c}{ROUGE-1} & \multicolumn{2}{c}{ROUGE-2} & \multicolumn{2}{c}{ROUGE-L} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}", r" & & P & R & P & R & P & R \\", r"\midrule"]
    for ratio in RATIOS:
        for i, meth in enumerate(methods):
            cells = ["%.4f" % M.get((meth, ratio, annotator, "%s_%s" % (m, pr)), float("nan")) for m in ("R1", "R2", "RL") for pr in "PR"]
            lines.append(("%d\\%%" % ratio if i == 0 else "") + " & " + PRETTY.get(meth, meth) + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if ratio != RATIOS[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else "%.3f" % p


def latex_stats_table(stats, metric, annotators, caption, label):
    sel = [s for s in stats if s["metric"] == metric]
    lines = [r"\begin{table*}[!t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
             r"\caption{%s}\label{%s}" % (caption, label),
             r"\begin{tabular}{ll" + "rcc" * len(annotators) + "}", r"\toprule",
             r"CR & Baseline & " + " & ".join(r"\multicolumn{3}{c}{%s}" % ("Annotator-%s" % a if len(annotators) > 1 else "Reference") for a in annotators) + r" \\",
             "".join(r"\cmidrule(lr){%d-%d}" % (3 + 3 * i, 5 + 3 * i) for i in range(len(annotators))),
             r" & & " + " & ".join([r"$\Delta$ [95\% CI] & $p_t$ & $p_W$"] * len(annotators)) + r" \\", r"\midrule"]
    for ratio in RATIOS:
        bases = sorted({s["baseline"] for s in sel if s["ratio"] == ratio})
        for i, b in enumerate(bases):
            cells = []
            for a in annotators:
                rec = next((s for s in sel if s["ratio"] == ratio and s["annotator"] == a and s["baseline"] == b), None)
                if rec is None:
                    cells += ["--", "--", "--"]
                else:
                    cells += ["%+.3f [%.3f, %.3f]" % (rec["mean_diff"], rec["ci_low"], rec["ci_high"]),
                              fmt_p(rec["p_two_sided_holm"]), fmt_p(rec["p_wilcoxon_holm"])]
            lines.append(("%d\\%%" % ratio if i == 0 else "") + " & " + PRETTY.get(b, b) + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if ratio != RATIOS[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def latex_ablation(M, annotators):
    names = list(ABLATIONS)
    lines = [r"\begin{table}[!t]", r"\centering", r"\footnotesize",
             r"\caption{Ablation of the sentence-scoring components (Gujarati, ROUGE F1 averaged over both annotators)}\label{tab:ablation}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Variant & \multicolumn{3}{c}{20\%} & \multicolumn{3}{c}{30\%} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r" & R-1 & R-2 & R-L & R-1 & R-2 & R-L \\", r"\midrule"]
    pretty = {"formality": "Full score (Eq.~\\ref{eq:sentscore})", "abl:core-formula-only": "Formality formula only",
              "abl:no-entropy": "$-$ Shannon entropy", "abl:no-numeric": "$-$ numeric bonus",
              "abl:no-short-penalty": "$-$ short-sentence penalty", "abl:no-length-bonus": "$-$ length bonus",
              "abl:formal-weight-1": "formal weight $w_f=1$"}
    for n in names:
        cells = []
        for ratio in (20, 30):
            for m in ("R1_F", "R2_F", "RL_F"):
                vals = [M[(n, ratio, a, m)] for a in annotators if (n, ratio, a, m) in M]
                cells.append("%.4f" % st.mean(vals) if vals else "--")
        lines.append(pretty[n] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="'gujarati Text summarization dataset' folder")
    ap.add_argument("--gu-pos", help="BIS-tagged Gujarati training file used for the POS tagger (gu.pos)")
    ap.add_argument("--stopwords", help="Gujarati stop-word list (stop-words.txt)")
    ap.add_argument("--bbc", help="BBC News Summary folder (contains 'News Articles' and 'Summaries') or the CSV mirror")
    ap.add_argument("--legal-pos", help="folder of manually tagged court judgements (*.pos)")
    ap.add_argument("--informal-pos", help="manually tagged informal file (code.pos)")
    ap.add_argument("--health-pos", help="optional folder/file of manually tagged health articles")
    ap.add_argument("--fuzzy-gu", help="optional: fuzzy-logic summaries as <dir>/<ratio>/<category>/<file>")
    ap.add_argument("--fuzzy-en", help="optional: same for English")
    ap.add_argument("--labse-model", default="sentence-transformers/LaBSE")
    ap.add_argument("--methods", default="formality,lead,tfidf,textrank,lsa,lda,fuzzy,labse-centroid,labse-textrank")
    ap.add_argument("--reuse", action="store_true",
                    help="load scores_<dataset>.csv from --out and only run methods that are not in it yet")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--no-aem-demo", action="store_true", help="skip LSA/LDA with sklearn's default token pattern")
    ap.add_argument("--only", choices=("gujarati", "english"), help="run only one corpus (e.g. to run both in parallel)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    report = open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg); report.write(msg + "\n"); report.flush()

    macros = {}
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # ------------------------------- Gujarati --------------------------------
    gu_docs = [] if args.only == "english" else load_gujarati(args.corpus)
    if args.only != "english":
        log("Gujarati documents: %d, sentences: %d" % (len(gu_docs), sum(len(d.sents) for d in gu_docs)))
        ctx = dict(stop=load_stopwords(args.stopwords), lower=False, labse=args.labse_model)

        k = sentence_kappa(gu_docs)
        rr = reference_ratios(gu_docs)
        log("Sentence-level Cohen's kappa = %.3f (observed agreement %.3f, %d sentences)" % (k["kappa"], k["observed"], k["sentences"]))
        log("Reference length (fraction of source sentences): A1 %.3f (sd %.3f), A2 %.3f (sd %.3f)" % (rr["1"] + rr["2"]))
        macros.update(KappaSent="%.2f" % k["kappa"], AgreeObs="%.1f" % (100 * k["observed"]),
                      RefRatioOne="%.1f" % (100 * rr["1"][0]), RefRatioTwo="%.1f" % (100 * rr["2"][0]))

        if args.gu_pos and "formality" in methods:
            ctx["gu_tagger"] = GuTagger(args.gu_pos, log)
            pa = pos_analysis(ctx["gu_tagger"], gu_docs, ctx["stop"])
            log("POS analysis: " + json.dumps(pa, indent=1))
            macros.update(TaggerTrain=str(pa["train_sents"]), TaggerTest=str(pa["test_sents"]),
                          TaggerAcc="%.2f" % (100 * pa["token_acc"]), TaggerClassAcc="%.2f" % (100 * pa["class_acc"]),
                          FormalRecall="%.1f" % (100 * pa["recall"]["F"]), DeicticRecall="%.1f" % (100 * pa["recall"]["D"]),
                          FormalPrecision="%.1f" % (100 * pa["precision"]["F"]), DeicticPrecision="%.1f" % (100 * pa["precision"]["D"]),
                          FormalityMAE="%.2f" % pa["f_mae"], FormalityPearson="%.3f" % pa["f_pearson"],
                          OOVRate="%.1f" % (100 * pa["oov_rate"]))
            sa = selection_analysis(gu_docs, ctx, args.out, log)
            macros.update(SelCoreSel="%.2f" % sa["core_sel"], SelCoreUnsel="%.2f" % sa["core_unsel"],
                          SelCoreAUC="%.3f" % sa["core_auc"], SelFullAUC="%.3f" % sa["full_auc"],
                          SelCoreP=fmt_p(sa.get("core_p", float("nan"))))
        elif "formality" in methods:
            log("!! --gu-pos not given: formality method skipped")
            methods.remove("formality")

        rows = load_scores(args.out, "gujarati") if args.reuse else []
        done = {r["method"] for r in rows}
        for m in methods:
            if m in done:
                log("Re-using stored scores for %s" % m); continue
            log("Running %s ..." % m)
            scorer = make_formality_scorer(ABLATIONS["formality"]) if m == "formality" else BASELINES[m]
            rows += run_method(m, scorer, gu_docs, ctx, args.out, log)
        if "formality" in methods and not args.no_ablation:
            for name, var in ABLATIONS.items():
                if name != "formality" and name not in done:
                    log("Running %s ..." % name)
                    rows += run_method(name, make_formality_scorer(var), gu_docs, ctx, args.out, log)
        if not args.no_aem_demo:
            for m in ("lsa", "lda"):
                if m in methods and m + "-defaulttok" not in done:
                    log("Running %s with sklearn default token pattern (AEM demonstration) ..." % m)
                    c2 = dict(ctx, default_token_pattern=True)
                    rows += run_method(m + "-defaulttok", BASELINES[m], gu_docs, c2, args.out, log)
        if args.fuzzy_gu:
            rows += evaluate_external("fuzzy", args.fuzzy_gu, gu_docs, ctx)
        write_outputs("gujarati", rows, ["1", "2"], args, log, macros)

    # -------------------------------- English --------------------------------
    if args.bbc and args.only != "gujarati":
        en_docs = load_bbc_csv(args.bbc) if args.bbc.endswith(".csv") else load_bbc(args.bbc)
        log("English documents: %d" % len(en_docs))
        ectx = dict(stop=load_stopwords(None, english=True), lower=True, labse=args.labse_model)
        erows = load_scores(args.out, "english") if args.reuse else []
        edone = {r["method"] for r in erows}
        for m in methods:
            if m in edone:
                log("Re-using stored English scores for %s" % m); continue
            log("Running %s (English) ..." % m)
            scorer = make_formality_scorer(set()) if m == "formality" else BASELINES[m]
            erows += run_method(m, scorer, en_docs, ectx, args.out, log)
        if args.fuzzy_en:
            erows += evaluate_external("fuzzy", args.fuzzy_en, en_docs, ectx)
        write_outputs("english", erows, ["1"], args, log, macros)

    # ------------------------- gold-tag formality check -----------------------
    if args.legal_pos and args.informal_pos:
        from scipy import stats as ss
        groups = [("Legal judgements", pos_files(args.legal_pos))]
        if args.health_pos:
            groups.append(("Health articles", pos_files(args.health_pos)))
        groups.append(("Informal comments", pos_files(args.informal_pos)))
        res = {name: gold_formality(paths) for name, paths in groups}
        legal, informal = res["Legal judgements"][0], res["Informal comments"][0]
        t = ss.ttest_ind(legal, informal, equal_var=False, alternative="greater")
        u = ss.mannwhitneyu(legal, informal, alternative="greater")
        log("Gold-tag formality: " + "; ".join("%s n=%d mean=%.2f sd=%.2f" % (n, len(v[0]), st.mean(v[0]), st.stdev(v[0])) for n, v in res.items()))
        log("  Welch t=%.2f p=%.3g | Mann-Whitney U=%.0f p=%.3g" % (t.statistic, t.pvalue, u.statistic, u.pvalue))
        lines = [r"\begin{table}[!t]", r"\centering", r"\footnotesize",
                 r"\caption{POS-class distribution (\% of non-punctuation tokens) and sentence-level formality computed from manually assigned BIS tags}\label{tab:goldpos}",
                 r"\begin{tabular}{l" + "c" * len(res) + "}", r"\toprule", "Class & " + " & ".join(res) + r" \\", r"\midrule"]
        for cl in ("noun", "adjective", "postposition", "pronoun", "verb", "adverb", "particle", "demonstrative", "conjunction"):
            lines.append(cl.capitalize() + " & " + " & ".join("%.1f" % v[1].get(cl, 0) for v in res.values()) + r" \\")
        lines += [r"\midrule", "Sentences & " + " & ".join(str(len(v[0])) for v in res.values()) + r" \\",
                  "Formality, mean (SD) & " + " & ".join("%.2f (%.2f)" % (st.mean(v[0]), st.stdev(v[0])) for v in res.values()) + r" \\",
                  r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(args.out, "tables", "gold_pos_formality.tex"), "w", encoding="utf-8").write("\n".join(lines))

    with open(os.path.join(args.out, "macros.tex"), "w", encoding="utf-8") as f:
        f.write("% Paste these lines into sn-article.tex directly BELOW the block marked 'RESULT MACROS'\n")
        for k_, v in sorted(macros.items()):
            f.write("\\def\\%s{%s}\n" % (k_, v))
    log("Done. Tables in %s/tables, macros in %s/macros.tex" % (args.out, args.out))


def load_scores(out_dir, dataset):
    p = os.path.join(out_dir, "scores_%s.csv" % dataset)
    if not os.path.exists(p):
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ratio"] = int(r["ratio"])
            for k in METRICS + ("R1_P", "R1_R", "R2_P", "R2_R", "RL_P", "RL_R"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def write_outputs(dataset, rows, annotators, args, log, macros):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "scores_%s.csv" % dataset), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    M = means(rows)
    main_methods = [m for m in ("formality", "lead", "tfidf", "textrank", "lsa", "lda", "fuzzy", "labse-centroid", "labse-textrank")
                    if any(r["method"] == m for r in rows)]
    for (meth, ratio, a, met), v in sorted(M.items()):
        if met in METRICS:
            log("%s %-24s %d%% A%s %s = %.4f" % (dataset, meth, ratio, a, met, v))
    tag = "gu" if dataset == "gujarati" else "en"
    cap = "Gujarati corpus" if dataset == "gujarati" else "English BBC News Summary corpus"
    open(os.path.join(args.out, "tables", "main_%s.tex" % tag), "w", encoding="utf-8").write(latex_main_table(
        M, main_methods, annotators,
        "Word-level ROUGE F1 and BLEU-4 on the %s at each compression ratio (CR); best value per column in bold" % cap,
        "tab:main%s" % tag))
    for a in annotators:
        open(os.path.join(args.out, "tables", "pr_%s_a%s.tex" % (tag, a)), "w", encoding="utf-8").write(latex_pr_table(
            M, main_methods, a, "ROUGE precision (P) and recall (R), %s%s" % (cap, ", Annotator-%s" % a if len(annotators) > 1 else ""),
            "tab:pr%s%s" % (tag, a)))
    demo = [m for m in ("lsa", "lsa-defaulttok", "lda", "lda-defaulttok") if any(r["method"] == m for r in rows)]
    if len(demo) == 4:
        open(os.path.join(args.out, "tables", "aem_tokeniser_%s.tex" % tag), "w", encoding="utf-8").write(latex_main_table(
            M, demo, annotators, "Effect of the tokeniser on LSA and LDA (AEM demonstration), %s" % cap, "tab:aemtok"))
    if any(r["method"] == "formality" for r in rows):
        stats = paired_stats([r for r in rows if r["method"] in main_methods])
        with open(os.path.join(args.out, "stats_%s.csv" % dataset), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stats[0].keys())); w.writeheader(); w.writerows(stats)
        for metric in ("R1_F", "BLEU"):
            open(os.path.join(args.out, "tables", "stats_%s_%s.tex" % (tag, metric)), "w", encoding="utf-8").write(latex_stats_table(
                stats, metric, annotators,
                "Paired comparison of the formality method with each baseline (%s, %s): mean per-document difference "
                "$\\Delta$ (formality $-$ baseline) with 95\\%% CI, Holm-adjusted two-sided paired $t$-test ($p_t$) and Wilcoxon signed-rank ($p_W$) $p$-values"
                % (cap, metric.replace("_F", " F1").replace("R1", "ROUGE-1")), "tab:stats%s%s" % (tag, metric.replace("_", ""))))
        if dataset == "gujarati" and any(r["method"].startswith("abl:") for r in rows):
            open(os.path.join(args.out, "tables", "ablation_gu.tex"), "w", encoding="utf-8").write(latex_ablation(M, annotators))
        # headline macros: formality vs best competing method, averaged over annotators
        pre = "Gu" if dataset == "gujarati" else "En"
        for ratio in RATIOS:
            for met in ("R1_F", "R2_F", "RL_F", "BLEU"):
                f_val = st.mean(M[("formality", ratio, a, met)] for a in annotators)
                others = {m: st.mean(M[(m, ratio, a, met)] for a in annotators) for m in main_methods if m != "formality"}
                best_m = max(others, key=others.get)
                nm = {"R1_F": "Rone", "R2_F": "Rtwo", "RL_F": "RL", "BLEU": "Bleu"}[met] + {20: "Twenty", 30: "Thirty", 40: "Forty", 50: "Fifty"}[ratio]
                macros[pre + "Form" + nm] = "%.4f" % f_val
                macros[pre + "Best" + nm] = "%.4f" % others[best_m]
                macros[pre + "BestName" + nm] = PRETTY.get(best_m, best_m)
                macros[pre + "Gain" + nm] = "%+.2f" % (100 * (f_val - others[best_m]) / others[best_m])


if __name__ == "__main__":
    main()
