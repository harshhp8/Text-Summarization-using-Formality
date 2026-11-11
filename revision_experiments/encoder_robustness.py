#!/usr/bin/env python3
"""Encoder-robustness check for the unsupervised neural baselines.

The main comparison (rerun_all.py) builds its neural baselines on LaBSE.  This script repeats the
Centroid and TextRank baselines with other publicly available multilingual sentence encoders that
cover Gujarati, to test whether the conclusions depend on the choice of encoder:

  * sentence-transformers/LaBSE                   (Feng et al., ACL 2022)
  * l3cube-pune/indic-sentence-similarity-sbert   (IndicSBERT; Deode et al., 2023) - Indic-specific
  * intfloat/multilingual-e5-base                 (Wang et al., 2024) - inputs prefixed with "query: "
  * BAAI/bge-m3                                   (Chen et al., 2024) - dense vectors

Sentence splitting, tokenization, selection rule and metrics are identical to rerun_all.py.
The formality method's per-document scores are read from the scores CSV written by rerun_all.py,
so the paired tests compare exactly the same documents.

Usage (after rerun_all.py has finished):
  python encoder_robustness.py --corpus ".../gujarati Text summarization dataset" \
      --stopwords .../stop-words.txt --scores results_gu/scores_gujarati.csv --out results_enc
"""
import argparse
import csv
import gc
import os
import statistics as st
from collections import defaultdict

import rerun_all as R

ENCODERS = [
    ("LaBSE", "sentence-transformers/LaBSE", ""),
    ("IndicSBERT", "l3cube-pune/indic-sentence-similarity-sbert", ""),
    ("mE5-base", "intfloat/multilingual-e5-base", "query: "),
    ("BGE-M3", "BAAI/bge-m3", ""),
]


def make_scorers(model, prefix):
    import numpy as np
    cache = {}

    def emb(doc):
        key = (doc.category, doc.name)
        if key not in cache:  # kept for all documents, so Centroid and TextRank share one encoding pass
            cache[key] = model.encode([prefix + s for s in doc.sents], normalize_embeddings=True,
                                      batch_size=32, show_progress_bar=False)
        return cache[key]

    def centroid(doc, ctx):
        E = emb(doc)
        c = E.mean(axis=0)
        c = c / (np.linalg.norm(c) or 1.0)
        return list(E @ c)

    def textrank(doc, ctx):
        E = emb(doc)
        S = np.clip(E @ E.T, 0, None)
        np.fill_diagonal(S, 0)
        return R._pagerank(S.tolist())

    return centroid, textrank


def load_reference_rows(path, method="formality"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["method"] != method:
                continue
            r = dict(r)
            r["ratio"] = int(r["ratio"])
            for k in list(r):
                if k[:2] in ("R1", "R2", "RL") or k == "BLEU":
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--stopwords")
    ap.add_argument("--scores", required=True, help="scores_gujarati.csv written by rerun_all.py")
    ap.add_argument("--encoders", default=",".join(e[0] for e in ENCODERS))
    ap.add_argument("--out", default="results_enc")
    ap.add_argument("--no-table", action="store_true", help="only compute and cache the requested encoders")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    report = open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); report.write(msg + "\n"); report.flush()

    from sentence_transformers import SentenceTransformer
    docs = R.load_gujarati(args.corpus)
    ctx = dict(stop=R.load_stopwords(args.stopwords), lower=False)
    rows = load_reference_rows(args.scores)
    log("Gujarati documents: %d; formality rows loaded: %d" % (len(docs), len(rows)))

    # LaBSE rows are taken from rerun_all.py's output (same documents, same code) instead of being recomputed
    for src, dst in (("labse-centroid", "LaBSE-Centroid"), ("labse-textrank", "LaBSE-TextRank")):
        extra = load_reference_rows(args.scores, src)
        for r in extra:
            r["method"] = dst
        rows += extra

    wanted = [e for e in ENCODERS if e[0] in args.encoders.split(",")]
    for short, hf_id, prefix in wanted:
        if short == "LaBSE":
            continue
        cache = os.path.join(args.out, "rows_%s.csv" % short)
        if os.path.exists(cache):  # allows running the encoders in separate sessions
            log("Re-using %s" % cache)
            for src in ("%s-Centroid" % short, "%s-TextRank" % short):
                rows += load_reference_rows(cache, src)
            continue
        log("Loading %s (%s) ..." % (short, hf_id))
        model = SentenceTransformer(hf_id, device="cpu")
        cen, tr = make_scorers(model, prefix)
        new = []
        for name, scorer in (("%s-Centroid" % short, cen), ("%s-TextRank" % short, tr)):
            log("Running %s ..." % name)
            new += R.run_method(name, scorer, docs, ctx, args.out, log)
        with open(cache, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(new[0].keys())); w.writeheader(); w.writerows(new)
        rows += new
        del model, cen, tr
        gc.collect()
    if args.no_table:
        return

    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "scores_encoders_gujarati.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    stats = R.paired_stats(rows)
    with open(os.path.join(args.out, "stats_encoders_gujarati.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys())); w.writeheader(); w.writerows(stats)
    sig = {(s["baseline"], s["ratio"], s["annotator"]): s for s in stats if s["metric"] == "R1_F"}

    M = defaultdict(list)
    for r in rows:
        M[(r["method"], r["ratio"], r["annotator"])].append(r["R1_F"])
    M = {k: st.mean(v) for k, v in M.items()}
    present = {r["method"] for r in rows}
    methods = ["formality"] + [m for e in ENCODERS for m in ("%s-Centroid" % e[0], "%s-TextRank" % e[0]) if m in present]

    def cell(m, ratio, a):
        v = "%.3f" % M[(m, ratio, a)]
        s = sig.get((m, ratio, a))
        if s and s["p_two_sided_holm"] < 0.05:
            v += r"$^{\dagger}$" if s["mean_diff"] > 0 else r"$^{\ddagger}$"
        return v

    lines = [r"\begin{table*}[!t]", r"\centering", r"\footnotesize",
             r"\caption{Gujarati ROUGE-1 F1 of the embedding-based baselines with four multilingual sentence encoders. "
             r"$^{\dagger}$/$^{\ddagger}$: the formality method is significantly better/worse "
             r"(paired $t$-test, Holm-corrected over the encoder baselines, $p<0.05$)}\label{tab:encoders}",
             r"\begin{tabular}{l" + "c" * 8 + "}", r"\toprule",
             r" & \multicolumn{4}{c}{Annotator-1} & \multicolumn{4}{c}{Annotator-2} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             "Method & " + " & ".join(["20\\% & 30\\% & 40\\% & 50\\%"] * 2) + r" \\", r"\midrule"]
    for m in methods:
        label = "Formality (ours)" if m == "formality" else m
        lines.append(label + " & " + " & ".join(cell(m, r, a) for a in ("1", "2") for r in R.RATIOS) + r" \\")
        if m == "formality":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(args.out, "tables", "encoders_gu.tex"), "w", encoding="utf-8").write("\n".join(lines))
    log("\n".join(lines))


if __name__ == "__main__":
    main()
