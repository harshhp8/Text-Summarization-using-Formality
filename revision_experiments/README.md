# Formality-based extractive summarization: reproducible experiments

This folder contains the code that produces every table of the article

> H. Mehta, S. K. Bharti, N. Doshi, S. Varshney, M. Bajaj, I. Zaitsev,
> *Extractive Text Summarization Using Artificial Intelligence: Formality as a Catalyst in
> Low-Resource Language Processing*, SN Computer Science (under review).

It builds on the SOV formality measure of
H. Mehta, S. K. Bharti, N. Doshi, "Extractive Text Summarization Using Formality of Language",
*IEEE Open Journal of the Computer Society*, vol. 6, pp. 1414–1425, 2025, doi:10.1109/OJCS.2025.3600632.

## What `rerun_all.py` does

* Summarizes the 500-document Gujarati corpus and the English BBC News Summary corpus at compression
  ratios of 20, 30, 40 and 50 % with:
  **Formality** (ours), **Lead**, **TF-IDF**, **TextRank**, **LSA**, **LDA**, **Fuzzy logic**, **LaBSE-Centroid** and
  **LaBSE-TextRank**. All methods share one sentence splitter, tokenizer, stop-word list and selection rule.
* Scores every summary with word-level **ROUGE-1, ROUGE-2, ROUGE-L** (precision, recall, F1;
  summary-level union-LCS ROUGE-L) and sentence-level **BLEU-4** (Chen & Cherry smoothing 1).
* Compares the formality method with every baseline using paired *t*-tests and Wilcoxon signed-rank
  tests, with Holm correction, mean differences and 95 % confidence intervals.
* Runs an ablation of every term of the sentence score.
* Shows the effect of the "Adverse Effect of Matra" (AEM) on LSA/LDA by comparing an Indic-aware
  tokenizer with scikit-learn's default token pattern.
* Analyses the POS tagger: held-out accuracy, formal/deictic confusion, sentence-formality error,
  and the unknown-word rate on the corpus.
* Measures sentence-level inter-annotator agreement and the length of the human summaries.
* Tests whether annotator-selected sentences are more formal than non-selected ones (AUC).
* Validates the formality measure on manually POS-tagged legal and informal Gujarati text.

`encoder_robustness.py` repeats the two embedding baselines (Centroid, TextRank) on the Gujarati corpus with
four multilingual sentence encoders - LaBSE, IndicSBERT (`l3cube-pune/indic-sentence-similarity-sbert`),
multilingual E5 (`intfloat/multilingual-e5-base`) and BGE-M3 (`BAAI/bge-m3`) - and tests them against the
formality method on the same documents.

## Data

| Data | Source |
|---|---|
| Gujarati summarization corpus (500 posts, 2 annotators) | https://github.com/harshhp8/Gujarati-Textsummarization-corpus |
| Gujarati POS-tagged training corpus (`gu.pos`, BIS tagset) | https://github.com/harshhp8/Gujarati_POS_tagger_corpus |
| Gujarati stop words | https://github.com/harshhp8/Gujarati-Stop-words |
| Manually tagged court judgements / informal comments | https://github.com/harshhp8/Court-cases-POS-tagging, https://github.com/harshhp8/informal_text_pos_tagging |
| BBC News Summary (English) | Kaggle `pariza/bbc-news-summary`, or the CSV mirror `huggingface.co/datasets/gopalkalpande/bbc-news-summary` |

## Running

```bash
python -m venv venv && source venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','stopwords']]"

# Gujarati (about 1 h on a 4-core CPU)
python rerun_all.py --only gujarati \
  --corpus "Gujarati-Textsummarization-corpus/gujarati Text summarization dataset" \
  --gu-pos Gujarati_POS_tagger_corpus/gu.pos \
  --stopwords Gujarati-Stop-words/stop-words.txt \
  --legal-pos Court-cases-POS-tagging --informal-pos informal_text_pos_tagging/code.pos \
  --out results_gu

# English (about 1.5 h)
curl -L -o bbc.csv https://huggingface.co/datasets/gopalkalpande/bbc-news-summary/resolve/main/bbc-news-summary.csv
python rerun_all.py --only english --corpus . --bbc bbc.csv --out results_en

# Encoder robustness (after the Gujarati run; about 2 h on CPU, BGE-M3 needs ~3 GB RAM)
python encoder_robustness.py \
  --corpus "Gujarati-Textsummarization-corpus/gujarati Text summarization dataset" \
  --stopwords Gujarati-Stop-words/stop-words.txt \
  --scores results_gu/scores_gujarati.csv --out results_enc
```

Add `--reuse` to `rerun_all.py` to load `scores_<dataset>.csv` from `--out` and run only methods that are not
in it yet (useful after adding a method). `encoder_robustness.py` caches each encoder in `rows_<encoder>.csv`,
so the encoders can be run in separate sessions (`--encoders IndicSBERT,mE5-base --no-table`, then the rest).

Outputs are written to `--out`:

| File | Contents |
|---|---|
| `scores_<dataset>.csv` | Per-document scores |
| `stats_<dataset>.csv` | All statistical tests |
| `tables/*.tex` | LaTeX tables |
| `macros.tex` | Headline numbers |
| `report.txt` | Full log |
| `summaries/` | Every generated summary |
| `formality_examples_gu.tsv` | Per-sentence scores and annotator selections for example documents |

## Notes on differences from the original notebooks

This script supersedes the per-method notebooks in the related repositories:

* The notebooks passed a flat word list to `rouge_n_summary_level`, which flattens its input again,
  so they reported **character-level** ROUGE. This script computes word-level ROUGE.
* The notebooks loaded `gu.pos` with `nltk.corpus.indian`, which splits `word_N_NN` at the last
  underscore (word=`word_N`, tag=`NN`). This script parses the hierarchical BIS tags correctly and
  assigns tags to the formal or deictic side by tag prefix.
* Fuzzy logic: the Mamdani system of the notebook is rebuilt with `skfuzzy.control` (same inputs, `automf(3)`
  terms and rules). Its TF-IDF input is now computed over words (the notebook computed it over whole sentences,
  which made it almost constant), and noun/pronoun tags come from the corrected tagger. Sentences for which no
  rule fires get rank 0.
* Stop-word removal now matches whole words; the notebooks tested substrings of the stop-word file.
  TextRank uses the original Mihalcea–Tarau overlap similarity. LSA/LDA use an Indic-aware tokenizer
  and a fixed random seed.
