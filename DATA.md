# Data

## Books (public domain, Project Gutenberg)

| Book | Gutenberg # | Title | Author | sha256 of `data/raw/pg<#>.txt` | slice used |
|---|---|---|---|---|---|
| 1 | 43608 | The Great Frozen Sea (voyage of the Alert, 1875-76) | Albert Hastings Markham | `0664882f181bbe104b6498669d8e6336b078e7ed927ed9124894b4aefa4e5c15` | 160,000 chars, 74 passages |
| 2 | 79056 | The German Arctic Expedition of 1869-70 and the wreck of the Hansa | Karl Koldewey | `837c14ec6f694074f6e3ded765e6d5ca66434de068e504a1d062f745eb328778` | 160,000 chars, 60 passages |
| 3 | 51910 | A Voyage to the Arctic in the Whaler Aurora | David Moore Lindsay | `c15a7951119f018d05669743cd263424ca6ff4f1798f64b4a045e72eb8b8f0a5` | 160,000 chars, 67 passages |

`scripts/prep_books.py` strips the Gutenberg header/footer and cuts a fixed 160k-character slice (offsets in the
script) into paragraph-aligned passages of about 1,800 characters (`data/books/bookN.jsonl`). The raw files are
committed so the slices are reproducible. The base model is asked each book's questions before training
(row `m0` in the results) to show it does not already know them.

## Skill subsets (own small subsets, fixed seed 0)

`scripts/prep_skills.py`: 200 GSM8K test problems (5-shot, strict `####` answer match, `openai/gsm8k`, MIT) and 500
MMLU test questions (5-shot, answer-letter logit, `cais/mmlu`, MIT). These are our own subsets, not the full
benchmarks, so absolute numbers are not comparable to published ones; only before/after differences are meaningful.

## Teacher-written data (`data/gen/`)

Written by an open Qwen teacher from the passages; nothing is hand-written. Which teacher made which file is in
[TEACHERS.md](TEACHERS.md). Per book: `eval_cand` (short-answer questions with answers copied from the passage),
`eval_verify` (the teacher re-answers each question from the passage), `qa` (training question-answer pairs),
`notes` (standalone factual statements for training). Generation is resumable (`*.partial.jsonl`, not committed).
Books 2 and 3 use 10 questions per request and 2 note samples per passage because requests are kept short for thermal
reasons; book 1 was generated earlier with 12 questions per request and 3 note samples.

## Filtering (`scripts/build_data.py`, `src/clb/dataprep.py`) -> `data/final/bookN/`

* Eval question kept only if: answer <= 6 words and occurs in the passage, the teacher re-answers it with the same
  string, the question names at least two specific anchors (so it is answerable without the passage), contains no
  "the passage / the text / the author ..." wording and no bare pronoun subject, no duplicate, at most 5 per passage.
* Kept questions are split into a dev set (40) and a test set (150), round-robin over passages.
* Training QA and notes that share an 8-gram with any dev/test question or answer are dropped; the build asserts the
  overlap count is 0. Counts per book, including the overlap check, are in `data/final/report.json`.

Scoring (`src/clb/scoring.py`): normalised string containment of the gold answer in the model output, greedy decoding,
32 new tokens. Confidence intervals: 95% bootstrap over questions.
