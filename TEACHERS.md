# Teachers

The student is `Qwen/Qwen2.5-7B-Instruct` (bf16, Apache-2.0). All training and evaluation text is written by an
open Qwen teacher from public-domain passages by the scripts in `scripts/`; nothing is hand-written and no text from a
closed model is used.

| File(s) | Teacher | How |
|---|---|---|
| `data/gen/book1_eval_cand.jsonl`, `book1_qa.jsonl`, `book1_notes.jsonl` | `Qwen/Qwen2.5-7B-Instruct`, bf16, plain transformers sampling | first version of `scripts/gen_teacher.py` (HF `generate`; temperature 0 for eval candidates, 0.7 / top-p 0.95 x3 for QA and notes) |
| `data/gen/book1_eval_verify.jsonl`, all `book2_*`, `book3_*`, `generic_prompts.jsonl` | `qwen2.5:7b-instruct` served by Ollama (Q4_K_M quantisation of the same model, Apache-2.0) | `src/clb/ollama_teacher.py`: `/api/chat`, `num_ctx` 4096, `seed + i` for sample i, one request at a time |

Why two teachers: the box froze several times under sustained bf16 generation, so the remaining generation moved to
the lighter Ollama path. Consequence to keep in mind: book 1 questions and notes come from the bf16 teacher, books 2
and 3 from the quantised one. Same family, same prompts, same filters (`scripts/build_data.py`), but not byte-identical
teachers. The answerability check of the book-1 eval questions (`eval_verify`) was run with the quantised teacher.

Teacher text is only used (a) as eval questions after filtering (answer must occur in the passage and be reproduced by
the teacher when it re-answers from the passage) and (b) as training data for the book being learned. Self-replay data
for the KL method is written by the student M1 itself (`scripts/gen_replay.py`), plus teacher-written generic prompts
answered by M1.
