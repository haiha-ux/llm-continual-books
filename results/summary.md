| run | book 1 acc % [95% CI] | book-1 answers kept (of M1-correct) % | book 2 acc % | book 3 acc % | GSM8K % | MMLU % | train GPU-h |
|---|---|---|---|---|---|---|---|
| m0 | 0.7 [0-2] (1/150) | 1.6 [0-5] (1/61) | 1.3 [0-3] (2/150) |  | 70.5 [64-76] (141/200) | 71.6 [68-75] (358/500) |  |
| m1 | 40.7 [33-48] (61/150) | 100.0 [100-100] (61/61) | 2.7 [1-5] (4/150) |  | 59.0 [52-66] (118/200) | 71.8 [68-76] (359/500) | 0.515 |
| naive_b2 | 6.0 [3-10] (9/150) | 13.1 [5-23] (8/61) | 36.7 [29-44] (55/150) |  | 54.5 [48-62] (109/200) | 69.2 [65-73] (346/500) | 0.203 |
| kl_b2 | 23.3 [17-30] (35/150) | 52.5 [39-64] (32/61) | 35.3 [28-43] (53/150) |  | 60.0 [54-66] (120/200) | 70.4 [66-74] (352/500) | 0.650 |
| expert_b2 | 17.3 [12-23] (26/150) | 42.6 [31-54] (26/61) | 31.3 [24-39] (47/150) |  | 59.0 [52-66] (118/200) | 71.4 [67-75] (357/500) | 0.159 |

## E1/E2: hard router on the insulated expert (expert_b2), see results/router_e1.json
Router = linear probe on M1 hidden states (mean over the last user turn). Positives: book-2 teacher questions. Threshold keeps 90% of book-2 dev positives.
Negatives usable without book-1 text (A_rw: M1-written book-1 questions restated by the teacher + generic): book-2 questions routed to the expert 93.3% (test), book-1 questions 30.0%, MMLU/GSM8K questions 0.6%.
With forced gate 0/1 from the true label (oracle): book-1 answers kept 61/61, book 2 47/150 (same as the soft-gate run). With the A_rw router: book-1 kept 45/61 (40/40 on questions routed away, 5/21 on questions routed to the expert), book 2 45/150.
Chat-format MMLU/GSM8K questions: outputs identical to M1 for 500/500 and 199/200. The GSM8K/MMLU few-shot skill evals of the router run were killed by the thermal guard and are not reported.
