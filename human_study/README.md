# Human Study

`annotations.jsonl` holds the human ratings used to validate the responsive LLM
judge (Section 3.3.2, Tables 5–7, and Appendix A of the paper). Three annotators
independently rated 56 webpages on the 1–5 Likert rubric:

| Subset | `subset` value | Pages | Dimensions |
|---|---|---|---|
| Responsive-Bench | `responsive_bench` | 30 | 5 responsive dimensions + Sketch Fidelity |
| Stress test | `stress_test` | 26 | 5 responsive dimensions (no sketches) |

Annotator identities are replaced with the pseudonyms `annotator_A`,
`annotator_B`, and `annotator_C`.

## Format

One JSON object per line, one line per (webpage, annotator):

```json
{"subset": "responsive_bench", "run_id": "run_accessible_js", "eval_id": "eval_accessible_js",
 "sample_id": "0001", "annotator_id": "annotator_A",
 "scores": {"fluid_layout_and_container_adaptation": 5.0, "flexible_grid_reflow": 5.0,
            "responsive_media": 5.0, "intentional_content_visibility": 5.0,
            "proper_alignment_and_spacing": 4.0, "sketch_fidelity": 4.0}}
```

`run_id` and `eval_id` identify the evaluation run whose LLM judge scores the
ratings are compared against: the GPT-5.4 generations for Responsive-Bench
pages, and the `stress_test/` pages for the stress subset.

## Reproducing the agreement tables

With the evaluation outputs for `run_accessible_js/eval_accessible_js` and
`test_adversarial/eval_test_adversarial` under `runs/`:

```bash
python scripts/export_human_alignment.py      # Tables 5–7 and 11 (JSON under frontend/public/data/)
python tools/visualized_alignment.py          # human-vs-LLM scatter figure + CSV under results/
```
