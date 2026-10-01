# Stress-Test Pages

The 26 deliberately non-responsive webpages used to validate the responsive LLM
judge (Section 3.3.2 and Appendix A of the paper). They have no aligned sketches,
so they are evaluated with the sketch-independent five-dimension rubric
(`llm_judge/sketch_free_v1`) and have no Sketch Fidelity score.

Human ratings for these pages are in `human_study/annotations.jsonl`
(`"subset": "stress_test"`).

## Reproduce

```bash
# 1. Stage the pages as a run (sample ID = filename without extension)
python scripts/prepare_test_run.py --input stress_test/ --run-id test_adversarial

# 2. Evaluate them with the sketch-free judge and the accessibility checks
python scripts/run_eval.py --config configs/eval_stress_test.json
```

Results are written to `runs/test_adversarial/eval/eval_test_adversarial/`.
The run ID `test_adversarial` is what `scripts/export_human_alignment.py`
expects when pairing these pages with the human ratings.
