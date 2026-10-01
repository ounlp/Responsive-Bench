# Usage

This guide covers configuration, the metrics, and how to reproduce each table and
figure in the paper. Setup is in the [README](README.md#quickstart). All commands
run from the repository root.

- [Pipeline overview](#pipeline-overview)
- [Generation](#generation)
- [Evaluation](#evaluation)
- [Metrics](#metrics)
- [Reproducing the paper](#reproducing-the-paper)
- [Results explorer](#results-explorer)
- [Providers](#providers)

## Pipeline overview

```
configs/run_*.json  ──► scripts/run_generate.py ──► runs/<run_id>/samples/<id>/output.html
configs/eval_*.json ──► scripts/run_eval.py     ──► runs/<run_id>/eval/<eval_id>/samples/<id>/metrics.json
                                                    runs/<run_id>/eval/<eval_id>/results.jsonl
```

Every run and eval writes a copy of its config, a log, and a manifest next to
its outputs. Both pipelines resume by default: samples with existing outputs are
skipped unless `overwrite` is `true`.

## Generation

```bash
python scripts/run_generate.py --config configs/run_accessible_js_gpt54.json
```

### Run configs

| Config | Model | Provider |
|---|---|---|
| `run_accessible_js_gpt54.json` | GPT-5.4 | `openai` |
| `run_accessible_js_claude.json` | Claude Sonnet 5 | `claude` |
| `run_accessible_js_gemini.json` | Gemini-3.5-Flash | `gemini` |
| `run_accessible_js_kimi_k3.json` | Kimi-K3 | `modal` |
| `run_accessible_js_qwen3_235b.json` | Qwen3-VL-235B-A22B-Instruct | `vllm` |
| `run_accessible_js_qwen35_397b.json` | Qwen3.5-397B-A17B-FP8 | `vllm` |
| `run_augment_from_sketches_all.json` | GPT-5.4, reference augmentation → `dataset/references/` | `openai` |
| `run_augment_screenshot_all_gpt5.4.json` | GPT-5.4, Appendix D.3 → `dataset/layout_similarity_study/html_and_screenshots/` | `openai` |
| `run_reconstruct_from_screenshot_all_gpt5.4.json` | GPT-5.4, Appendix D.3 → `dataset/layout_similarity_study/screenshots_only/` | `openai` |

The augmentation modes read each sample's original page (`source_html` in the
metadata), never the released reference. The three augmentation configs write
into `dataset/`. They use
`"overwrite": false`, so running them will not replace the released files. Set a
different `dataset_output_dir` to regenerate into a new directory.

### Modes

| `mode` | What it does | Templates (`src/responsive/prompts/templates/`) |
|---|---|---|
| `generate_a11y` | Benchmark task: both sketches → one responsive, accessible HTML page | `generate_a11y.j2` |
| `augment_from_sketch` | Two-stage reference augmentation (Section 3.2): analyze sketches + reference → JSON requirements, then augment the reference HTML | `augment_from_sketch_analyze.j2`, `augment_from_sketch_html.j2` |
| `augment_from_screenshot` | Augment the reference HTML from its three screenshots (Appendix D.3, "HTML + screenshots") | `augment_from_screenshot.j2` |
| `reconstruct_from_screenshot` | Rebuild the page from screenshots alone (Appendix D.3, "Screenshots only") | `reconstruct_from_screenshot.j2` |

### Run config fields

| Field | Default | Description |
|---|---|---|
| `run_id` | required | Output folder name under `output.root` |
| `provider`, `model` | required | See [Providers](#providers) |
| `prompt_template` | required | Template name without `.j2` |
| `mode` | `generate_a11y` | See above |
| `dataset_root` | `dataset` | Dataset directory |
| `ids` | all | Explicit list of sample IDs, e.g. `["0001", "0002"]` |
| `generation` | | `temperature`, `max_tokens` |
| `concurrency` | `8` | Samples generated in parallel |
| `output` | `{"root": "runs", "overwrite": false}` | |
| `retry` | 3 attempts | Exponential backoff on transient errors |
| `dataset_output_dir` | per mode | Target `dataset/` subdirectory for augmenting modes |

`configs/run.schema.json` is the full schema.

## Evaluation

```bash
python scripts/run_eval.py --config configs/eval_accessible_js.json
```

### Eval configs

| Config | Evaluates | Used for |
|---|---|---|
| `eval_accessible_js.json` | `run_accessible_js` (GPT-5.4) | Table 8; human study; Table 13 |
| `eval_accessible_js_{claude,gemini,kimi_k3,qwen3_235b,qwen35_397b}.json` | the matching run | Table 8 |
| `eval_accessible_js_structure.json` | `run_accessible_js` | Tables 9, 10, 14 (`layout_block_f1`, `layout_block_f1_rra`) |
| `eval_stress_test.json` | `test_adversarial` (the stress pages) | Tables 5–7, stress row of Table 8 |

The Table 8 configs run `layout_similarity`, `llm_judge_screenshots`,
`accessibility`, and the five `accessible_*` checks, all judged by GPT-5.4.

### Eval config fields

| Field | Default | Description |
|---|---|---|
| `eval_id`, `run_id` | required | Output goes to `runs/<run_id>/eval/<eval_id>/` |
| `metrics` | required | Metric names (see [Metrics](#metrics)) |
| `ids` | all | Subset of sample IDs |
| `judge.provider`, `judge.model` | | Judge used by `llm_judge_screenshots` and the LLM-judged `accessible_*` checks |
| `judge.prompt` | `llm_judge/unified_responsive_v1` | Judge template under `src/responsive/eval/prompts/`; use `llm_judge/sketch_free_v1` for pages without sketches |
| `judge.include_sketches` | `true` | Send both sketches to the judge |
| `judge.temperature`, `judge.max_tokens`, `judge.max_retries` | `0.3`, `1000`, `3` | The paper uses temperature `0.0` |
| `judge.response_format`, `judge.system_prompt`, `judge.extra_body` | | Provider-specific options (e.g. JSON mode, disabling Qwen3 thinking) |
| `layout_block_f1.iou_threshold` | `0.5` | Matching threshold for Appendix F |
| `concurrency.max_workers` | `1` | Samples evaluated in parallel |
| `output.overwrite`, `caching.skip_if_metrics_exist` | `false`, `true` | Resume behavior |

`configs/eval.schema.json` is the full schema.

## Metrics

All browser-based metrics render pages with Playwright (Chromium) at 375 px,
768 px, and 1280 px.

### Benchmark metrics

| Metric | Output | Paper |
|---|---|---|
| `layout_similarity` | Modified Layout Similarity: text blocks in the same column are merged, then each category's boxes are unioned and compared by IoU, weighted by the category's union area. Reported per viewport; the benchmark score is the mean of the three. | §3.3.1, App. D |
| `llm_judge_screenshots` | 1–5 scores and rationales for six dimensions: Fluid Layout, Grid Reflow, Responsive Media, Content Visibility, Alignment/Spacing, and Sketch Fidelity. Each is normalized as (s − 1)/4. | §3.3.2, App. C.1 |
| `accessibility` | axe-core violations at the mobile viewport, counted by rule | §3.3.3, App. B |
| `accessible_form_labels` | Failing eligible form controls; added to the A11y Violations count | §3.3.3 |
| `accessible_landmarks` | LLM judge over the HTML and ARIA trees at all three viewports | App. C.5 |
| `accessible_active_nav_item` | LLM judge over `aria-current` usage in the mobile ARIA tree | App. C.2 |
| `accessible_hamburger_menu` | Applicability precheck, then LLM judge over the ARIA trees before and after clicking the toggle | App. C.3–C.4 |
| `accessible_no_keyboard_trap` | Deterministic Tab / Shift-Tab / Escape simulation with rule-based 1–5 scoring (no LLM) | App. B |

A11y Judge is the mean normalized score of the landmark, active-nav, hamburger
(when applicable), and keyboard checks.

### Structural diagnostics

| Metric | Output | Paper |
|---|---|---|
| `layout_block_f1_rra` | Rank-based count consistency: per-category monotonic alignment of rank-normalized boxes (τ = γ = 0.15) → precision, recall, F1 | §4.2, App. E |
| `layout_block_f1` | Element-level correspondence: greedy one-to-one matching at IoU ≥ 0.5 → precision, recall, F1 | App. F |

### Optional automated responsiveness checks

These rule-based checks are not reported in the paper but remain available.
Add them to an eval config's `metrics` list:
`viewport_meta_tag`, `heading_order`, `horizontal_scroll`, `tap_targets`,
`line_spacing`, `responsive_media`, `relative_units`, `computed_font_sizes`,
`rotation_friendly`, `text_media_cutoff_overlap`, `text_enlargement`.

To score a flat directory of HTML files without setting up a run:

```bash
python scripts/eval_dataset.py --input-dir path/to/html --metrics horizontal_scroll tap_targets
```

## Reproducing the paper

Items marked † need the evaluation outputs under `runs/`, which you either
produce with the commands above or download.

### Benchmark results (Table 8) †

```bash
for m in gpt54 claude gemini kimi_k3 qwen3_235b qwen35_397b; do
  python scripts/run_generate.py --config configs/run_accessible_js_${m}.json
done
for e in "" _claude _gemini _kimi_k3 _qwen3_235b _qwen35_397b; do
  python scripts/run_eval.py --config configs/eval_accessible_js${e}.json
done
python scripts/build_runs_data.py --no-copy-runs   # writes frontend/public/data/runs.json
```

`runs.json` holds the per-run leaderboard columns. Its Responsive Score
averages the normalized scores of every judge dimension in a sample's
`score_normalized`, including Sketch Fidelity.

### Stress test (Table 8 stress row; Tables 5–7) †

```bash
python scripts/prepare_test_run.py --input stress_test/ --run-id test_adversarial
python scripts/run_eval.py --config configs/eval_stress_test.json
```

### Human alignment (Tables 5–7, 11; alignment figure) †

```bash
python scripts/export_human_alignment.py   # alignment_{benchmark,stress}.json, inter_annotator_*.json
python tools/visualized_alignment.py        # results/human_llm_alignment_scatter.{pdf,svg,png} + CSV
```

Human ratings come from `human_study/annotations.jsonl`. The judge scores come
from `runs/run_accessible_js/eval/eval_accessible_js/` and
`runs/test_adversarial/eval/eval_test_adversarial/`.

### Structural analysis (Tables 9, 10, 13, 14) †

```bash
python scripts/run_eval.py --config configs/eval_accessible_js_structure.json
E=runs/run_accessible_js/eval/eval_accessible_js_block_f1

python scripts/aggregate_layout_block_f1.py $E --metric-key layout_block_f1_rra   # Table 9
python scripts/layout_block_f1_rra_category_std.py $E                           # Table 10
python scripts/aggregate_layout_block_f1.py $E --metric-key layout_block_f1       # Table 14
python scripts/layout_similarity_category_std.py runs/run_accessible_js/eval/eval_accessible_js  # Table 13
```

### Dataset composition (Table 3)

```bash
python tools/annotate_layout_objects.py dataset/references
```

### Original vs. modified Layout Similarity (Table 12, Figure 2)

```bash
python tools/check_metrics.py                       # both variants on both D.3 settings → results/metrics/
python tools/plot_layout_similarity_examples.py     # Figure 2 → results/layout_similarity_examples.pdf
```

`tools/plot_layout_comparison.py` and `tools/plot_text_merge.py` draw the
explanatory diagrams of the two modifications.

### Reference augmentation (Section 3.2)

```bash
python scripts/run_generate.py --config configs/run_augment_from_sketches_all.json
```

### Dataset validation

```bash
PYTHONPATH=src python -m responsive.dataset.loader validate --root dataset
```

## Results explorer

`frontend/` is a static React app for browsing runs, per-sample renders, metric
results, and the human-alignment tables.

```bash
python scripts/build_runs_data.py            # copies runs/ into frontend/public and writes data/runs.json
python scripts/export_human_alignment.py     # alignment tables shown on the home page
cd frontend && npm install && npm run dev
```

Use `npm run build` for a static build in `frontend/dist/`.

The full explorer bundles every run's screenshots and artifacts (several GB).
For a small public site, build with `VITE_LEADERBOARD_ONLY=1`: only the
leaderboard, human-alignment, and score-distribution pages are included, so
`frontend/public/data/` needs just `runs.json` and the alignment JSON files.

```bash
cd frontend && VITE_LEADERBOARD_ONLY=1 npm run build
```

The hosted leaderboard is built this way: `vercel.json` copies
`frontend/leaderboard_data/*.json` (the paper's runs plus the human-alignment
tables) into `public/data/` and runs the leaderboard-only build on every push.
To refresh it, regenerate the JSON files with `build_runs_data.py` and
`export_human_alignment.py`, then copy `runs.json`, `annotator_scores.json`,
`human_alignment.json`, `alignment_*.json`, and `inter_annotator_*.json` into
`frontend/leaderboard_data/`.

## Providers

Set credentials in `.env` (see `.env.example`).

| `provider` | Backend | Environment |
|---|---|---|
| `openai` | OpenAI API | `OPENAI_API_KEY` |
| `claude` | Anthropic API | `ANTHROPIC_API_KEY` |
| `gemini` | Google Gen AI API | `GEMINI_API_KEY` |
| `vllm` | Any OpenAI-compatible server (e.g. self-hosted vLLM) | `VLLM_BASE_URL` |
| `modal` | OpenAI-compatible endpoint on modal.com | `MODAL_ENDPOINT`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` |
| `hf` | In-process Hugging Face `transformers` model | See `src/responsive/providers/hf_provider.py` |
| `fake` | Returns a placeholder page; for smoke tests without API calls | none |

Each provider can serve as the generator (`provider` in a run config) or as the
judge (`judge.provider` in an eval config).
