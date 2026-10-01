# Responsive-Bench

**Evaluating Responsive and Accessibility-Aware Sketch-to-Code Generation**

🏆 **Leaderboard:** [responsive-bench.vercel.app](https://responsive-bench.vercel.app)

Sketch-to-code benchmarks usually check whether a generated webpage matches one
rendered target. Responsive-Bench instead asks a model to turn a **pair of
low-fidelity sketches (desktop + mobile)** into **one self-contained HTML page**,
then tests that page at mobile (375 px), tablet (768 px), and desktop (1280 px)
widths. The tablet width is deliberately unannotated: it reveals whether the
implementation defines a coherent transition or just two disconnected endpoint
layouts.

Evaluation reports three families of evidence separately, so progress on one
cannot hide a regression in another:

| Family | Metric | What it measures |
|---|---|---|
| Visual reconstruction | **Layout Similarity** | Geometric-union IoU of semantic layout elements against the reference, averaged over the three viewports |
| Responsive behavior | **Responsive Score** | LLM judge (GPT-5.4) scoring cross-viewport behavior on a 1–5 rubric, normalized to [0, 1] |
| Accessibility | **A11y Violations**, **A11y Judge** | axe-core violations plus interaction-aware checks: landmarks, active-navigation state, hamburger-menu semantics, keyboard focus, form labels |

## Dataset

200 webpages derived from [Sketch2Code](https://huggingface.co/datasets/SALT-NLP/Sketch2Code).
A web developer screened all 484 Sketch2Code pages, retaining the 350 with
meaningful responsive behavior; 200 of these were annotated.

| | |
|---|---|
| Desktop sketches | 200, original Sketch2Code sketches (1280 px layout) |
| Mobile sketches | 200, drawn for Responsive-Bench (375 px layout) |
| Reference HTML | 200, the Sketch2Code HTML augmented with landmarks, accessible names, ARIA state, and keyboard behavior |
| Annotated layout objects | 4,358 |
| Stress-test pages | 26 deliberately non-responsive pages for judge validation |
| Human ratings | 3 annotators × 56 pages, pseudonymized |

```
dataset/
  sketches/{id}-d-v2.png, {id}-m-v2.png   input sketches (desktop, mobile)
  references/{id}.html                    benchmark reference HTML
  metadata/{id}.json                      per-sample metadata
  source_html/{id}.html                   original Sketch2Code HTML the references were built from
  layout_similarity_study/                Appendix D.3 only; not needed to run the benchmark
stress_test/                              26 stress-test pages
human_study/annotations.jsonl             human ratings for the 56 validation pages
```

See [dataset/DATASET_SCHEMA.md](dataset/DATASET_SCHEMA.md) for the full schema and
[ANNOTATOR_GUIDE.md](ANNOTATOR_GUIDE.md) for the sketching protocol.

## Results

GPT-5.4 judges every run. Numbers are from Table 8 of the paper; the
[online leaderboard](https://responsive-bench.vercel.app) also shows the
human-alignment results.

| Model | Responsive Score ↑ | Layout Sim. ↑ | A11y Viol. ↓ | A11y Judge ↑ |
|---|---:|---:|---:|---:|
| GPT-5.4 | **0.860** | 0.302 | 0.4 | 0.465 |
| Kimi-K3 | 0.826 | **0.362** | **0.1** | **0.501** |
| Claude Sonnet 5 | 0.799 | 0.304 | 0.3 | 0.482 |
| Gemini-3.5-Flash | 0.777 | 0.311 | 0.6 | 0.448 |
| Qwen3-VL-235B-A22B | 0.757 | 0.223 | 1.3 | 0.364 |
| Qwen3.5-397B-A17B-FP8 | 0.753 | 0.292 | 0.9 | 0.419 |

No model leads on all three families.

## Quickstart

Requires Python ≥ 3.10 and Node.js (for axe-core).

```bash
pip install -r requirements.txt
python -m playwright install chromium
npm install                      # axe-core, injected by the accessibility metric
cp .env.example .env             # then fill in the API keys you need
```

Generate pages with a model, then evaluate them:

```bash
python scripts/run_generate.py --config configs/run_accessible_js_gpt54.json
python scripts/run_eval.py     --config configs/eval_accessible_js.json
```

Generated pages land in `runs/<run_id>/samples/<id>/output.html` and per-sample
scores in `runs/<run_id>/eval/<eval_id>/samples/<id>/metrics.json`.

To try the pipeline without API keys, copy a run config and set
`"provider": "fake"`.

[USAGE.md](USAGE.md) covers every config, the metrics, and how to reproduce each
table and figure in the paper.

## Repository layout

```
configs/             Run and eval configs for every experiment in the paper, plus JSON schemas
dataset/             Benchmark data (see above)
stress_test/         Stress-test pages for judge validation
human_study/         Pseudonymized human ratings
src/responsive/
  dataset/           Dataset loader and validator
  providers/         Model backends: openai, claude, gemini, vllm, hf, modal, fake
  prompts/           Generation and reference-augmentation prompt templates
  runner/            Generation runner
  eval/              Evaluation orchestrator, metrics, judge client, and judge prompts
scripts/             CLI entry points and analysis scripts
tools/               Layout Similarity comparison and paper figures
frontend/            Static results explorer (React + Vite)
```

## License

Code is released under the [MIT License](LICENSE). Data (`dataset/`,
`stress_test/`, `human_study/`) is released under
[ODC-By 1.0](dataset/LICENSE), following Sketch2Code. Please attribute
Sketch2Code and Design2Code as described in [dataset/LICENSE](dataset/LICENSE).

## Citation

```bibtex
@misc{responsivebench2026,
  title  = {Responsive-Bench: Evaluating Responsive and Accessibility-Aware Sketch-to-Code Generation},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

Please also cite Sketch2Code:

```bibtex
@inproceedings{li2025sketch2code,
  title     = {Sketch2Code: Evaluating Vision-Language Models for Interactive Web Design Prototyping},
  author    = {Li, Ryan and Zhang, Yanzhe and Yang, Diyi},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2025}
}
```
