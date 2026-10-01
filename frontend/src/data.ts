const base = import.meta.env.BASE_URL;

// When base is relative (e.g. "./"), use origin-relative paths ("/data/...") so fetches
// work from any route (e.g. /runs/foo). Otherwise use base as prefix (e.g. /ResponsiveBench/).
const urlPrefix =
  base === "./" || base === "." ? "" : base.endsWith("/") ? base : base + "/";

/** Build with VITE_LEADERBOARD_ONLY=1 to ship only the leaderboard and alignment pages
 * (no per-run / per-sample drill-down, which needs the full runs/ data). */
export const LEADERBOARD_ONLY = import.meta.env.VITE_LEADERBOARD_ONLY === "1";

export function dataUrl(path: string): string {
  const p = path.startsWith("/") ? path.slice(1) : path;
  const dataPath = p.startsWith("data/") ? p : "data/" + p;
  return (urlPrefix || "/") + dataPath;
}

export function assetUrl(relativePath: string | null): string | null {
  if (!relativePath) return null;
  const p = relativePath.startsWith("/") ? relativePath.slice(1) : relativePath;
  return (urlPrefix || "/") + p;
}

export async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(dataUrl(path));
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

/** Plain text under `data/` or `run-assets/` (same base URL rules as JSON/assets). */
export async function fetchText(path: string): Promise<string> {
  const url = path.startsWith("run-assets/")
    ? assetUrl(path)!
    : dataUrl(path);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.text();
}

export interface EvalLeaderboardRow {
  run_id: string;
  eval_id: string;
  /** Provider used for generation (e.g. "openai", "claude"), from the run's saved config.json. */
  generation_provider: string | null;
  /** Model used for generation (e.g. "gpt-5.4", "claude-sonnet-5"), from the run's saved config.json. */
  generation_model: string | null;
  /** Provider used for the LLM judge, from the eval's saved config.json. */
  judge_provider: string | null;
  /** Model used for the LLM judge, from the eval's saved config.json. */
  judge_model: string | null;
  /**
   * Mean LLM judge score in [0, 1] over samples with valid judge output (see build_runs_data.py).
   * Null when the eval didn't run llm_judge_screenshots at all (e.g. layout_similarity-only evals).
   */
  llm_judge_avg_normalized: number | null;
  llm_judge_sample_count: number;
  /** Mean mobile (375px) layout similarity weighted IoU. */
  layout_similarity_avg_mobile_iou: number | null;
  /** Mean tablet (768px) layout similarity weighted IoU. */
  layout_similarity_avg_tablet_iou: number | null;
  /** Mean desktop (1280px) layout similarity weighted IoU. */
  layout_similarity_avg_desktop_iou: number | null;
  /** Mean per-sample average of mobile, tablet, and desktop IoU. */
  layout_similarity_avg_iou: number | null;
  layout_similarity_sample_count: number;
  /** Mean per-sample total violations + fails across 11 responsiveness tests. Lower is better. */
  avg_responsiveness_violations: number | null;
  /** Mean per-sample axe-core violations + accessible_form_labels fails. Lower is better. */
  avg_accessibility_violations: number | null;
  /** Mean normalized [0,1] LLM judge score for landmarks, hamburger, active nav, keyboard trap. Higher is better. */
  avg_accessible_judge_score: number | null;
}

export interface RunsListResponse {
  /** Back-compat key name: entries are eval-level rows. */
  runs: EvalLeaderboardRow[];
}

export interface RunDetailResponse {
  samples: {
    id: string;
    reference_path: string | null;
    generated_path: string | null;
    sketch_desktop_path?: string | null;
    sketch_mobile_path?: string | null;
  }[];
  evals: { eval_id: string }[];
}

export interface EvalSamplesResponse {
  samples: { id: string; render_mobile_path: string | null }[];
}

export interface SampleMetricsResponse {
  sample_id: string;
  metrics: Record<string, unknown>;
  render_desktop_path: string | null;
  render_mobile_path: string | null;
  render_tablet_path: string | null;
  reference_path?: string | null;
  sketch_desktop_path?: string | null;
  sketch_mobile_path?: string | null;
}

export interface DimensionAlignment {
  key: string;
  label: string;
  n: number;
  mse: number | null;
  mae: number | null;
  acc: number | null;
  acc1: number | null;
  pearson: number | null;
  spearman: number | null;
  kendall: number | null;
  human_mean: number | null;
  human_std: number | null;
  llm_mean: number | null;
  llm_std: number | null;
  /** 95% CI for this dimension's Pearson correlation via webpage-level clustered bootstrap. */
  pearson_ci_low: number | null;
  pearson_ci_high: number | null;
  /** 95% CI for this dimension's Spearman correlation via webpage-level clustered bootstrap. */
  spearman_ci_low: number | null;
  spearman_ci_high: number | null;
}

export interface HumanAlignmentData {
  generated_at: string;
  /** Human-readable description of which judge(s) produced this variant (see export_human_alignment.py). */
  label?: string;
  /** run_id -> eval_id overrides used to select a non-default judge for that run's samples. */
  eval_overrides?: Record<string, string>;
  /** run_id -> judge model actually used when this variant was computed. */
  judge_models?: Record<string, string | null>;
  n_annotations: number;
  n_matched_samples: number;
  dimensions: DimensionAlignment[];
  /** Pools every (human, LLM) pair across all dimensions before computing metrics and CIs. */
  overall: Omit<DimensionAlignment, "key"> & { label: string };
}

export async function fetchHumanAlignment(): Promise<HumanAlignmentData | null> {
  try {
    return await fetchJson<HumanAlignmentData>("human_alignment.json");
  } catch {
    return null;
  }
}

/** The 30 Responsive-Bench-sampled webpages (full 6-dim rubric incl. Sketch Fidelity). */
export async function fetchAlignmentBenchmark(): Promise<HumanAlignmentData | null> {
  try {
    return await fetchJson<HumanAlignmentData>("alignment_benchmark.json");
  } catch {
    return null;
  }
}

/** The 26 curated stress-test webpages (sketch-independent 5-dim rubric). */
export async function fetchAlignmentStress(): Promise<HumanAlignmentData | null> {
  try {
    return await fetchJson<HumanAlignmentData>("alignment_stress.json");
  } catch {
    return null;
  }
}

export interface InterAnnotatorDimension {
  key: string;
  label: string;
  n: number;
  mse: number | null;
  mae: number | null;
  acc: number | null;
  acc1: number | null;
  pearson: number | null;
  spearman: number | null;
  kendall: number | null;
  /** 95% CI for this dimension's pair-averaged Pearson correlation via webpage-level clustered bootstrap. */
  pearson_ci_low: number | null;
  pearson_ci_high: number | null;
  /** 95% CI for this dimension's pair-averaged Spearman correlation via webpage-level clustered bootstrap. */
  spearman_ci_low: number | null;
  spearman_ci_high: number | null;
}

/** Human--human (inter-annotator) agreement, averaged across every annotator
 * pair. Same subset split as HumanAlignmentData, but each pair compares raw
 * unaveraged scores against each other rather than against an LLM judge. */
export interface InterAnnotatorData {
  generated_at: string;
  label: string;
  annotator_ids: string[];
  annotator_pairs: [string, string][];
  n_webpages: number;
  dimensions: InterAnnotatorDimension[];
  overall: Omit<InterAnnotatorDimension, "key">;
}

async function fetchInterAnnotator(file: string): Promise<InterAnnotatorData | null> {
  try {
    return await fetchJson<InterAnnotatorData>(file);
  } catch {
    return null;
  }
}

export const fetchInterAnnotatorBenchmark = () => fetchInterAnnotator("inter_annotator_benchmark.json");
export const fetchInterAnnotatorStress = () => fetchInterAnnotator("inter_annotator_stress.json");
export const fetchInterAnnotatorCombined = () => fetchInterAnnotator("inter_annotator_combined.json");

/** One selectable judge-comparison variant, as listed in human_alignment_index.json. */
export interface HumanAlignmentIndexEntry {
  file: string;
  label: string;
  judge_models: Record<string, string | null> | null;
  generated_at: string | null;
  n_matched_samples: number | null;
}

export interface HumanAlignmentIndex {
  generated_at: string;
  variants: HumanAlignmentIndexEntry[];
}

/** List of available human-vs-judge alignment variants (see export_human_alignment.py --eval-override). */
export async function fetchHumanAlignmentIndex(): Promise<HumanAlignmentIndex | null> {
  try {
    return await fetchJson<HumanAlignmentIndex>("human_alignment_index.json");
  } catch {
    return null;
  }
}

/** Fetch a specific alignment variant by its filename (from HumanAlignmentIndexEntry.file). */
export async function fetchHumanAlignmentByFile(file: string): Promise<HumanAlignmentData> {
  return fetchJson<HumanAlignmentData>(file);
}

export interface AnnotatorDimensionData {
  key: string;
  label: string;
  annotator_averages: Record<string, number>;
  samples: { sample_id: string; scores: Record<string, number> }[];
}

export interface AnnotatorScoresData {
  generated_at: string;
  annotators: string[];
  dimensions: AnnotatorDimensionData[];
}

export async function fetchAnnotatorScores(): Promise<AnnotatorScoresData | null> {
  try {
    return await fetchJson<AnnotatorScoresData>("annotator_scores.json");
  } catch {
    return null;
  }
}
