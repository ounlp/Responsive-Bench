#!/usr/bin/env python3
"""
Build-time script for the React Runs Explorer.

Scans the runs/ directory and:
1. Writes JSON index files to frontend/public/data/ (only run folders that contain an eval/ subdir)
2. Copies those same runs to frontend/public/run-assets/ so images are deployed as static assets (run-assets avoids conflicting with SPA route /runs/*).
3. Copies reference.png and the two sketches (desktop, mobile) per sample into
   frontend/public/data/runs/{run_id}/samples/{sample_id}/ so the explorer can
   show them without referencing the original run/dataset paths.

Sketches are flattened onto white and copied from (in order): sample's sketch_paths.json
if present and files exist, else --dataset-root/sketches/{id}-d-v2.png and {id}-m-v2.png
when --dataset-root is set.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:
    Image = None


def _rmtree_robust(path: Path, attempts: int = 5) -> None:
    """shutil.rmtree, retrying past macOS's Finder/.DS_Store race.

    Finder/Spotlight can drop a fresh .DS_Store into a directory the instant
    rmtree empties it, right before rmtree's os.rmdir() on that directory —
    which then fails with ENOTEMPTY even though nothing meaningful is left.
    Retrying (optionally after clearing stray .DS_Store files) resolves it.
    """
    last_error: OSError | None = None
    for _ in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError as e:
            last_error = e
            for ds_store in path.rglob(".DS_Store"):
                ds_store.unlink(missing_ok=True)
    if path.exists():
        raise last_error


def _flatten_png_to_white(src: Path, dst: Path) -> None:
    """Flatten a possibly-transparent PNG onto white for visibility."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Image is None:
        # Best-effort fallback: copy the file as-is.
        shutil.copy2(src, dst)
        return
    img = Image.open(src).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    out = bg.convert("RGB")
    out.save(dst, format="PNG")


def _llm_judge_sample_normalized(metrics_data: dict[str, Any]) -> float | None:
    """Single-sample LLM judge score in [0, 1] for leaderboard aggregation.

    Prefer per-dimension aggregates (score_normalized, then scores) over score_overall.
    The unified judge often returns an overall_score that does not match the mean of
    per-dimension Likert scores (e.g. overall 5 with several dimensions at 4).
    """
    metrics = metrics_data.get("metrics") or {}
    lj = metrics.get("llm_judge_screenshots")
    if not isinstance(lj, dict) or lj.get("status") != "success":
        return None

    sn = lj.get("score_normalized")
    if isinstance(sn, dict):
        vals = [
            float(v)
            for v in sn.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if vals:
            return sum(vals) / len(vals)

    scores = lj.get("scores")
    if isinstance(scores, dict):
        norms: list[float] = []
        for v in scores.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    s = max(1, min(5, int(v)))
                except (TypeError, ValueError):
                    continue
                norms.append((s - 1) / 4.0)
            elif isinstance(v, dict):
                raw = v.get("score")
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    try:
                        s = max(1, min(5, int(raw)))
                    except (TypeError, ValueError):
                        continue
                    norms.append((s - 1) / 4.0)
        if norms:
            return sum(norms) / len(norms)

    overall = lj.get("score_overall")
    if isinstance(overall, (int, float)) and not isinstance(overall, bool):
        try:
            o = float(overall)
        except (TypeError, ValueError):
            o = None
        if o is not None:
            o = max(1.0, min(5.0, o))
            return (o - 1.0) / 4.0

    return None


def _pick_llm_leaderboard_eval(eval_norms: dict[str, list[float]]) -> tuple[str | None, list[float]]:
    """Choose eval with the most valid scores; tie-break lexicographically smaller eval_id."""
    best_id: str | None = None
    best_count = 0
    for eid in sorted(eval_norms.keys()):
        c = len(eval_norms[eid])
        if c > best_count:
            best_count = c
            best_id = eid
    if best_id is None or best_count == 0:
        return None, []
    return best_id, eval_norms[best_id]


LAYOUT_SIMILARITY_VIEWPORTS = ("mobile", "tablet", "desktop")


def _layout_similarity_viewport_iou(metrics_data: dict[str, Any], viewport: str) -> float | None:
    """weighted_iou for one viewport, from the current {"viewports": {"mobile": {...}, ...}} schema."""
    metrics = metrics_data.get("metrics") or {}
    ls = metrics.get("layout_similarity")
    if not isinstance(ls, dict):
        return None
    viewports = ls.get("viewports")
    if not isinstance(viewports, dict):
        return None
    vp = viewports.get(viewport)
    if not isinstance(vp, dict):
        return None
    raw = vp.get("weighted_iou")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    return None


def _responsiveness_violations_count(metrics_data: dict[str, Any]) -> int | None:
    """Total violations + fails across the 11 responsiveness tests for one sample."""
    metrics = metrics_data.get("metrics") or {}
    total = 0
    found = False

    def _int_violation(m: dict[str, Any]) -> int:
        if m.get("status") == "error":
            return 0
        for key in ("violations_count", "violation_count", "total_violations"):
            v = m.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        violations = m.get("violations")
        if isinstance(violations, list):
            return len(violations)
        return 0 if m.get("passed") else 1

    for metric_key in (
        "horizontal_scroll",
        "tap_targets",
        "line_spacing",
        "viewport_meta_tag",
        "responsive_media",
        "relative_units",
        "computed_font_sizes",
        "rotation_friendly",
        "text_media_cutoff_overlap",
        "heading_order",
        "text_enlargement",
    ):
        m = metrics.get(metric_key)
        if not isinstance(m, dict):
            continue
        found = True
        if metric_key == "horizontal_scroll":
            total += 1 if m.get("horizontal_scroll_needed") is True else 0
        else:
            total += _int_violation(m)

    return total if found else None


def _accessibility_violations_count(metrics_data: dict[str, Any]) -> int | None:
    """Total axe-core violations + accessible_form_labels fails for one sample."""
    metrics = metrics_data.get("metrics") or {}
    total = 0
    found = False

    axe = metrics.get("accessibility")
    if isinstance(axe, dict) and axe.get("status") != "error":
        found = True
        if axe.get("passed"):
            pass  # 0 violations
        else:
            count = axe.get("total_violations_count")
            if isinstance(count, (int, float)) and not isinstance(count, bool):
                total += int(count)
            else:
                total += 1

    form = metrics.get("accessible_form_labels")
    if isinstance(form, dict) and form.get("status") != "error":
        vr = form.get("viewport_results")
        if isinstance(vr, dict):
            found = True
            for vp_data in vr.values():
                if not isinstance(vp_data, dict):
                    continue
                checked = vp_data.get("checked")
                passed = vp_data.get("passed")
                if isinstance(checked, (int, float)) and isinstance(passed, (int, float)):
                    total += max(0, int(checked) - int(passed))

    return total if found else None


def _accessible_judge_score_normalized(metrics_data: dict[str, Any]) -> float | None:
    """Mean normalized [0,1] score for LLM-judged accessibility metrics for one sample.

    Includes: accessible_landmarks, accessible_hamburger_menu (skip if not_applicable),
    accessible_active_nav_item, accessible_no_keyboard_trap (average across viewports).
    """
    metrics = metrics_data.get("metrics") or {}
    norms: list[float] = []

    for key in ("accessible_landmarks", "accessible_active_nav_item"):
        m = metrics.get(key)
        if not isinstance(m, dict) or m.get("status") == "error":
            continue
        score = m.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            norms.append((max(1, min(5, int(score))) - 1) / 4.0)

    hamburger = metrics.get("accessible_hamburger_menu")
    if isinstance(hamburger, dict) and hamburger.get("status") != "error":
        if hamburger.get("reason") != "not_applicable":
            score = hamburger.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                norms.append((max(1, min(5, int(score))) - 1) / 4.0)

    keyboard = metrics.get("accessible_no_keyboard_trap")
    if isinstance(keyboard, dict) and keyboard.get("status") != "error":
        vr = keyboard.get("viewport_results")
        if isinstance(vr, dict):
            vp_norms: list[float] = []
            for vp_data in vr.values():
                if not isinstance(vp_data, dict):
                    continue
                score = vp_data.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    vp_norms.append((max(1, min(5, int(score))) - 1) / 4.0)
            if vp_norms:
                norms.append(sum(vp_norms) / len(vp_norms))

    return sum(norms) / len(norms) if norms else None


def _read_run_generation_model(run_dir: Path) -> tuple[str | None, str | None]:
    """Read (provider, model) used for generation from the run's saved config.json."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None, None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return None, None
    return config.get("provider"), config.get("model")


def _read_eval_judge_model(eval_dir: Path) -> tuple[str | None, str | None]:
    """Read (provider, model) used for the LLM judge from the eval's saved config.json."""
    config_path = eval_dir / "config.json"
    if not config_path.exists():
        return None, None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return None, None
    judge = config.get("judge") or {}
    return judge.get("provider"), judge.get("model")


def _copy_sketches_and_reference(
    run_id: str,
    sample_id: str,
    sample_dir: Path,
    data_samples_dir: Path,
    dataset_root: Path | None,
) -> tuple[bool, bool, bool]:
    """Copy reference.png and sketch_desktop_for_judge.png, sketch_mobile_for_judge.png into data_samples_dir.
    Sketches: prefer dataset_root/sketches when set, else use sample's sketch_paths.json.
    Returns (reference_copied, sketch_desktop_copied, sketch_mobile_copied).
    """
    data_samples_dir.mkdir(parents=True, exist_ok=True)
    ref_copied = False
    desk_copied = False
    mob_copied = False

    ref_src = sample_dir / "assets" / "reference.png"
    if ref_src.exists():
        shutil.copy2(ref_src, data_samples_dir / "reference.png")
        ref_copied = True

    # Prefer dataset_root when provided (works in CI / any machine)
    if dataset_root:
        sketches_src = dataset_root / "sketches"
        if sketches_src.exists():
            d = sketches_src / f"{sample_id}-d-v2.png"
            m = sketches_src / f"{sample_id}-m-v2.png"
            if d.exists():
                _flatten_png_to_white(d, data_samples_dir / "sketch_desktop_for_judge.png")
                desk_copied = True
            if m.exists():
                _flatten_png_to_white(m, data_samples_dir / "sketch_mobile_for_judge.png")
                mob_copied = True
    # Fallback: copy from paths recorded in run (when run was created on this machine)
    if not desk_copied or not mob_copied:
        sketch_paths_file = sample_dir / "sketch_paths.json"
        if sketch_paths_file.exists():
            try:
                with open(sketch_paths_file, "r", encoding="utf-8") as f:
                    paths = json.load(f)
            except (json.JSONDecodeError, OSError):
                paths = None
            mob_src: Path | None = None
            desk_src: Path | None = None
            if isinstance(paths, dict):
                mob_src = Path(paths["mobile"]) if "mobile" in paths else None
                desk_src = Path(paths["desktop"]) if "desktop" in paths else None
            elif isinstance(paths, list) and len(paths) >= 2:
                # Legacy list — use image_ordering from prompt_meta.json to resolve correctly
                ordering: list[str] = []
                meta_file = sample_dir / "prompt_meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as mf:
                            ordering = json.load(mf).get("image_ordering") or []
                    except Exception:
                        pass
                if not ordering:
                    ordering = ["mobile", "desktop"]  # legacy default
                for idx, device in enumerate(ordering):
                    if idx < len(paths):
                        if device == "mobile":
                            mob_src = Path(paths[idx])
                        elif device == "desktop":
                            desk_src = Path(paths[idx])
            if not mob_copied and mob_src and mob_src.exists():
                _flatten_png_to_white(mob_src, data_samples_dir / "sketch_mobile_for_judge.png")
                mob_copied = True
            if not desk_copied and desk_src and desk_src.exists():
                _flatten_png_to_white(desk_src, data_samples_dir / "sketch_desktop_for_judge.png")
                desk_copied = True
    return ref_copied, desk_copied, mob_copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runs data for static Explorer")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory containing run folders (default: runs)",
    )
    parser.add_argument(
        "--frontend-public",
        type=Path,
        default=Path("frontend/public"),
        help="Frontend public directory (default: frontend/public)",
    )
    parser.add_argument(
        "--no-copy-runs",
        action="store_true",
        help="Skip copying runs/ into public (e.g. when runs are already there or too large)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root (containing sketches/). If set, copy desktop and mobile sketches per sample into frontend data. Default: dataset/ when it exists.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    runs_root = (repo_root / args.runs_root).resolve()
    public_root = (repo_root / args.frontend_public).resolve()
    if args.dataset_root is not None:
        dataset_root = (repo_root / args.dataset_root).resolve()
    else:
        default_dataset = repo_root / "dataset"
        dataset_root = default_dataset if (default_dataset / "sketches").exists() else None

    if not runs_root.exists():
        print(f"Runs root does not exist: {runs_root}", file=sys.stderr)
        sys.exit(1)

    public_root.mkdir(parents=True, exist_ok=True)
    data_dir = public_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Remove legacy runs/ so dist never contains it (SPA route /runs/* must not match filesystem)
    legacy_runs = public_root / "runs"
    if legacy_runs.exists() and legacy_runs.is_dir():
        _rmtree_robust(legacy_runs)
        print(f"Removed legacy {legacy_runs} so SPA rewrites apply for /runs/*")

    run_dirs = sorted(
        d
        for d in runs_root.iterdir()
        if d.is_dir() and not d.name.startswith(".") and (d / "eval").is_dir()
    )
    leaderboard_rows: list[dict[str, Any]] = []

    run_assets_dirname = "run-assets"

    for run_dir in run_dirs:
        run_id = run_dir.name
        samples_dir = run_dir / "samples"
        eval_dir = run_dir / "eval"
        generation_provider, generation_model = _read_run_generation_model(run_dir)

        samples = []
        if samples_dir.exists():
            for sample_dir in sorted(samples_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                sample_id = sample_dir.name
                assets = sample_dir / "assets"
                ref = assets / "reference.png" if assets.exists() else None
                gen = assets / "generated.png" if assets.exists() else None
                ref_path = f"{run_assets_dirname}/{run_id}/samples/{sample_id}/assets/reference.png" if ref and ref.exists() else None
                gen_path = f"{run_assets_dirname}/{run_id}/samples/{sample_id}/assets/generated.png" if gen and gen.exists() else None
                data_sample_dir = data_dir / "runs" / run_id / "samples" / sample_id
                ref_ok, desk_ok, mob_ok = _copy_sketches_and_reference(
                    run_id, sample_id, sample_dir, data_sample_dir, dataset_root
                )
                sketch_desktop_path = (
                    f"data/runs/{run_id}/samples/{sample_id}/sketch_desktop_for_judge.png"
                    if (data_sample_dir / "sketch_desktop_for_judge.png").exists()
                    else None
                )
                sketch_mobile_path = (
                    f"data/runs/{run_id}/samples/{sample_id}/sketch_mobile_for_judge.png"
                    if (data_sample_dir / "sketch_mobile_for_judge.png").exists()
                    else None
                )
                samples.append({
                    "id": sample_id,
                    "reference_path": ref_path,
                    "generated_path": gen_path,
                    "sketch_desktop_path": sketch_desktop_path,
                    "sketch_mobile_path": sketch_mobile_path,
                })
                if not (ref_ok and desk_ok and mob_ok) and (ref_ok or desk_ok or mob_ok):
                    print(
                        f"  [{run_id}/{sample_id}] reference={ref_ok}, sketch_desktop={desk_ok}, sketch_mobile={mob_ok}",
                        file=sys.stderr,
                    )

        evals_list = []
        if eval_dir.exists():
            evals_list = [
                {"eval_id": d.name}
                for d in sorted(eval_dir.iterdir())
                if d.is_dir() and not d.name.startswith(".")
            ]

        run_data_dir = data_dir / "runs"
        run_data_dir.mkdir(parents=True, exist_ok=True)
        with open(run_data_dir / f"{run_id}.json", "w", encoding="utf-8") as f:
            json.dump({"samples": samples, "evals": evals_list}, f, indent=2)

        eval_llm_norms: dict[str, list[float]] = {e["eval_id"]: [] for e in evals_list}
        # Per-viewport weighted_iou lists, keyed eval_id -> viewport -> values.
        eval_layout_ious: dict[str, dict[str, list[float]]] = {
            e["eval_id"]: {vp: [] for vp in LAYOUT_SIMILARITY_VIEWPORTS} for e in evals_list
        }
        # Per-sample average across whichever viewports are present, then averaged across samples.
        eval_layout_combined_ious: dict[str, list[float]] = {e["eval_id"]: [] for e in evals_list}
        eval_responsiveness_violations: dict[str, list[int]] = {e["eval_id"]: [] for e in evals_list}
        eval_accessibility_violations: dict[str, list[int]] = {e["eval_id"]: [] for e in evals_list}
        eval_accessible_judge_norms: dict[str, list[float]] = {e["eval_id"]: [] for e in evals_list}
        eval_judge_model: dict[str, tuple[str | None, str | None]] = {}

        for eval_entry in evals_list:
            eval_id = eval_entry["eval_id"]
            eval_judge_model[eval_id] = _read_eval_judge_model(eval_dir / eval_id)
            eval_path = eval_dir / eval_id / "samples"
            eval_samples = []
            if eval_path.exists():
                for sample_dir in sorted(eval_path.iterdir()):
                    if not sample_dir.is_dir():
                        continue
                    sid = sample_dir.name
                    metrics_file = sample_dir / "metrics.json"
                    artifacts = sample_dir / "artifacts"
                    render_desktop = artifacts / "render_desktop.png" if artifacts.exists() else None
                    render_mobile = artifacts / "render_mobile.png" if artifacts.exists() else None
                    render_tablet = artifacts / "render_tablet.png" if artifacts.exists() else None
                    desktop_path = f"{run_assets_dirname}/{run_id}/eval/{eval_id}/samples/{sid}/artifacts/render_desktop.png" if render_desktop and render_desktop.exists() else None
                    mobile_path = f"{run_assets_dirname}/{run_id}/eval/{eval_id}/samples/{sid}/artifacts/render_mobile.png" if render_mobile and render_mobile.exists() else None
                    tablet_path = f"{run_assets_dirname}/{run_id}/eval/{eval_id}/samples/{sid}/artifacts/render_tablet.png" if render_tablet and render_tablet.exists() else None
                    eval_samples.append({
                        "id": sid,
                        "render_desktop_path": desktop_path,
                        "render_mobile_path": mobile_path,
                        "render_tablet_path": tablet_path,
                    })

            evals_dir = run_data_dir / run_id / "evals"
            evals_dir.mkdir(parents=True, exist_ok=True)
            with open(evals_dir / f"{eval_id}.json", "w", encoding="utf-8") as f:
                json.dump({"samples": eval_samples}, f, indent=2)

            eval_samples_dir = run_data_dir / run_id / "evals" / eval_id / "samples"
            eval_samples_dir.mkdir(parents=True, exist_ok=True)
            for es in eval_samples:
                sid = es["id"]
                sample_path = eval_path / sid
                metrics_file = sample_path / "metrics.json"
                artifacts = sample_path / "artifacts"
                if not metrics_file.exists():
                    continue
                with open(metrics_file, "r", encoding="utf-8") as mf:
                    metrics_data = json.load(mf)
                norm = _llm_judge_sample_normalized(metrics_data)
                if norm is not None:
                    eval_llm_norms[eval_id].append(norm)
                sample_iou_vals = []
                for vp in LAYOUT_SIMILARITY_VIEWPORTS:
                    vp_iou = _layout_similarity_viewport_iou(metrics_data, vp)
                    if vp_iou is not None:
                        eval_layout_ious[eval_id][vp].append(vp_iou)
                        sample_iou_vals.append(vp_iou)
                # Per-sample average across whichever viewports are present (usually all three).
                if sample_iou_vals:
                    eval_layout_combined_ious[eval_id].append(sum(sample_iou_vals) / len(sample_iou_vals))
                rv = _responsiveness_violations_count(metrics_data)
                if rv is not None:
                    eval_responsiveness_violations[eval_id].append(rv)
                av = _accessibility_violations_count(metrics_data)
                if av is not None:
                    eval_accessibility_violations[eval_id].append(av)
                aj = _accessible_judge_score_normalized(metrics_data)
                if aj is not None:
                    eval_accessible_judge_norms[eval_id].append(aj)
                rd = f"{run_assets_dirname}/{run_id}/eval/{eval_id}/samples/{sid}/artifacts/render_desktop.png"
                rm = f"{run_assets_dirname}/{run_id}/eval/{eval_id}/samples/{sid}/artifacts/render_mobile.png"
                rt = f"{run_assets_dirname}/{run_id}/eval/{eval_id}/samples/{sid}/artifacts/render_tablet.png"
                metrics_data["render_desktop_path"] = rd if (artifacts / "render_desktop.png").exists() else None
                metrics_data["render_mobile_path"] = rm if (artifacts / "render_mobile.png").exists() else None
                metrics_data["render_tablet_path"] = rt if (artifacts / "render_tablet.png").exists() else None
                data_sample_dir = data_dir / "runs" / run_id / "samples" / sid
                metrics_data["reference_path"] = f"data/runs/{run_id}/samples/{sid}/reference.png" if (data_sample_dir / "reference.png").exists() else None
                # Prefer eval-generated, flattened sketches if present; fall back to flattened dataset/run copies.
                eval_sketch_d = artifacts / "sketch_desktop_for_judge.png"
                eval_sketch_m = artifacts / "sketch_mobile_for_judge.png"
                metrics_data["sketch_desktop_path"] = (
                    f"{run_assets_dirname}/{run_id}/eval/{eval_id}/samples/{sid}/artifacts/sketch_desktop_for_judge.png"
                    if eval_sketch_d.exists()
                    else (
                        f"data/runs/{run_id}/samples/{sid}/sketch_desktop_for_judge.png"
                        if (data_sample_dir / "sketch_desktop_for_judge.png").exists()
                        else None
                    )
                )
                metrics_data["sketch_mobile_path"] = (
                    f"{run_assets_dirname}/{run_id}/eval/{eval_id}/samples/{sid}/artifacts/sketch_mobile_for_judge.png"
                    if eval_sketch_m.exists()
                    else (
                        f"data/runs/{run_id}/samples/{sid}/sketch_mobile_for_judge.png"
                        if (data_sample_dir / "sketch_mobile_for_judge.png").exists()
                        else None
                    )
                )
                with open(eval_samples_dir / f"{sid}.json", "w", encoding="utf-8") as f:
                    json.dump(metrics_data, f, indent=2)

        # Emit one leaderboard row per eval that was actually run, regardless of which
        # metrics it included — an eval with e.g. only layout_similarity (no
        # llm_judge_screenshots) still belongs on the leaderboard, just with blanks
        # for the metrics it didn't compute.
        for eval_id, norms in sorted(eval_llm_norms.items()):
            viewport_ious = eval_layout_ious.get(eval_id) or {vp: [] for vp in LAYOUT_SIMILARITY_VIEWPORTS}
            combined_ious = eval_layout_combined_ious.get(eval_id) or []
            resp_viols = eval_responsiveness_violations.get(eval_id) or []
            a11y_viols = eval_accessibility_violations.get(eval_id) or []
            a11y_judge = eval_accessible_judge_norms.get(eval_id) or []
            judge_provider, judge_model = eval_judge_model.get(eval_id, (None, None))
            row = {
                "run_id": run_id,
                "eval_id": eval_id,
                "generation_provider": generation_provider,
                "generation_model": generation_model,
                "judge_provider": judge_provider,
                "judge_model": judge_model,
                "llm_judge_avg_normalized": round(sum(norms) / len(norms), 6) if norms else None,
                "llm_judge_sample_count": len(norms),
                # layout_similarity_avg_iou = mean over samples of that sample's own
                # mobile/tablet/desktop average (i.e. the mean of all three viewports).
                "layout_similarity_avg_iou": round(sum(combined_ious) / len(combined_ious), 6) if combined_ious else None,
                "layout_similarity_sample_count": len(combined_ious),
                "avg_responsiveness_violations": round(sum(resp_viols) / len(resp_viols), 3) if resp_viols else None,
                "avg_accessibility_violations": round(sum(a11y_viols) / len(a11y_viols), 3) if a11y_viols else None,
                "avg_accessible_judge_score": round(sum(a11y_judge) / len(a11y_judge), 6) if a11y_judge else None,
            }
            for vp in LAYOUT_SIMILARITY_VIEWPORTS:
                vals = viewport_ious.get(vp) or []
                row[f"layout_similarity_avg_{vp}_iou"] = round(sum(vals) / len(vals), 6) if vals else None
            leaderboard_rows.append(row)

    with open(data_dir / "runs.json", "w", encoding="utf-8") as f:
        # Back-compat key name: "runs", but entries are eval-level rows.
        json.dump({"runs": leaderboard_rows}, f, indent=2)

    if not args.no_copy_runs:
        runs_dest = public_root / run_assets_dirname
        if runs_dest.exists():
            _rmtree_robust(runs_dest)
        runs_dest.mkdir(parents=True)
        for rd in run_dirs:
            shutil.copytree(rd, runs_dest / rd.name)
        print(f"Copied {len(run_dirs)} run(s) with eval/ from {runs_root} -> {runs_dest}")
    else:
        print(f"Skipped copying runs/ (--no-copy-runs)")

    data_runs = data_dir / "runs"
    if data_runs.exists():
        n_refs = sum(1 for _ in data_runs.rglob("reference.png"))
        n_sketches = sum(1 for _ in data_runs.rglob("sketch_desktop_for_judge.png"))
        print(f"Frontend data: {n_refs} reference images, {n_sketches} sketch sets in {data_dir}")
    if dataset_root:
        print(f"Sketches source: {dataset_root}/sketches/")
    print(f"Wrote data to {data_dir}, leaderboard rows: {len(leaderboard_rows)}")


if __name__ == "__main__":
    main()
