#!/usr/bin/env python3
"""Generic per-run aggregator for an inference/run_eval.py output directory.

Reads `<run_dir>/log_ep*.json` (per-step) and, for the optional wall-time row,
`<run_dir>/summary_ep*.json`. NOTE: run_eval.py names its shard summaries
`summary<suffix>.json`, so that glob normally matches nothing and the
wall-time row is simply omitted. Emits:

  <run_dir>/per_episode.csv     one row per episode, final-step metrics
  <run_dir>/per_step.csv        one row per step, aggregated mean ± std across eps
  <run_dir>/aggregate.json      structured per-step stats {steps: {t: {key: {mean,std,min,max}}}}
  <run_dir>/REPORT.md           human-readable headline + per-step table

Works for any per-step log whose entries follow the run_eval.py schema
(miou / iou_occ / iou_free / sem_miou / sem_acc / mapped_frac / step_time +
action='view'|'push'). Extra fields (best_view_ig, sigma_o_time, score_ua,
score_ub, …) are picked up automatically if every step entry carries them.

Usage (from the repo root, with PYTHONPATH=$PWD):
  python -m inference.aggregate --run_dir experiments/low_v1_recursive_step25
  python -m inference.aggregate --run_dir experiments/smoke --title "smoke test"
"""
import argparse
import csv
import glob
import json
import math
import statistics as _st
from pathlib import Path


# Always reported (must be present in every step entry).
CORE_STEP_KEYS = ["miou", "iou_occ", "iou_free",
                  "sem_miou", "sem_acc", "mapped_frac", "step_time"]
# Optional — reported if present in every step entry of every episode.
OPTIONAL_STEP_KEYS = [
    "best_view_ig", "obs_horizon_ig", "best_push_score",
    "sigma_o_time", "push_eval_time",
    "score_ua", "score_ub", "ua_max", "ub_max",
]


def _mean_std(vals):
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan")
    m = float(sum(vals) / len(vals))
    s = float(_st.stdev(vals)) if len(vals) > 1 else 0.0
    return m, s, float(min(vals)), float(max(vals))


def _is_num(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir",  type=str, required=True)
    ap.add_argument("--title",    type=str, default="",
                    help="Title for REPORT.md. Defaults to the run dir name.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"[aggregate] run_dir not found: {run_dir}")

    log_paths = sorted(glob.glob(str(run_dir / "log_ep*.json")))
    if not log_paths:
        raise SystemExit(f"[aggregate] no log_ep*.json under {run_dir}")

    # ── load every episode (skip empty / corrupted) ─────────────────────────
    episodes = []
    n_steps_set = set()
    for lp in log_paths:
        ep_id = int(Path(lp).stem.replace("log_ep", ""))
        try:
            log = json.load(open(lp))
        except Exception as e:
            print(f"[aggregate] could not parse {lp}: {e}")
            continue
        if not log:
            continue
        episodes.append((ep_id, log))
        n_steps_set.add(len(log))

    if not episodes:
        raise SystemExit(f"[aggregate] no usable episodes in {run_dir}")
    episodes.sort(key=lambda x: x[0])

    if len(n_steps_set) > 1:
        print(f"[aggregate] WARNING — episodes have varying step counts: {sorted(n_steps_set)}")
    n_steps = max(n_steps_set)

    # ── decide which optional step keys are uniformly present ───────────────
    available_opt = []
    for k in OPTIONAL_STEP_KEYS:
        ok = all(_is_num(e.get(k)) for _, log in episodes for e in log)
        if ok:
            available_opt.append(k)
    step_keys = CORE_STEP_KEYS + available_opt

    # ── per-step aggregate ──────────────────────────────────────────────────
    per_step_agg = {t: {k: [] for k in step_keys} for t in range(n_steps)}
    pushes_at_step = [0] * n_steps
    for _, log in episodes:
        for e in log:
            t = e["step"]
            if t >= n_steps:
                continue
            for k in step_keys:
                v = e.get(k)
                if _is_num(v):
                    per_step_agg[t][k].append(float(v))
            if e.get("action") == "push":
                pushes_at_step[t] += 1

    # ── per-episode summary (final-step metrics) ─────────────────────────────
    per_ep_rows = []
    for ep_id, log in episodes:
        final = log[-1]
        pushes = sum(1 for e in log if e.get("action") == "push")
        per_ep_rows.append({
            "episode":           ep_id,
            "final_mIoU":        round(float(final.get("miou", 0)), 4),
            "final_iou_occ":     round(float(final.get("iou_occ", 0)), 4),
            "final_sem_mIoU":    round(float(final.get("sem_miou", 0)), 4),
            "final_sem_acc":     round(float(final.get("sem_acc", 0)), 4),
            "final_mapped_frac": round(float(final.get("mapped_frac", 0)), 4),
            "pushes":            pushes,
        })

    # ── aggregate.json ──────────────────────────────────────────────────────
    aggregate = {
        "n_episodes": len(episodes),
        "steps": {},
        "push_step_histogram": pushes_at_step,
        "pushes_per_ep_dist": sorted([r["pushes"] for r in per_ep_rows]),
    }
    for t in range(n_steps):
        bucket = {}
        for k in step_keys:
            m, s, lo, hi = _mean_std(per_step_agg[t][k])
            bucket[k] = {"mean": m, "std": s, "min": lo, "max": hi}
        aggregate["steps"][str(t)] = bucket
    json.dump(aggregate, open(run_dir / "aggregate.json", "w"), indent=2)
    print(f"[aggregate] wrote {run_dir/'aggregate.json'}")

    # ── per_step.csv ────────────────────────────────────────────────────────
    per_step_csv = run_dir / "per_step.csv"
    # CSV header style: mIoU_mean, mIoU_std, iou_occ_mean, iou_occ_std, ...
    header = ["step"]
    csv_cols = [("mIoU",        "miou"),
                ("iou_occ",     "iou_occ"),
                ("sem_miou",    "sem_miou"),
                ("sem_acc",     "sem_acc"),
                ("mapped",      "mapped_frac")]
    # optional fields with `_mean` only (single-stat report — keeps CSV thin)
    opt_mean_only = [k for k in available_opt if k != "step_time"]
    for label, _ in csv_cols:
        header += [f"{label}_mean", f"{label}_std"]
    for k in opt_mean_only:
        header.append(f"{k}_mean")
    if "step_time" in step_keys:
        header.append("step_time_mean")
    header.append("pushes_at_step")
    with open(per_step_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t in range(n_steps):
            row = [t]
            for label, key in csv_cols:
                m, s, *_ = _mean_std(per_step_agg[t][key])
                row += [round(m, 4), round(s, 4)]
            for k in opt_mean_only:
                m, *_ = _mean_std(per_step_agg[t][k])
                row.append(round(m, 4))
            if "step_time" in step_keys:
                m, *_ = _mean_std(per_step_agg[t]["step_time"])
                row.append(round(m, 4))
            row.append(pushes_at_step[t])
            w.writerow(row)
    print(f"[aggregate] wrote {per_step_csv}")

    # ── per_episode.csv ─────────────────────────────────────────────────────
    per_ep_csv = run_dir / "per_episode.csv"
    fieldnames = list(per_ep_rows[0].keys())
    with open(per_ep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_ep_rows)
    print(f"[aggregate] wrote {per_ep_csv}")

    # ── REPORT.md ──────────────────────────────────────────────────────────
    title = args.title or run_dir.name
    final_step = n_steps - 1
    headline = aggregate["steps"][str(final_step)]
    pushes = [r["pushes"] for r in per_ep_rows]
    walls = []
    # run_eval.py writes summary<--summary_suffix>.json, one per shard, and each
    # holds every episode of that shard.
    for sj in sorted(run_dir.glob("summary*.json")):
        try:
            s = json.load(open(sj))
            for ep in s.get("episodes") or []:
                wall = ep.get("wall_time")
                if wall:
                    walls.append(float(wall))
        except Exception:
            pass
    push_mean, push_std, *_ = _mean_std(pushes) if pushes else (float('nan'), float('nan'), 0, 0)
    wall_mean, wall_std, *_ = _mean_std(walls) if walls else (float('nan'), float('nan'), 0, 0)

    lines = []
    lines.append(f"# {title} — {len(episodes)}-episode summary")
    lines.append("")
    lines.append(f"Run dir: `{run_dir.relative_to(run_dir.parent.parent) if run_dir.is_relative_to(run_dir.parent.parent) else run_dir}`  ")
    lines.append(f"Episodes: {len(episodes)}  steps/ep: {n_steps}")
    lines.append("")
    lines.append(f"## Final metrics (step {final_step}, mean ± std over {len(episodes)} episodes)")
    lines.append("")
    lines.append("| metric | mean | std |")
    lines.append("|---|---|---|")
    lines.append(f"| mIoU (occ+free) | {headline['miou']['mean']:.4f} | {headline['miou']['std']:.4f} |")
    lines.append(f"| IoU occupancy | {headline['iou_occ']['mean']:.4f} | {headline['iou_occ']['std']:.4f} |")
    lines.append(f"| sem_mIoU (paper) | **{headline['sem_miou']['mean']:.4f}** | {headline['sem_miou']['std']:.4f} |")
    lines.append(f"| sem_acc (pixel) | {headline['sem_acc']['mean']:.4f} | {headline['sem_acc']['std']:.4f} |")
    lines.append(f"| mapped_frac | {headline['mapped_frac']['mean']:.4f} | {headline['mapped_frac']['std']:.4f} |")
    lines.append(f"| pushes / episode | {push_mean:.2f} | {push_std:.2f} |")
    if walls:
        lines.append(f"| wall / ep (s) | {wall_mean:.1f} | {wall_std:.1f} |")
    lines.append("")
    lines.append(f"## Metrics by step (mean over {len(episodes)} episodes)")
    lines.append("")
    lines.append("| step | mIoU | iou_occ | sem_mIoU | sem_acc | mapped | pushes_here |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in range(n_steps):
        b = aggregate["steps"][str(t)]
        lines.append(
            f"| {t:>2} | {b['miou']['mean']:.4f} | {b['iou_occ']['mean']:.4f} | "
            f"{b['sem_miou']['mean']:.4f} | {b['sem_acc']['mean']:.4f} | "
            f"{b['mapped_frac']['mean']:.4f} | {pushes_at_step[t]} |"
        )
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[aggregate] wrote {run_dir/'REPORT.md'}")


if __name__ == "__main__":
    main()
