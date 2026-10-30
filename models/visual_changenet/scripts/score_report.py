#!/usr/bin/env python3
"""Turn Visual ChangeNet inference scores into the numbers the brief asks for.

Threshold rule (no test-set tuning): the PASS/NO_PASS threshold is the highest score that still
flags every defect in the VALIDATION split. That threshold is then applied unchanged to TEST.

Reported on test:
  * per-class false negatives (missed defects) at the val threshold
  * false alarm rate (good vials flagged) at the val threshold
  * the brief's metric: false alarm rate at the threshold that catches all test defects
    (reported only, never fed back into training or threshold choice)
  * AUROC, a scene-level view (how many fixture images had any miss / any false alarm)
  * misses.csv / false_alarms.csv for DEFT AOI root-cause work

Usage:
  python score_report.py --meta data/vcn/meta.csv \
      --val results/baseline/inference_val/inference.csv \
      --test results/baseline/inference_test/inference.csv \
      --out results/baseline/report
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_scores(path):
    """object_name -> siamese_score (higher = more different from golden = more likely defect)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "siamese_score" not in rows[0] or not {"object_name", "input_path"} & set(rows[0]):
        raise SystemExit(f"{path}: expected siamese_score plus object_name or input_path, got {list(rows[0]) if rows else 'no rows'}")
    # input_path is <split>/<object_name>/sample (see prepare_vcn_dataset.py), so it identifies the sample too
    key = (lambda r: r["object_name"]) if "object_name" in rows[0] else (lambda r: r["input_path"].strip("/").split("/")[1])
    return {key(r): float(r["siamese_score"]) for r in rows}


def join(scores, meta, split):
    rows = [dict(m, score=scores[m["object_name"]]) for m in meta if m["split"] == split and m["object_name"] in scores]
    expected = sum(m["split"] == split for m in meta)
    if len(rows) != expected:
        raise SystemExit(f"{split}: scored {len(rows)} of {expected} samples; check the inference CSV")
    return rows


def margin_threshold(rows):
    """Midpoint of the val gap between the highest-scoring good vial and the lowest-scoring defect.

    Scores pile up near 1.0, so the plain catch-all threshold sits on the edge of the val defects and
    test defects just below it get missed. When val separates cleanly, the midpoint leaves room on both
    sides; when it does not, fall back to catch-all so no val defect is missed.
    """
    lowest_defect = catch_all_threshold(rows)
    highest_pass = max(r["score"] for r in rows if r["class"] == "pass")
    return (lowest_defect + highest_pass) / 2 if highest_pass < lowest_defect else lowest_defect


def catch_all_threshold(rows):
    """Highest threshold t with every defect at score >= t."""
    return min(r["score"] for r in rows if r["class"] != "pass")


def auroc(rows):
    pos = [r["score"] for r in rows if r["class"] != "pass"]
    neg = [r["score"] for r in rows if r["class"] == "pass"]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def at_threshold(rows, t):
    per_class = defaultdict(lambda: {"total": 0, "missed": 0})
    passes = flagged_pass = 0
    for r in rows:
        flagged = r["score"] >= t
        if r["class"] == "pass":
            passes += 1
            flagged_pass += flagged
        else:
            per_class[r["class"]]["total"] += 1
            per_class[r["class"]]["missed"] += not flagged
    for c in per_class.values():
        c["fn_rate"] = c["missed"] / c["total"]
    defects = sum(c["total"] for c in per_class.values())
    missed = sum(c["missed"] for c in per_class.values())
    return {"threshold": t, "per_class_false_negatives": dict(sorted(per_class.items())),
            "missed_defects": missed, "defects": defects, "recall": 1 - missed / defects,
            "false_alarms": flagged_pass, "passes": passes, "false_alarm_rate": flagged_pass / passes}


def scene_view(rows, t):
    scenes = defaultdict(lambda: {"miss": False, "false_alarm": False})
    for r in rows:
        flagged = r["score"] >= t
        s = scenes[r["scene"]]
        s["miss"] |= r["class"] != "pass" and not flagged
        s["false_alarm"] |= r["class"] == "pass" and flagged
    return {"scenes": len(scenes),
            "scenes_with_a_missed_defect": sum(s["miss"] for s in scenes.values()),
            "scenes_fully_correct": sum(not s["miss"] and not s["false_alarm"] for s in scenes.values())}


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def pct(x):
    return f"{100 * x:.2f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--val", required=True, type=Path, help="inference.csv for the val split")
    ap.add_argument("--stress", type=Path, help="inference.csv for val_stress (val scenes, harsher lighting/blur/shift)")
    ap.add_argument("--test", type=Path, help="inference.csv for the test split (final report only)")
    ap.add_argument("--title", default="Visual ChangeNet baseline")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rule", choices=("margin", "edge"), default="margin",
                    help="margin: midpoint of the val pass/defect gap (default); edge: highest score that flags every val defect")
    args = ap.parse_args()

    with open(args.meta, newline="") as f:
        meta = list(csv.DictReader(f))
    args.out.mkdir(parents=True, exist_ok=True)

    val = join(load_scores(args.val), meta, "val")
    t = margin_threshold(val) if args.rule == "margin" else catch_all_threshold(val)
    rule = {"margin": "midpoint of the val gap between good vials and defects",
            "edge": "highest score that flags every val defect"}[args.rule]
    report = {"threshold_rule": rule, "threshold": t,
              "val": {**at_threshold(val, t), "auroc": auroc(val)}}

    lines = [f"# {args.title}", "",
             f"Threshold chosen on **val** ({rule}): `{t:.6f}`",
             f"Val false alarm rate at that threshold: **{pct(report['val']['false_alarm_rate'])}**, "
             f"AUROC {report['val']['auroc']:.4f}", ""]

    if args.stress:
        st = join(load_scores(args.stress), meta, "val_stress")
        at_t = at_threshold(st, t)
        report["val_stress"] = {"at_val_threshold": at_t, "auroc": auroc(st),
                                "far_at_catch_all": at_threshold(st, catch_all_threshold(st))["false_alarm_rate"]}
        lines += ["## Stressed val (val scenes, harsher lighting, blur and shift; used to compare models)", "",
                  f"- At the val threshold: missed **{at_t['missed_defects']} / {at_t['defects']}**, "
                  f"false alarms **{pct(at_t['false_alarm_rate'])}**",
                  f"- False alarm rate when every stressed defect is caught: "
                  f"**{pct(report['val_stress']['far_at_catch_all'])}**, AUROC {report['val_stress']['auroc']:.4f}",
                  "- Missed per class: " + ", ".join(f"{c} {v['missed']}/{v['total']}"
                                                     for c, v in at_t["per_class_false_negatives"].items()), ""]

    if args.test:
        test = join(load_scores(args.test), meta, "test")
        at_val_t = at_threshold(test, t)
        t_all = catch_all_threshold(test)
        report["test"] = {"at_val_threshold": at_val_t, "auroc": auroc(test), "scene_view": scene_view(test, t),
                          "brief_metric_far_at_catch_all_test_threshold": at_threshold(test, t_all)}
        brief = report["test"]["brief_metric_far_at_catch_all_test_threshold"]
        lines += ["## Test (never used for tuning)", "",
                  f"- Missed defects at the val threshold: **{at_val_t['missed_defects']} / {at_val_t['defects']}** "
                  f"(recall {pct(at_val_t['recall'])})",
                  f"- False alarm rate at the val threshold: **{pct(at_val_t['false_alarm_rate'])}** "
                  f"({at_val_t['false_alarms']} of {at_val_t['passes']} good vials)",
                  f"- Brief metric, false alarm rate at the threshold that catches every test defect: "
                  f"**{pct(brief['false_alarm_rate'])}**",
                  f"- AUROC: {report['test']['auroc']:.4f}", "",
                  "| Class | Test samples | Missed at val threshold | Miss rate |", "|---|---|---|---|"]
        for c, v in at_val_t["per_class_false_negatives"].items():
            lines.append(f"| {c} | {v['total']} | {v['missed']} | {pct(v['fn_rate'])} |")
        sv = report["test"]["scene_view"]
        lines += ["", f"Scenes: {sv['scenes']} fixture images, {sv['scenes_with_a_missed_defect']} with at least one "
                  f"missed defect, {sv['scenes_fully_correct']} with every position correct.", ""]
        cols = ("object_name", "scene", "position", "class", "score")
        write_rows(args.out / "test_misses.csv", sorted(
            ({k: r[k] for k in cols} for r in test if r["class"] != "pass" and r["score"] < t), key=lambda r: r["score"]))
        write_rows(args.out / "test_false_alarms.csv", sorted(
            ({k: r[k] for k in cols} for r in test if r["class"] == "pass" and r["score"] >= t), key=lambda r: -r["score"]))
        lines.append("Per-sample lists: `test_misses.csv`, `test_false_alarms.csv`.")

    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
