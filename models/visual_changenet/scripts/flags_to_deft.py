#!/usr/bin/env python3
"""Operator flags (runs/<run>/flags.csv from the review app) -> inputs for the next DEFT round.

    python scripts/flags_to_deft.py --runs runs/lab_day1 runs/lab_day2 --out deft/round2

Writes to --out:
  rca.md             what went wrong: missed defects and false alarms by class and position, which model was
                     right when ChangeNet and the second opinion disagreed, scores of the misses
  targets.csv        position x class counts to render more of in the digital twin (or oversample, like HARD_* in
                     prepare_vcn_dataset.py) for the next training round
  kpi_set.csv + images/   every operator-checked position as a TAO pair with the operator's label: the gate set the
                     next model must beat (score it with `tao inference` like val)
  train_add.csv      only with --allow-real-train: the same pairs for training. The brief says train on twin data
                     and use real data for testing, so this is off by default; confirm with the organisers first.
"""
import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

FIELDS = ("input_path", "golden_path", "label", "object_name")
LIGHT = "SolderLight"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--allow-real-train", action="store_true")
    args = ap.parse_args()

    flags = []
    for run in args.runs:
        path = run / "flags.csv"
        if not path.exists():
            print(f"skip {run}: no flags.csv")
            continue
        with open(path, newline="") as f:
            latest = {(r["scene"], r["position"]): r for r in csv.DictReader(f)}  # last verdict per position
        flags += [dict(r, run=run) for r in latest.values()]
    if not flags:
        raise SystemExit("no flags found")

    out = args.out
    if (out / "images").exists():
        shutil.rmtree(out / "images")
    out.mkdir(parents=True, exist_ok=True)

    pairs = []
    for r in flags:
        obj = f"kpi_{r['run'].name}_{r['scene']}_{r['position']}"
        base = f"kpi/{obj}"
        for kind in ("crop", "golden"):
            dst = out / "images" / base / ("sample" if kind == "crop" else "golden") / f"{obj}_{LIGHT}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(r["run"] / r[kind], dst)
        label = "PASS" if r["operator_label"] == "pass" else r["operator_label"]
        pairs.append({"input_path": f"{base}/sample", "golden_path": f"{base}/golden", "label": label, "object_name": obj})
    for name in ("kpi_set.csv",) + (("train_add.csv",) if args.allow_real_train else ()):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(pairs)

    wrong = [r for r in flags if r["flag_type"] in ("missed_defect", "false_alarm", "review_defect", "review_good")]
    targets = Counter((r["position"], r["operator_label"]) for r in wrong)
    with open(out / "targets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "class", "hard_cases"])
        w.writerows((p, c, n) for (p, c), n in targets.most_common())

    types = Counter(r["flag_type"] for r in flags)
    dis = [r for r in flags if r.get("disagreement") == "1"]

    def right(r, who):
        truth_defect = r["operator_label"] != "pass"
        if who == "changenet":
            return (r["model_call"] == "no_pass") == truth_defect
        return (r["second_label"].lower() not in ("pass", "good", "ok")) == truth_defect

    misses = sorted((float(r["score"]), r["scene"], r["position"], r["operator_label"])
                    for r in flags if r["flag_type"] == "missed_defect")
    lines = [f"# DEFT input from {len(flags)} operator-checked positions ({', '.join(str(r) for r in args.runs)})", "",
             "| Flag | Count |", "|---|---|", *(f"| {k} | {v} |" for k, v in types.most_common()), "",
             "## Missed defects (model said PASS)", ""]
    lines += [f"- {s} {p}: {c}, score {sc:.4f}" for sc, s, p, c in misses] or ["- none"]
    lines += ["", "## Wrong or uncertain calls by class and position", "",
              "| Position | True class | Count |", "|---|---|---|",
              *(f"| {p} | {c} | {n} |" for (p, c), n in targets.most_common(20))]
    if dis:
        lines += ["", f"## Disagreements ({len(dis)} checked by an operator)", "",
                  f"- ChangeNet right: {sum(right(r, 'changenet') for r in dis)}",
                  f"- Second opinion right: {sum(right(r, 'second') for r in dis)}"]
    lines += ["", "Next round: render or oversample the positions and classes in `targets.csv`, retrain, and gate the new "
              "model on `kpi_set.csv` (no missed defects, fewer false alarms than iter1)."]
    (out / "rca.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
