#!/usr/bin/env python3
"""Full-scene inspection with the deployed Visual ChangeNet model: scenes in, PASS / NO_PASS / REVIEW per position out.

Steps (run_brev.sh `inspect` does crop -> TAO inference -> decide for you):

  crop    scene images -> 40x40 crop per fixture position + the golden crop of that position, in the TAO pair
          layout (runs/<run>/pairs) with runs/<run>/infer.csv. Position centres come from --labels (the
          dataset's labels.csv format: image,position,cx,cy[,class]) when given with --use-label-coords,
          otherwise from config/fixture_G.csv shifted onto each scene by phase correlation with the golden scene.
          Photos from another camera: --calibration calibration.json (app/calibrate.py) first warps each photo
          onto the golden view, so position, scale and rotation match training.
  decide  TAO scores (+ optional second opinion) -> runs/<run>/decisions.csv and decisions.json, read by the
          review app (app/review_app.py). Can be re-run any time, e.g. when the in-context model's answers arrive.
  demo    builds a run from the dataset's test scenes and the round-1 iter1 test scores, no GPU needed
          (to try the review app, or to demo if the lab GPU is not available).

Decision rule (config/deploy.json, set on val data only):
  score >= no_pass_cutoff (0.2327)     NO_PASS
  review_from (0.10) <= score < cutoff REVIEW   (near-miss buffer, a person decides)
  score < review_from                  PASS
  second opinion disagrees with the model's PASS/NO_PASS call -> REVIEW (never auto-pass on a disagreement)
"""
import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import second_opinion  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIGHT = "SolderLight"
FIELDS = ("input_path", "golden_path", "label", "object_name")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_config():
    cfg = json.loads((ROOT / "config" / "deploy.json").read_text())
    cfg.pop("notes", None)
    return cfg


def position_number(pos):
    return int(re.sub(r"\D", "", pos) or 0)


# ---------------------------------------------------------------- crop

def phase_shift(scene, golden, limit=None):
    """Integer (dx, dy) that moves the golden scene onto this scene (Hann-windowed phase correlation).
    limit: only consider shifts up to this many px (calibrated photos are already aligned; their black
    borders from the warp would otherwise pull the peak)."""
    a, b = scene - scene.mean(), golden - golden.mean()
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    f = np.fft.fft2(a * w) * np.conj(np.fft.fft2(b * w))
    r = np.fft.ifft2(f / (np.abs(f) + 1e-9)).real
    if limit is not None:
        keep = np.zeros_like(r, bool)
        for sy in range(-limit, limit + 1):
            for sx in range(-limit, limit + 1):
                keep[sy % r.shape[0], sx % r.shape[1]] = True
        r = np.where(keep, r, -np.inf)
    dy, dx = np.unravel_index(r.argmax(), r.shape)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return int(dx), int(dy)


def cut(img, cx, cy, size):
    x0, y0 = int(round(cx)) - size // 2, int(round(cy)) - size // 2
    return img.crop((x0, y0, x0 + size, y0 + size))  # PIL pads outside the image with black


def safe(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def list_scenes(paths):
    out = []
    for p in map(Path, paths):
        out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p]
    if not out:
        raise SystemExit("no scene images found")
    return out


def read_labels(path):
    """labels.csv (image,position,cx,cy[,class]) -> {scene stem: {position: row}}."""
    table = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            table[Path(r["image"]).stem][r["position"]] = r
    return table


def crop(args, cfg):
    run = Path(args.run)
    if (run / "pairs").exists():
        shutil.rmtree(run / "pairs")
    (run / "scenes").mkdir(parents=True, exist_ok=True)
    size = cfg["crop_size"]
    golden_dir = ROOT / cfg["golden_dir"]
    with open(ROOT / cfg["fixture_map"], newline="") as f:
        fixture = {r["position"]: (float(r["cx"]), float(r["cy"])) for r in csv.DictReader(f)}
    golden_scene = Image.open(golden_dir / "scene.png").convert("L")
    calib = json.loads(Path(args.calibration).read_text()) if getattr(args, "calibration", None) else None
    golden_crops = {}
    if getattr(args, "golden_photo", None):
        # real golden: a photo of a tray with all 30 positions confirmed good, from the same camera. The twin
        # golden looks very different from real photos (lighting, tall vials in perspective), so real scenes
        # are compared with a real reference instead. This is inference only, nothing is trained on it.
        g = ImageOps.exif_transpose(Image.open(args.golden_photo)).convert("RGB")
        if calib:
            g = g.transform(golden_scene.size, Image.PERSPECTIVE, calib["perspective"], Image.BICUBIC)
        if g.size != golden_scene.size:
            raise SystemExit("--golden-photo needs --calibration (or a 640x480 photo from the dataset view)")
        golden_scene = g.convert("L")
        golden_crops = {p: cut(g, cx, cy, size) for p, (cx, cy) in fixture.items()}
        g.save(run / "scenes" / "_golden.png")
    labels = read_labels(args.labels) if args.labels else {}

    tao_rows, crops, scenes = [], [], []
    for path in list_scenes(args.scenes):
        scene = safe(path.stem)
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # phone photos: honour the EXIF rotation
        if calib:
            if calib.get("photo_size") and list(img.size) != list(calib["photo_size"]):
                print(f"WARNING {path.name} is {img.size}, calibration photo was {calib['photo_size']}")
            img = img.transform(golden_scene.size, Image.PERSPECTIVE, calib["perspective"], Image.BICUBIC)
        img.save(run / "scenes" / f"{scene}.png")  # the view that was cropped, so review-app boxes line up
        known = labels.get(path.stem, {})
        if args.use_label_coords and known:
            centres = {p: (float(r["cx"]), float(r["cy"])) for p, r in known.items()}
            dx = dy = 0
        else:
            if img.size != golden_scene.size:
                raise SystemExit(f"{path.name} is {img.size}, the fixture map is for {golden_scene.size}: "
                                 "calibrate this camera with app/calibrate.py and pass --calibration")
            dx, dy = phase_shift(np.asarray(img.convert("L"), float), np.asarray(golden_scene, float),
                                 limit=args.max_shift if calib else None)
            if max(abs(dx), abs(dy)) > args.max_shift:
                print(f"WARNING {path.name}: fixture looks shifted by ({dx}, {dy}) px, check the camera/fixture")
            centres = {p: (cx + dx, cy + dy) for p, (cx, cy) in fixture.items()}
        scenes.append({"scene": scene, "image": f"scenes/{scene}.png", "source": path.name,
                       "width": img.size[0], "height": img.size[1], "shift": [dx, dy]})
        for pos, (cx, cy) in sorted(centres.items(), key=lambda kv: position_number(kv[0])):
            if pos not in golden_crops and not (golden_dir / f"{pos}.png").exists():
                raise SystemExit(f"no golden crop for {pos} in {golden_dir}")
            obj = f"lab_{scene}_{pos}"
            base = f"lab/{obj}"
            for kind, im in (("sample", cut(img, cx, cy, size)), ("golden", golden_crops[pos] if pos in golden_crops else Image.open(golden_dir / f"{pos}.png"))):
                dst = run / "pairs" / base / kind / f"{obj}_{LIGHT}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                im.convert("RGB").save(dst)
            true_class = known.get(pos, {}).get("class", "")
            # label is required by the TAO CSV; inference ignores it, so unknown positions get PASS
            tao_rows.append({"input_path": f"{base}/sample", "golden_path": f"{base}/golden",
                             "label": "PASS" if true_class in ("", "pass") else true_class, "object_name": obj})
            crops.append({"object_name": obj, "scene": scene, "position": pos, "cx": round(cx, 1), "cy": round(cy, 1),
                          "crop": f"pairs/{base}/sample/{obj}_{LIGHT}.png",
                          "golden": f"pairs/{base}/golden/{obj}_{LIGHT}.png", "true_class": true_class})

    write_csv(run / "infer.csv", tao_rows, FIELDS)
    write_csv(run / "crops.csv", crops)
    (run / "scenes.json").write_text(json.dumps(scenes, indent=1))
    print(f"{len(scenes)} scenes, {len(crops)} positions -> {run}")


def write_csv(path, rows, fields=None):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields or rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- decide

def load_scores(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "siamese_score" not in rows[0]:
        raise SystemExit(f"{path}: no siamese_score column")
    key = (lambda r: r["object_name"]) if "object_name" in rows[0] else \
        (lambda r: r["input_path"].strip("/").split("/")[1])
    return {key(r): float(r["siamese_score"]) for r in rows}


def model_call(score, cfg):
    if score >= cfg["no_pass_cutoff"]:
        return "no_pass"
    return "review" if score >= cfg["review_from"] else "pass"


def fixture_state(row):
    """empty / occupied_good / occupied_bad / uncertain. Needs a class from the second opinion to tell
    an empty hole from a defect, because ChangeNet only says 'different from golden'."""
    label = row["second_label"].lower()
    if label:
        if label == "no_vial":
            return "empty"
        return "occupied_bad" if second_opinion.is_defect(label) else "occupied_good"
    return {"pass": "occupied_good", "no_pass": "occupied_bad_or_empty"}.get(row["model_call"], "uncertain")


def fill_gaps(rows):
    """Robot fills positions in number order (rightmost column top to bottom, then leftward). An empty
    position before an occupied one is a missed placement; trailing empties may just be waiting."""
    order = sorted(rows, key=lambda r: position_number(r["position"]))
    last_occupied = max((i for i, r in enumerate(order) if r["state"].startswith("occupied")), default=-1)
    return [r["position"] for i, r in enumerate(order) if r["state"] == "empty" and i < last_occupied]


def decide(args, cfg):
    run = Path(args.run)
    scores_path = Path(args.scores) if args.scores else next(iter(sorted((run / "tao").rglob("inference.csv"))), None)
    if not scores_path or not scores_path.exists():
        raise SystemExit(f"no inference.csv under {run / 'tao'}; run TAO inference first (run_brev.sh inspect)")
    scores = load_scores(scores_path)
    with open(run / "crops.csv", newline="") as f:
        crops = list(csv.DictReader(f))
    scenes = json.loads((run / "scenes.json").read_text())
    ask = second_opinion.load(args.second_opinion) if args.second_opinion else None

    rows = []
    for c in crops:
        if c["object_name"] not in scores:
            raise SystemExit(f"no score for {c['object_name']} in {scores_path}")
        score = scores[c["object_name"]]
        call = model_call(score, cfg)
        reasons = {"no_pass": ["score >= cutoff"], "review": ["score in review band"], "pass": []}[call]
        second = None
        if ask:
            scene_img = next(s["image"] for s in scenes if s["scene"] == c["scene"])
            second = ask({"scene": c["scene"], "position": c["position"], "score": score,
                          "crop": str((run / c["crop"]).resolve()), "golden": str((run / c["golden"]).resolve()),
                          "scene_image": str((run / scene_img).resolve())})
        disagree = bool(second) and second_opinion.is_defect(second["label"]) != (score >= cfg["no_pass_cutoff"])
        decision = "review" if disagree else call
        if disagree:
            reasons.append(f"second opinion says {second['label']}")
        row = {"scene": c["scene"], "position": c["position"], "object_name": c["object_name"],
               "score": round(score, 6), "model_call": call,
               "second_label": second["label"] if second else "",
               "second_confidence": second.get("confidence", "") if second else "",
               "second_reason": second.get("reason", "") if second else "",
               "disagreement": int(disagree), "decision": decision, "reasons": "; ".join(reasons),
               "true_class": c.get("true_class", ""), "cx": c["cx"], "cy": c["cy"],
               "crop": c["crop"], "golden": c["golden"]}
        row["state"] = fixture_state(row)
        rows.append(row)

    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)
    for s in scenes:
        s["counts"] = dict(Counter(r["decision"] for r in by_scene[s["scene"]]))
        s["missed_placements"] = fill_gaps(by_scene[s["scene"]])
        s["fixture_ok"] = s["counts"].get("pass", 0) == len(by_scene[s["scene"]]) and not s["missed_placements"]

    write_csv(run / "decisions.csv", rows)
    summary = {"positions": len(rows), "decisions": dict(Counter(r["decision"] for r in rows)),
               "disagreements": sum(r["disagreement"] for r in rows),
               "scenes": len(scenes), "scenes_all_pass": sum(s["fixture_ok"] for s in scenes)}
    known = [r for r in rows if r["true_class"]]
    if known:  # labelled scenes (e.g. the test split): the brief's numbers at the deployed rule
        defects = [r for r in known if r["true_class"] != "pass"]
        good = [r for r in known if r["true_class"] == "pass"]
        summary["vs_labels"] = {
            "defects": len(defects), "missed (decision pass)": sum(r["decision"] == "pass" for r in defects),
            "missed_per_class": dict(Counter(r["true_class"] for r in defects if r["decision"] == "pass")),
            "good": len(good), "false_alarms (no_pass)": sum(r["decision"] == "no_pass" for r in good),
            "good_sent_to_review": sum(r["decision"] == "review" for r in good)}
    (run / "decisions.json").write_text(json.dumps(
        {"config": cfg, "scores_file": str(scores_path), "second_opinion": args.second_opinion or "",
         "summary": summary, "scenes": scenes}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {run / 'decisions.csv'}; review with: python app/review_app.py {run}")


# ---------------------------------------------------------------- demo

def demo(args, cfg):
    """Replay: the dataset's test scenes with the iter1 test scores from round 1 (results/round1/iter1)."""
    src = Path(args.dataset)
    test = src / "test"
    images = sorted((test / "images").glob("*.png"))[: args.scenes]
    run = Path(args.run)
    crop(argparse.Namespace(run=args.run, scenes=[str(p) for p in images], labels=str(test / "labels.csv"),
                            use_label_coords=False, max_shift=8, calibration=None, golden_photo=None), cfg)
    # round-1 scores are keyed test_<scene>_<pos>; rename to this run's lab_<scene>_<pos>
    scores = load_scores(args.scores)
    out = [{"object_name": k.replace("test_", "lab_", 1), "siamese_score": v} for k, v in scores.items()]
    (run / "tao").mkdir(exist_ok=True)
    write_csv(run / "tao" / "inference.csv", out)
    decide(argparse.Namespace(run=args.run, scores=None, second_opinion=args.second_opinion), cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crop")
    c.add_argument("--run", required=True)
    c.add_argument("--scenes", required=True, nargs="+", help="scene image files or folders")
    c.add_argument("--labels", help="optional labels.csv (image,position,cx,cy[,class]); class enables scoring")
    c.add_argument("--use-label-coords", action="store_true", help="crop at the cx,cy in --labels")
    c.add_argument("--calibration", help="calibration.json from app/calibrate.py (photos from another camera)")
    c.add_argument("--golden-photo", help="photo of an all-good tray from the same camera, used as the golden reference")
    c.add_argument("--max-shift", type=int, default=8, help="warn when a scene is shifted more than this (px)")
    d = sub.add_parser("decide")
    d.add_argument("--run", required=True)
    d.add_argument("--scores", help="TAO inference.csv (default: the one under <run>/tao)")
    d.add_argument("--second-opinion", help="csv:<file> or py:<file.py>:<function>, see second_opinion.py")
    m = sub.add_parser("demo")
    m.add_argument("--dataset", required=True, help="extracted vial_dataset folder (has test/)")
    m.add_argument("--scores", default=str(ROOT / "results/round1/iter1/inference_test.csv"))
    m.add_argument("--run", default=str(ROOT / "runs/demo"))
    m.add_argument("--scenes", type=int, default=10)
    m.add_argument("--second-opinion")
    args = ap.parse_args()
    {"crop": crop, "decide": decide, "demo": demo}[args.cmd](args, load_config())


if __name__ == "__main__":
    main()
