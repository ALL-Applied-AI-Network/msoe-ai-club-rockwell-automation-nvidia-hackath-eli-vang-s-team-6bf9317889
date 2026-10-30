#!/usr/bin/env python3
"""Convert the vial dataset into the TAO Visual ChangeNet classify layout.

Input (the extracted vial_dataset.zip):
    <src>/golden/labels.csv + crops/pass/img_0000_GNN.png   (one golden fixture, 30 positions)
    <src>/train/labels.csv  + crops/<class>/...              (200 scenes x 30 positions)
    <src>/test/labels.csv   + crops/<class>/...              (50 scenes x 30 positions)

Output (<out>):
    images/<split>/<object_name>/sample/<object_name>_SolderLight.png   crop under inspection
    images/<split>/<object_name>/golden/<object_name>_SolderLight.png   golden crop, same position
    train.csv, val.csv, test.csv      4-column TAO CSV (input_path,golden_path,label,object_name)
    train_hard.csv                    train.csv with hard cases repeated (iteration 1, see HARD_* below)
    val_stress.csv                    val crops re-rendered with harsher lighting, blur and shift (iteration 1)
    meta.csv                          per-sample class, position, scene and render params (for eval)
    split.json                        which scenes went to train vs val

Labels: "pass" -> "PASS"; every other class (missing_cap, crooked_cap, no_vial, ...) keeps its
name, which TAO collapses to NO_PASS. Keeping the name lets the eval report per-class misses.

The validation split is carved out of train BY SCENE (all 30 positions of a scene stay together),
so neighbouring-vial context never leaks between train and val. The test split is copied as-is
and is only ever used for the final report, never for threshold or checkpoint selection.
"""
import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

LIGHT = "SolderLight"
FIELDS = ("input_path", "golden_path", "label", "object_name")

# Iteration 1 hard cases, chosen from the baseline's VAL scores only: the lowest-scoring val defects were all
# crooked caps, mostly in the leftmost columns (G19-G30) where the fixture edge darkens the background.
HARD_CLASS, HARD_CLASS_REPEAT = "crooked_cap", 3
HARD_POSITIONS, HARD_POSITION_REPEAT = {f"G{i:02d}" for i in range(19, 31)}, 2


def hard_repeat(row):
    n = HARD_CLASS_REPEAT if row["class"] == HARD_CLASS else 1
    return n * (HARD_POSITION_REPEAT if row["position"] in HARD_POSITIONS else 1)


def stress(img, rng):
    """Harsher version of the dataset's own domain randomization (light_stops 0.6-1.0, shifts of a few px)."""
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.55, 1.45))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.3))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 1.0)))
    dx, dy = rng.randint(-3, 3), rng.randint(-3, 3)
    return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def read_labels(split_dir):
    with open(split_dir / "labels.csv", newline="") as f:
        return list(csv.DictReader(f))


def scene_id(row):
    return Path(row["image"]).stem


def place(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="extracted vial_dataset folder (has golden/ train/ test/)")
    ap.add_argument("--out", required=True, type=Path, help="output folder, e.g. data/vcn")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of train scenes held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = args.src.resolve(), args.out.resolve()
    if (out / "images").exists():
        shutil.rmtree(out / "images")
    out.mkdir(parents=True, exist_ok=True)

    golden = {r["position"]: src / "golden" / r["crop"] for r in read_labels(src / "golden")}

    train_rows = read_labels(src / "train")
    scenes = sorted({scene_id(r) for r in train_rows})
    rng = random.Random(args.seed)
    val_scenes = set(rng.sample(scenes, round(len(scenes) * args.val_frac)))

    splits = {
        "train": [r for r in train_rows if scene_id(r) not in val_scenes],
        "val": [r for r in train_rows if scene_id(r) in val_scenes],
        "test": read_labels(src / "test"),
    }
    src_dir = {"train": "train", "val": "train", "test": "test"}

    meta = []
    for split, rows in splits.items():
        tao_rows = []
        for r in rows:
            if r["position"] not in golden:
                raise SystemExit(f"no golden crop for position {r['position']} ({r['crop']})")
            obj = f"{split}_{scene_id(r)}_{r['position']}"
            base = f"{split}/{obj}"
            place(src / src_dir[split] / r["crop"], out / "images" / base / "sample" / f"{obj}_{LIGHT}.png")
            place(golden[r["position"]], out / "images" / base / "golden" / f"{obj}_{LIGHT}.png")
            label = "PASS" if r["class"] == "pass" else r["class"]
            tao_rows.append({"input_path": f"{base}/sample", "golden_path": f"{base}/golden",
                             "label": label, "object_name": obj})
            meta.append({"object_name": obj, "split": split, "scene": scene_id(r), "position": r["position"],
                         "class": r["class"], **{k: r[k] for k in
                         ("light_stops", "tint", "shift_x", "shift_y", "roll_deg", "image_mean")}})
        with open(out / f"{split}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(tao_rows)
        print(f"{split:5s} {len(rows):5d} samples  {dict(Counter(r['class'] for r in rows))}")

    with open(out / "train_hard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        by_obj = {m["object_name"]: m for m in meta if m["split"] == "train"}
        with open(out / "train.csv", newline="") as tf:
            hard = [r for r in csv.DictReader(tf) for _ in range(hard_repeat(by_obj[r["object_name"]]))]
        w.writerows(hard)
    print(f"train_hard {len(hard):5d} rows (crooked_cap x{HARD_CLASS_REPEAT}, G19-G30 x{HARD_POSITION_REPEAT})")

    rng = random.Random(args.seed + 1)
    stress_rows = []
    for m in [m for m in meta if m["split"] == "val"]:
        obj = m["object_name"].replace("val_", "valstress_", 1)
        base = f"val_stress/{obj}"
        src_img = out / "images" / "val" / m["object_name"] / "sample" / f"{m['object_name']}_{LIGHT}.png"
        dst = out / "images" / base / "sample" / f"{obj}_{LIGHT}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        stress(Image.open(src_img).convert("RGB"), rng).save(dst)
        place(golden[m["position"]], out / "images" / base / "golden" / f"{obj}_{LIGHT}.png")
        stress_rows.append({"input_path": f"{base}/sample", "golden_path": f"{base}/golden",
                            "label": "PASS" if m["class"] == "pass" else m["class"], "object_name": obj})
        meta.append({**m, "object_name": obj, "split": "val_stress"})
    with open(out / "val_stress.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(stress_rows)
    print(f"val_stress {len(stress_rows):5d} samples")

    with open(out / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(meta)
    (out / "split.json").write_text(json.dumps(
        {"seed": args.seed, "val_scenes": sorted(val_scenes),
         "train_scenes": sorted(set(scenes) - val_scenes)}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
