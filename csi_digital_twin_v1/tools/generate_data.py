"""Generate labeled synthetic images of the good fixture (G01-G30) from the CapSealCam view.

    <isaac-sim>/python.sh tools/generate_data.py --count 20 --out /workspace/dataset_test --subframes 8
    <isaac-sim>/python.sh tools/generate_data.py --count 300 --out /workspace/dataset

For every image, each of the 30 positions is randomly set to one of: pass, missing_cap, crooked_cap, damaged_vial, no_vial.
(damaged_vial is an approximation: a crushed, tilted body. The scrap fixture is not generated yet.) Output folder:

    images/img_0000.png            full 640x480 image, same look as the real camera
    annotated/img_0000.png         the same image with each position's box and class drawn on it
    crops/<class>/img_0000_G26.png one labeled crop per position
    labels.csv                     image, position, class, cx, cy (pixel centre of that position), crop file

Edits are made in memory only; tray_inspection.usdc is never modified.
"""
import argparse, csv, os, random, sys, time, traceback
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))

ap = argparse.ArgumentParser()
ap.add_argument("--count", type=int, default=20, help="number of full images to render")
ap.add_argument("--out", default="dataset", help="output folder")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--subframes", type=int, default=32, help="path-tracing quality; use 8 for a quick test")
ap.add_argument("--defect-rate", type=float, default=0.4,
                help="fraction of positions that get a defect, split equally between the 4 defect classes")
ap.add_argument("--crop", type=int, default=40, help="crop size in pixels of the final 640x480 image")
ap.add_argument("--stage", default=os.path.join(HERE, "csi_line.usda"),
                help="csi_line.usda (full plant, matches the real camera view) or a single station file")
ap.add_argument("--kit-arg", action="append", default=[], help="extra Isaac Sim / Kit argument (repeatable)")
a = ap.parse_args()
if not 0 <= a.defect_rate <= 1:
    ap.error("--defect-rate must be between 0 and 1")
if not os.path.isfile(a.stage):
    ap.error(f"stage does not exist: {a.stage}")

out = Path(a.out).expanduser().resolve()
for d in ("images", "annotated"):
    (out / d).mkdir(parents=True, exist_ok=True)

import json
cam = json.load(open(os.path.join(HERE, "cameras", "CapSealCam.json")))
rw, rh = cam["render_px"]
W, H = cam["final_image_px"]

from isaacsim import SimulationApp
kit = SimulationApp({"headless": True, "renderer": "PathTracing", "width": rw, "height": rh, "extra_args": a.kit_arg})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import numpy as np, cv2
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom
    import twin_defects as td

    ctx = omni.usd.get_context()
    if not ctx.open_stage(a.stage):
        raise RuntimeError(f"could not open stage: {a.stage}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(10):
        kit.update()
    stage = ctx.get_stage()
    prim = next((p for p in [cam["prim_in_csi_line"], cam["prim_in_station_file"]]
                 if stage.GetPrimAtPath(p).IsValid()), None)
    if prim is None:
        raise RuntimeError(f"camera CapSealCam not found in {a.stage}")
    root = prim.rsplit("/", 2)[0]   # /CSI_Line/TrayInspection or /TrayInspection

    rp = rep.create.render_product(prim, (rw, rh))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()

    # Same first-sync workaround as render_example.py (scaled camera renders black otherwise).
    clip = UsdGeom.Camera(stage.GetPrimAtPath(prim)).GetClippingRangeAttr()
    near_far = clip.Get()
    stage.SetEditTarget(stage.GetSessionLayer())
    clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1])); kit.update()
    clip.Set(near_far); kit.update()

    # Lens distortion maps, built once (same maths as tools/apply_lens.py).
    f, k1 = cam["focal_px_final"], cam["lens_k1"]
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    und = cv2.undistortPoints(np.stack([u.ravel(), v.ravel()], 1)[:, None, :], K,
                              np.array([k1, 0, 0, 0, 0.0])).reshape(H, W, 2)
    mapx = (und[..., 0] * f + rw / 2).astype(np.float32)
    mapy = (und[..., 1] * f + rh / 2).astype(np.float32)

    da = td.DefectApplier(stage, root=root)
    centers = da.pixel_centers(cam)
    rng = random.Random(a.seed)
    per_defect = a.defect_rate / 4
    weights = {"pass": 1 - a.defect_rate, "missing_cap": per_defect, "crooked_cap": per_defect, "damaged_vial": per_defect,
               "no_vial": per_defect}
    colors = {"pass": (0, 200, 0), "missing_cap": (0, 165, 255), "crooked_cap": (255, 0, 160),
              "damaged_vial": (0, 0, 255), "no_vial": (255, 120, 0)}
    half = a.crop // 2

    rows, t0 = [], time.time()
    for i in range(a.count):
        labels = td.sample_labels(td.GOOD_POSITIONS, rng, weights)
        da.apply(labels, rng)
        for _ in range(2):
            kit.update()
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
        data = np.asarray(ann.get_data())
        if data.ndim != 3 or data.shape[:2] != (rh, rw) or data.shape[2] < 3:
            raise RuntimeError(f"RGB capture returned shape {data.shape}; check GPU/RTX startup errors")
        bgr = data[..., :3][..., ::-1].copy()
        final = cv2.remap(bgr, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        name = f"img_{i:04d}"
        cv2.imwrite(str(out / "images" / f"{name}.png"), final)
        note = final.copy()
        for pos in td.GOOD_POSITIONS:
            cls, (cx, cy) = labels[pos], centers[pos]
            x0 = int(min(max(round(cx) - half, 0), W - a.crop)); y0 = int(min(max(round(cy) - half, 0), H - a.crop))
            crop_file = Path("crops") / cls / f"{name}_{pos}.png"
            (out / crop_file).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / crop_file), final[y0:y0 + a.crop, x0:x0 + a.crop])
            rows.append([f"images/{name}.png", pos, cls, round(float(cx), 1), round(float(cy), 1), str(crop_file)])
            if cls != "pass":
                cv2.rectangle(note, (x0, y0), (x0 + a.crop, y0 + a.crop), colors[cls], 1)
                cv2.putText(note, pos, (x0, max(y0 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[cls], 1)
        for k, (cname, col) in enumerate(c for c in colors.items() if c[0] != "pass"):
            cv2.putText(note, cname, (6, 14 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.imwrite(str(out / "annotated" / f"{name}.png"), note)
        n_def = sum(c != "pass" for c in labels.values())
        print(f"[{i + 1}/{a.count}] {name}: {n_def} defects, {time.time() - t0:.0f}s elapsed", flush=True)
        # Keep labels.csv current so a stopped run still leaves usable data.
        with open(out / "labels.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["image", "position", "class", "cx", "cy", "crop"]); w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r[2]] = counts.get(r[2], 0) + 1
    print("done:", a.count, "images,", len(rows), "labeled crops", counts, "->", out, flush=True)
except BaseException:
    traceback.print_exc()
    kit.close(exit_code=1)
    raise
kit.close()