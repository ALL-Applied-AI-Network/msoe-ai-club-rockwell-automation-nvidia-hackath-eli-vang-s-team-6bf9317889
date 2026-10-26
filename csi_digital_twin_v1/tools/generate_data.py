"""Generate labeled synthetic images of the good fixture (G01-G30) from the CapSealCam view.

    <isaac-sim>/python.sh tools/generate_data.py --count 20 --out /workspace/dataset_test --subframes 8
    <isaac-sim>/python.sh tools/generate_data.py --count 300 --out /workspace/dataset

For every image the fixture is either full or partly filled (the robot fills G01, G02, ... in order), and each filled
position is randomly pass, missing_cap or crooked_cap (add damaged_vial with --include-damaged). Empty positions
are no_vial.
(damaged_vial is an approximation: a crushed, tilted body. The scrap fixture is not generated yet.) Output folder:

    images/img_0000.png            full 640x480 image, same look as the real camera
    annotated/img_0000.png         the same image with each position's box and class drawn on it
    crops/<class>/img_0000_G26.png one labeled crop per position
    labels.csv                     image, position, class, cx, cy (pixel centre of that position), crop file, and
                                   per image: light_stops, tint, shift_x/y (pixels), roll_deg, image_mean (brightness 0-255)

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
                help="fraction of FILLED positions that get a vial defect, split equally between the defect classes "
                     "(missing_cap and crooked_cap; plus damaged_vial with --include-damaged)")
ap.add_argument("--include-damaged", action="store_true",
                help="also generate damaged_vial (off by default: it is only an approximation, a crushed tilted body)")
ap.add_argument("--partial-fill", type=float, default=0.7,
                help="fraction of images where the fixture is only partly filled (robot mid-run): the first n "
                     "positions G01.. hold vials and the rest are empty (no_vial). 0 = always full")
ap.add_argument("--gap-rate", type=float, default=0.03,
                help="chance that a position inside the filled area is empty (a missed placement)")
ap.add_argument("--light-offset", type=float, default=1.0,
                help="brighten all lights by this many stops (measured: the real plate is 0.8-1.2 stops brighter "
                     "than the twin's default)")
ap.add_argument("--light-jitter", type=float, default=0.4,
                help="random brightness variation: max change in exposure in stops (0.4 = about +/-30%%); 0 = off")
ap.add_argument("--light-tint", type=float, default=0.3,
                help="random warm-to-cool colour shift in stops (real images range from warm to cooler); 0 = off")
ap.add_argument("--roll-deg", type=float, default=3.6,
                help="turn the picture clockwise by this many degrees (positive = tilted right). Measured: the real "
                     "images are rotated about 3.6 deg clockwise compared with the twin camera")
ap.add_argument("--roll-jitter", type=float, default=0.3, help="random +/- degrees added to --roll-deg per image")
ap.add_argument("--shift-px", type=float, default=6.0,
                help="random image shift up to +/- this many pixels, like the small camera/fixture movement between "
                     "real images (labels move with it); 0 = off. Must stay under about 25")
ap.add_argument("--scrap-vials", type=float, default=0.15,
                help="fraction of images that show any vials in the scrap (right) tray; most real photos show it empty")
ap.add_argument("--crooked-tilt", type=float, nargs=2, default=[20.0, 45.0], metavar=("MIN", "MAX"),
                help="how far a crooked cap tips, in degrees (tilt of the flanged cap)")
ap.add_argument("--crooked-shift", type=float, nargs=2, default=[0.0, 1.5], metavar=("MIN", "MAX"),
                help="how far a crooked cap also slides off the vial mouth, in mm")
ap.add_argument("--no-cap-flange", action="store_true",
                help="keep the twin's plain cup-shaped cap instead of the flanged top-hat shape")
ap.add_argument("--no-cap-ring", action="store_true",
                help="do not add the visible circle inside the caps, and keep the twin's original cap colour")
ap.add_argument("--cap-color", type=float, nargs=3, default=[0.95, 0.88, 0.03], metavar=("R", "G", "B"),
                help="cap body colour (0-1). Default is a saturated yellow; the twin's original (0.95 0.9 0.1) "
                     "renders pale and olive compared with the real caps")
ap.add_argument("--recess-color", type=float, nargs=3, default=[0.85, 0.65, 0.0], metavar=("R", "G", "B"),
                help="colour of the circle inside the cap (0-1); darker or more orange makes it stand out more")
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
    def lens_maps(roll_deg, dx, dy):
        """Where each final pixel samples the overscan render: undo the roll, then the shift (see twin_defects.project)."""
        c, s_ = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        vx, vy = und[..., 0] + dx / f, und[..., 1] + dy / f
        nx, ny = c * vx + s_ * vy, -s_ * vx + c * vy
        return (nx * f + rw / 2).astype(np.float32), (ny * f + rh / 2).astype(np.float32)

    da = td.DefectApplier(stage, root=root, crooked_tilt=tuple(a.crooked_tilt), crooked_shift=tuple(a.crooked_shift))
    lights = td.LightJitter(stage)
    scrap = td.ScrapVials(stage, root=root)
    if not a.no_cap_flange:
        td.CapFlange(stage, root=root)
    if not a.no_cap_ring:
        td.CapRing(stage, root=root, color=tuple(a.recess_color), cap_color=tuple(a.cap_color),
                   floor_z=None if a.no_cap_flange else td.CapFlange.DISH_Z,
                   radius=None if a.no_cap_flange else td.CapFlange.DISH_RADIUS - 0.02)
    print(f"found {len(lights.lights)} lights", flush=True)
    rng = random.Random(a.seed)
    defects = ["missing_cap", "crooked_cap"] + (["damaged_vial"] if a.include_damaged else [])
    weights = {"pass": 1 - a.defect_rate, **{d: a.defect_rate / len(defects) for d in defects}}
    colors = {"pass": (0, 200, 0), "missing_cap": (0, 165, 255), "crooked_cap": (255, 0, 160),
              "damaged_vial": (0, 0, 255), "no_vial": (255, 120, 0)}
    half = a.crop // 2

    rows, t0 = [], time.time()
    for i in range(a.count):
        labels = td.sample_labels_fill(td.GOOD_POSITIONS, rng, weights, a.partial_fill, gap_prob=a.gap_rate)
        da.apply(labels, rng)
        stops, tint = lights.apply(rng, a.light_jitter, a.light_offset, a.light_tint)
        n_scrap = scrap.apply(rng, a.scrap_vials)
        dx, dy = rng.uniform(-a.shift_px, a.shift_px), rng.uniform(-a.shift_px, a.shift_px)
        roll = a.roll_deg + (rng.uniform(-a.roll_jitter, a.roll_jitter) if a.roll_jitter > 0 else 0.0)
        centers = da.pixel_centers(cam, (dx, dy), roll)
        for _ in range(2):
            kit.update()
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
        data = np.asarray(ann.get_data())
        if data.ndim != 3 or data.shape[:2] != (rh, rw) or data.shape[2] < 3:
            raise RuntimeError(f"RGB capture returned shape {data.shape}; check GPU/RTX startup errors")
        bgr = data[..., :3][..., ::-1].copy()
        mapx, mapy = lens_maps(roll, dx, dy)
        final = cv2.remap(bgr, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        mean_level = float(final.mean())

        name = f"img_{i:04d}"
        cv2.imwrite(str(out / "images" / f"{name}.png"), final)
        note = final.copy()
        for pos in td.GOOD_POSITIONS:
            cls, (cx, cy) = labels[pos], centers[pos]
            x0 = int(min(max(round(cx) - half, 0), W - a.crop)); y0 = int(min(max(round(cy) - half, 0), H - a.crop))
            crop_file = Path("crops") / cls / f"{name}_{pos}.png"
            (out / crop_file).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / crop_file), final[y0:y0 + a.crop, x0:x0 + a.crop])
            rows.append([f"images/{name}.png", pos, cls, round(float(cx), 1), round(float(cy), 1), str(crop_file),
                         round(stops, 2), round(tint, 2), round(dx, 1), round(dy, 1), round(roll, 2),
                         round(mean_level, 1)])
            if cls != "pass":
                cv2.rectangle(note, (x0, y0), (x0 + a.crop, y0 + a.crop), colors[cls], 1)
                cv2.putText(note, pos, (x0, max(y0 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[cls], 1)
        for k, (cname, col) in enumerate(c for c in colors.items() if c[0] != "pass" and (c[0] != "damaged_vial" or a.include_damaged)):
            cv2.putText(note, cname, (6, 14 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.imwrite(str(out / "annotated" / f"{name}.png"), note)
        n_def = sum(c not in ("pass", "no_vial") for c in labels.values())
        n_empty = sum(c == "no_vial" for c in labels.values())
        print(f"[{i + 1}/{a.count}] {name}: {30 - n_empty} filled, {n_def} defects, light {stops:+.2f} stops "
              f"tint {tint:.2f}, roll {roll:+.1f}, brightness {mean_level:.0f}, {n_scrap} scrap vials, {time.time() - t0:.0f}s elapsed",
              flush=True)
        # Keep labels.csv current so a stopped run still leaves usable data.
        with open(out / "labels.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["image", "position", "class", "cx", "cy", "crop", "light_stops", "tint", "shift_x", "shift_y",
                        "roll_deg", "image_mean"]); w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r[2]] = counts.get(r[2], 0) + 1
    print("done:", a.count, "images,", len(rows), "labeled crops", counts, "->", out, flush=True)
except BaseException:
    traceback.print_exc()
    kit.close(exit_code=1)
    raise
kit.close()