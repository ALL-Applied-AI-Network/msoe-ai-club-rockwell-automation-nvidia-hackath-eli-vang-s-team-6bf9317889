"""Render one image from a camera of the digital twin with Isaac Sim (headless).

    ./python.sh tools/render_example.py --camera CapSealCam --out cap_seal.png
    ./python.sh tools/render_example.py --camera FillCam --frame 301 --out fill.png
    ./python.sh tools/render_example.py --camera CapSealCam --stage stations/tray_inspection/tray_inspection.usdc --out cap_seal.png
    ./python.sh tools/render_example.py --camera FillCam --stage stations/fill_station/fill_station.usda --out fill.png

Run with Isaac Sim's python (python.sh in the Isaac Sim install). By default it opens the full line
(csi_line.usda); --stage opens a single station file instead. The camera settings come from cameras/<name>.json.
CapSealCam is rendered with overscan and then given the real lens distortion (tools/apply_lens.py).
"""
import argparse, json, os, subprocess, sys, traceback
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--camera", required=True, choices=["CapSealCam", "FillCam"],
                help="camera name; settings come from cameras/<name>.json")
ap.add_argument("--stage", default=os.path.join(HERE, "csi_line.usda"), help="csi_line.usda or one station file")
ap.add_argument("--frame", type=int, default=1, help="time code (fill station animation: 1-686 at 30 fps)")
ap.add_argument("--out", default="render.png")
ap.add_argument("--subframes", type=int, default=32, help="path-tracing subframes (quality)")
ap.add_argument("--kit-arg", action="append", default=[], help="extra Isaac Sim / Kit argument (repeatable)")
a = ap.parse_args()

cam = json.load(open(os.path.join(HERE, "cameras", a.camera + ".json")))
stage_path = Path(a.stage).expanduser()
if not stage_path.is_absolute():
    stage_path = Path(HERE) / stage_path
stage_path = str(stage_path.resolve())
if not os.path.isfile(stage_path):
    ap.error(f"stage does not exist: {stage_path}")
if a.frame < 1 or (a.camera == "FillCam" and a.frame > 686):
    ap.error("use a positive time code; the fill animation covers 1-686")
if a.subframes < 1:
    ap.error("--subframes must be positive")
Path(a.out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
a.out = str(Path(a.out).expanduser().resolve())
rw, rh = cam["render_px"]

from isaacsim import SimulationApp
kit = SimulationApp({"headless": True, "renderer": "PathTracing", "width": rw, "height": rh, "extra_args": a.kit_arg})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import numpy as np, cv2
    import omni.usd, omni.timeline
    import omni.replicator.core as rep

    ctx = omni.usd.get_context()
    if not ctx.open_stage(stage_path):
        raise RuntimeError(f"could not open stage: {stage_path}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(10):
        kit.update()
    stage = ctx.get_stage()
    prim = next((p for p in [cam["prim_in_csi_line"], cam["prim_in_station_file"]]
                 if stage.GetPrimAtPath(p).IsValid()), None)
    if prim is None:
        raise RuntimeError(f"camera {a.camera} not found in {stage_path}")

    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(a.frame / stage.GetTimeCodesPerSecond())
    kit.update()

    rp = rep.create.render_product(prim, (rw, rh))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()

    # RTX's first sync of a scaled camera (CapSealCam carries the 0.0254 inch scale) misreads its
    # clippingRange and renders black. Changing and restoring the value in the session layer forces a
    # correct re-sync; the authored camera in the USD file is untouched.
    from pxr import Gf, UsdGeom
    clip = UsdGeom.Camera(stage.GetPrimAtPath(prim)).GetClippingRangeAttr()
    near_far = clip.Get()
    stage.SetEditTarget(stage.GetSessionLayer())
    clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1]))
    kit.update()
    clip.Set(near_far)
    kit.update()

    for _ in range(2):
        rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
    data = np.asarray(ann.get_data())
    if data.ndim != 3 or data.shape[:2] != (rh, rw) or data.shape[2] < 3:
        raise RuntimeError(f"RGB capture returned shape {data.shape}; check GPU/RTX startup errors")
    img = data[..., :3].copy()

    # Save and post-process before close(): Isaac's default fast shutdown exits
    # the process and will never execute statements after kit.close().
    if cam.get("overscan", 1.0) != 1.0:
        tmp = a.out + ".pinhole.png"
        if not cv2.imwrite(tmp, img[..., ::-1]):
            raise OSError(f"could not write {tmp}")
        try:
            subprocess.run([sys.executable, os.path.join(HERE, "tools", "apply_lens.py"), tmp,
                            os.path.join(HERE, "cameras", a.camera + ".json"), a.out], check=True)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    elif not cv2.imwrite(a.out, img[..., ::-1]):
        raise OSError(f"could not write {a.out}")
    print("wrote", a.out, flush=True)
except BaseException:
    traceback.print_exc()
    # Preserve failure status even when Kit's fast shutdown terminates Python.
    kit.close(exit_code=1)
    raise
kit.close()
