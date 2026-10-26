"""Helpers for generate_dataset.py: apply per-position defects to the tray inspection twin and find
where each numbered position lands in the final CapSealCam image.

Nothing here needs Isaac Sim, only pxr (usd-core), numpy and opencv, so it can be tested on its own.
All edits go into the stage's SESSION layer, so tray_inspection.usdc on disk is never changed.
"""
import json
import math
import random

import cv2
import numpy as np
from pxr import Gf, Usd, UsdGeom

CLASSES = ["pass", "missing_cap", "crooked_cap", "no_vial"]
RIM_Z = 1.222  # world z (m) of the fixture top surface; tray_good_top spans 1.2187 - 1.2250
GOOD_POSITIONS = [f"G{i:02d}" for i in range(1, 31)]


def load_camera(cam_json_path):
    return json.load(open(cam_json_path))


def project(cam, pts_world):
    """World points (N,3) in meters -> pixels (N,2) in the FINAL 640x480 image (lens distortion included)."""
    f, k1 = cam["focal_px_final"], cam["lens_k1"]
    W, H = cam["final_image_px"]
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    R = np.array(cam["R_world_to_cv"], dtype=np.float64)
    C = np.array(cam["position_m"], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    tvec = (-R @ C).reshape(3, 1)
    px, _ = cv2.projectPoints(np.asarray(pts_world, dtype=np.float64).reshape(-1, 1, 3), rvec, tvec, K,
                              np.array([k1, 0, 0, 0, 0.0]))
    return px.reshape(-1, 2)


class DefectApplier:
    def __init__(self, stage, positions=GOOD_POSITIONS, root="/TrayInspection"):
        self.stage = stage
        self.positions = positions
        self.session = stage.GetSessionLayer()
        self.tube, self.cap = {}, {}
        self.orig_world = {}      # cap world matrix, untouched
        self.parent_inv = {}      # inverse of the cap's parent world matrix
        self.centers = {}         # world xyz of each aperture, on the fixture surface
        self.cap_base = {}        # world xyz of the centre of each cap's underside
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for pos in positions:
            t = stage.GetPrimAtPath(f"{root}/{pos}_tube")
            c = stage.GetPrimAtPath(f"{root}/{pos}_cap")
            if not t or not c:
                raise RuntimeError(f"{pos}: tube/cap prim not found under {root}")
            self.tube[pos], self.cap[pos] = t, c
            self.orig_world[pos] = cache.GetLocalToWorldTransform(c)
            self.parent_inv[pos] = cache.GetLocalToWorldTransform(c.GetParent()).GetInverse()
            tp = cache.GetLocalToWorldTransform(t).ExtractTranslation()
            self.centers[pos] = (tp[0], tp[1], RIM_Z)
            bb = UsdGeom.Imageable(c).ComputeWorldBound(Usd.TimeCode.Default(), "default").ComputeAlignedRange()
            mid = (bb.GetMin() + bb.GetMax()) / 2
            self.cap_base[pos] = Gf.Vec3d(mid[0], mid[1], bb.GetMin()[2])
        # One transform op per cap, authored in the session layer, so each image just sets its matrix.
        with Usd.EditContext(stage, self.session):
            self.cap_op = {}
            for pos in positions:
                xf = UsdGeom.Xformable(self.cap[pos])
                xf.ClearXformOpOrder()
                self.cap_op[pos] = xf.AddTransformOp()
        self.reset()

    def _set_visible(self, prim, visible):
        with Usd.EditContext(self.stage, self.session):
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set("inherited" if visible else "invisible")

    def _set_cap_world(self, pos, world):
        with Usd.EditContext(self.stage, self.session):
            self.cap_op[pos].Set(world * self.parent_inv[pos])

    def reset(self):
        for pos in self.positions:
            self._set_visible(self.tube[pos], True)
            self._set_visible(self.cap[pos], True)
            self._set_cap_world(pos, self.orig_world[pos])

    def tilt_cap(self, pos, rng, tilt_deg=(15.0, 40.0), shift_mm=3.0, lift_mm=(0.0, 4.0)):
        """Tilt the cap about the centre of its underside, in a random horizontal direction, and nudge it."""
        axis_ang = rng.uniform(0, 2 * math.pi)
        axis = Gf.Vec3d(math.cos(axis_ang), math.sin(axis_ang), 0)
        rot = Gf.Matrix4d().SetRotate(Gf.Rotation(axis, rng.uniform(*tilt_deg)))
        p = self.cap_base[pos]
        about_pivot = Gf.Matrix4d().SetTranslate(-p) * rot * Gf.Matrix4d().SetTranslate(p)
        off = Gf.Vec3d(rng.uniform(-shift_mm, shift_mm), rng.uniform(-shift_mm, shift_mm),
                       rng.uniform(*lift_mm)) / 1000.0
        self._set_cap_world(pos, self.orig_world[pos] * about_pivot * Gf.Matrix4d().SetTranslate(off))

    def apply(self, labels, rng):
        """labels: {position: class}. Resets everything first, then applies each defect."""
        self.reset()
        for pos, cls in labels.items():
            if cls == "pass":
                continue
            if cls == "missing_cap":
                self._set_visible(self.cap[pos], False)
            elif cls == "no_vial":
                self._set_visible(self.cap[pos], False)
                self._set_visible(self.tube[pos], False)
            elif cls == "crooked_cap":
                self.tilt_cap(pos, rng)
            else:
                raise ValueError(f"unknown class {cls}")

    def pixel_centers(self, cam):
        pts = np.array([self.centers[p] for p in self.positions])
        return dict(zip(self.positions, project(cam, pts)))


def sample_labels(positions, rng, weights):
    classes = list(weights)
    w = [weights[c] for c in classes]
    return {p: rng.choices(classes, w)[0] for p in positions}