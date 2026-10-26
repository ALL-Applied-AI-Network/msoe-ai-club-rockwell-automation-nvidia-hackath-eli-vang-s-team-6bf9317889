"""Helpers for generate_data.py: apply per-position defects to the tray inspection twin and find
where each numbered position lands in the final CapSealCam image.

Nothing here needs Isaac Sim, only pxr (usd-core), numpy and opencv, so it can be tested on its own.
All edits go into the stage's SESSION layer, so tray_inspection.usdc on disk is never changed.

Classes: pass, missing_cap, crooked_cap, damaged_vial, no_vial.
damaged_vial is an approximation (the twin has no damaged-vial model): the body is crushed, stretched
sideways and tilted so it leans out of its hole, with the cap still on it.
"""
import json
import math

import cv2
import numpy as np
from pxr import Gf, Usd, UsdGeom

CLASSES = ["pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"]
DEFECTS = [c for c in CLASSES if c != "pass"]
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


def _world_bbox(prim):
    bb = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), "default").ComputeAlignedRange()
    return bb.GetMin(), bb.GetMax()


class DefectApplier:
    def __init__(self, stage, positions=GOOD_POSITIONS, root="/TrayInspection"):
        self.stage = stage
        self.positions = positions
        self.session = stage.GetSessionLayer()
        self.tube, self.cap = {}, {}
        self.orig_world = {}       # (prim path) -> original world matrix
        self.parent_inv = {}       # (prim path) -> inverse of the parent's world matrix
        self.centers = {}          # world xyz of each aperture, on the fixture surface
        self.cap_base = {}         # world xyz of the centre of each cap's underside
        self.tube_base = {}        # world xyz of the centre of each tube's bottom
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for pos in positions:
            t = stage.GetPrimAtPath(f"{root}/{pos}_tube")
            c = stage.GetPrimAtPath(f"{root}/{pos}_cap")
            if not t or not c:
                raise RuntimeError(f"{pos}: tube/cap prim not found under {root}")
            self.tube[pos], self.cap[pos] = t, c
            for p in (t, c):
                self.orig_world[p.GetPath()] = cache.GetLocalToWorldTransform(p)
                self.parent_inv[p.GetPath()] = cache.GetLocalToWorldTransform(p.GetParent()).GetInverse()
            tp = cache.GetLocalToWorldTransform(t).ExtractTranslation()
            self.centers[pos] = (tp[0], tp[1], RIM_Z)
            lo, hi = _world_bbox(c)
            self.cap_base[pos] = Gf.Vec3d((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2])
            lo, hi = _world_bbox(t)
            self.tube_base[pos] = Gf.Vec3d((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2])
        # One transform op per tube and cap, authored in the session layer, so each image just sets matrices.
        self.op = {}
        with Usd.EditContext(stage, self.session):
            for pos in positions:
                for p in (self.tube[pos], self.cap[pos]):
                    xf = UsdGeom.Xformable(p)
                    xf.ClearXformOpOrder()
                    self.op[p.GetPath()] = xf.AddTransformOp()
        self.reset()

    # -- low-level helpers ---------------------------------------------------------------------
    def _set_visible(self, prim, visible):
        with Usd.EditContext(self.stage, self.session):
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set("inherited" if visible else "invisible")

    def _set_world(self, prim, world):
        key = prim.GetPath()
        with Usd.EditContext(self.stage, self.session):
            self.op[key].Set(world * self.parent_inv[key])

    def reset(self):
        for pos in self.positions:
            for p in (self.tube[pos], self.cap[pos]):
                self._set_visible(p, True)
                self._set_world(p, self.orig_world[p.GetPath()])

    @staticmethod
    def _about(pivot, m):
        """Matrix that applies m about `pivot` (row-vector convention: translate, transform, translate back)."""
        return Gf.Matrix4d().SetTranslate(-pivot) * m * Gf.Matrix4d().SetTranslate(pivot)

    @staticmethod
    def _tilt(rng, tilt_deg):
        a = rng.uniform(0, 2 * math.pi)
        axis = Gf.Vec3d(math.cos(a), math.sin(a), 0)
        return Gf.Matrix4d().SetRotate(Gf.Rotation(axis, rng.uniform(*tilt_deg)))

    # -- defects ---------------------------------------------------------------------------------
    def tilt_cap(self, pos, rng, tilt_deg=(15.0, 40.0), shift_mm=3.0, lift_mm=(0.0, 4.0)):
        """Tilt the cap about the centre of its underside, in a random horizontal direction, and nudge it."""
        about = self._about(self.cap_base[pos], self._tilt(rng, tilt_deg))
        off = Gf.Vec3d(rng.uniform(-shift_mm, shift_mm), rng.uniform(-shift_mm, shift_mm),
                       rng.uniform(*lift_mm)) / 1000.0
        self._set_world(self.cap[pos], self.orig_world[self.cap[pos].GetPath()] * about
                        * Gf.Matrix4d().SetTranslate(off))

    def damage_vial(self, pos, rng, tilt_deg=(10.0, 30.0), squash=(0.55, 0.85), widen=(1.05, 1.3)):
        """Crush the body (shorter and wider), then tilt it about the hole's rim so it leans out of the fixture.
        The cap stays attached to the top of the body."""
        c = self.centers[pos]
        pivot = Gf.Vec3d(c[0], c[1], RIM_Z)   # the body leans out of the hole, pivoting at the rim
        sz, sxy = rng.uniform(*squash), rng.uniform(*widen)
        scale = Gf.Matrix4d().SetScale(Gf.Vec3d(sxy, rng.uniform(1.0, sxy), sz))
        rot = self._tilt(rng, tilt_deg)
        body = self._about(pivot, scale * rot)
        self._set_world(self.tube[pos], self.orig_world[self.tube[pos].GetPath()] * body)
        # Cap: same tilt, moved so its underside sits on the new top of the body.
        cap_rot = self._about(pivot, rot)
        top_old = self.cap_base[pos]
        top_new = body.Transform(top_old)          # where the crushed body's top ends up
        top_cap = cap_rot.Transform(top_old)       # where a rigid tilt would put the cap
        self._set_world(self.cap[pos], self.orig_world[self.cap[pos].GetPath()] * cap_rot
                        * Gf.Matrix4d().SetTranslate(top_new - top_cap))

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
            elif cls == "damaged_vial":
                self.damage_vial(pos, rng)
            else:
                raise ValueError(f"unknown class {cls}")

    def pixel_centers(self, cam):
        pts = np.array([self.centers[p] for p in self.positions])
        return dict(zip(self.positions, project(cam, pts)))


def sample_labels(positions, rng, weights):
    classes = list(weights)
    w = [weights[c] for c in classes]
    return {p: rng.choices(classes, w)[0] for p in positions}