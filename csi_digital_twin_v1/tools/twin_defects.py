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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

CLASSES = ["pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"]
DEFECTS = [c for c in CLASSES if c != "pass"]
RIM_Z = 1.222  # world z (m) of the fixture top surface; tray_good_top spans 1.2187 - 1.2250
GOOD_POSITIONS = [f"G{i:02d}" for i in range(1, 31)]


def load_camera(cam_json_path):
    return json.load(open(cam_json_path))


def roll_matrix(roll_deg):
    """2x2 rotation for image coordinates (x right, y down). Positive = the picture turns clockwise."""
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    return np.array([[c, -s], [s, c]])


def project(cam, pts_world, shift=(0.0, 0.0), roll_deg=0.0):
    """World points (N,3) in meters -> pixels (N,2) in the FINAL 640x480 image (lens distortion included).

    roll_deg and shift=(dx, dy) must match what generate_data.py does to the lens remap: the camera is rolled
    (positive = picture turned clockwise about the image centre) and then the picture is moved by -shift."""
    f, k1 = cam["focal_px_final"], cam["lens_k1"]
    W, H = cam["final_image_px"]
    R = np.array(cam["R_world_to_cv"], dtype=np.float64)
    C = np.array(cam["position_m"], dtype=np.float64)
    pc = (np.asarray(pts_world, dtype=np.float64).reshape(-1, 3) - C) @ R.T      # camera coordinates (OpenCV)
    n = (pc[:, :2] / pc[:, 2:3]) @ roll_matrix(roll_deg).T - np.array(shift, dtype=np.float64) / f
    r2 = (n ** 2).sum(1, keepdims=True)
    return n * (1 + k1 * r2) * f + np.array([W / 2, H / 2])                     # radial model, k1 only


def _world_bbox(prim):
    bb = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), "default").ComputeAlignedRange()
    return bb.GetMin(), bb.GetMax()


class DefectApplier:
    def __init__(self, stage, positions=GOOD_POSITIONS, root="/TrayInspection",
                 crooked_tilt=(20.0, 45.0), crooked_shift=(0.0, 1.5)):
        self.crooked_tilt = crooked_tilt      # degrees, how far a crooked cap tips
        self.crooked_shift = crooked_shift    # mm, how far it also slides off the mouth
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
    CAP_RADIUS = 0.017   # m (the cap mesh is 0.67 in in radius)

    def tilt_cap(self, pos, rng, tilt_deg=None, extra_shift_mm=None, side_mm=1.0):
        """A cap that is not seated: it rests on one edge of the vial mouth with the other side lifted, and sits
        shifted off the mouth, like the real crooked caps (tilted about its rim, not about its centre)."""
        tilt_deg = tilt_deg or self.crooked_tilt
        extra_shift_mm = extra_shift_mm or self.crooked_shift
        a = rng.uniform(0, 2 * math.pi)
        axis = Gf.Vec3d(math.cos(a), math.sin(a), 0)            # horizontal tilt axis
        n = Gf.Vec3d(-math.sin(a), math.cos(a), 0)              # horizontal, perpendicular to the axis (z x axis)
        # Tip the cap about its own centre line (it stays over the vial mouth), then raise it just enough that the
        # low edge still rests on the rim of the mouth and the opposite edge is lifted.
        base = self.cap_base[pos]
        pivot = base + Gf.Vec3d(0, 0, 0.012)
        rot = Gf.Matrix4d().SetRotate(Gf.Rotation(axis, rng.uniform(*tilt_deg)))   # lifts the +n side
        about = self._about(pivot, rot)
        rim = base - n * 0.0146                      # where the low edge of the cap meets the rim of the mouth
        lift = rim[2] - about.Transform(rim)[2]
        off = Gf.Vec3d(0, 0, lift) + (-n * rng.uniform(*extra_shift_mm) + axis * rng.uniform(-side_mm, side_mm)) / 1000.0
        self._set_world(self.cap[pos], self.orig_world[self.cap[pos].GetPath()] * about
                        * Gf.Matrix4d().SetTranslate(off))

    def damage_vial(self, pos, rng, tilt_deg=(20.0, 50.0), squash=(0.55, 0.85), widen=(1.05, 1.3)):
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

    def pixel_centers(self, cam, shift=(0.0, 0.0), roll_deg=0.0):
        pts = np.array([self.centers[p] for p in self.positions])
        return dict(zip(self.positions, project(cam, pts, shift, roll_deg)))


class CapRing:
    """Makes the recessed circle on top of each cap visible.

    The cap model already has a recess in its top (about 1 mm deep, radius 0.43 in), but it has the same colour
    as the rest of the cap, so it barely shows. This lays a thin, slightly darker yellow disc on the recess floor
    as a child of every *_cap prim, so it moves with the cap (tilted, damaged) and hides with it (missing cap).
    Units below are the cap's own local units (inches); the cap mesh spans z 0 to 0.75, recess floor at z 0.71.
    """
    RECESS_RADIUS = 0.43
    FLOOR_Z = 0.71

    def __init__(self, stage, root="/TrayInspection", color=(0.85, 0.65, 0.0), cap_color=(0.95, 0.88, 0.03),
                 floor_z=None, radius=None):
        floor_z = self.FLOOR_Z if floor_z is None else floor_z
        radius = self.RECESS_RADIUS if radius is None else radius
        session = stage.GetSessionLayer()
        with Usd.EditContext(stage, session):
            if cap_color is not None:
                # The twin's caps render pale and olive (measured blue 101 vs 56 on the real caps); make the
                # body colour a more saturated yellow.
                mat_prim = stage.GetPrimAtPath(f"{root}/_materials/tw_cap_yellow")
                for child in (mat_prim.GetChildren() if mat_prim else []):
                    attr = child.GetAttribute("inputs:diffuseColor")
                    if attr:
                        attr.Set(Gf.Vec3f(*cap_color))
            mats = f"{root}/_materials"
            if not stage.GetPrimAtPath(mats):
                UsdGeom.Scope.Define(stage, mats)
            mat = UsdShade.Material.Define(stage, f"{mats}/tw_cap_recess")
            sh = UsdShade.Shader.Define(stage, f"{mats}/tw_cap_recess/Shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self.rings = []
            for p in stage.GetPrimAtPath(root).GetChildren():
                if not p.GetName().endswith("_cap"):
                    continue
                cyl = UsdGeom.Cylinder.Define(stage, p.GetPath().AppendChild("recess_ring"))
                cyl.CreateAxisAttr("Z")
                cyl.CreateRadiusAttr(radius)
                cyl.CreateHeightAttr(0.012)
                UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(0, 0, floor_z + 0.004))
                UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(mat)
                self.rings.append(cyl.GetPrim())


class CapFlange:
    """Replaces the twin's plain cup-shaped cap with the real cap's shape: a wide thin flange on top of a narrower
    plug body (an upside-down top hat), with a hollow underside and a dished (inward dip) top face.
    The original cap mesh is hidden and a revolved mesh named `cap_hat` is added under every *_cap prim, so it
    moves with tilt/damage and hides with the cap. Units are the cap's local inches (original mesh: z 0 to 0.75,
    radius 0.668)."""
    # (radius, z) outline of the cap's cross-section, revolved around Z
    PROFILE = [(0.46, 0.0), (0.53, 0.0), (0.53, 0.61), (0.668, 0.61), (0.668, 0.75), (0.47, 0.75),
               (0.42, 0.52), (0.0, 0.52), (0.0, 0.35), (0.46, 0.35)]
    DISH_RADIUS = 0.42
    DISH_Z = 0.52
    SEGMENTS = 64

    @classmethod
    def build(cls):
        """Returns (points, faceVertexCounts, faceVertexIndices) of the revolved cap."""
        n, prof = cls.SEGMENTS, cls.PROFILE
        pts = []
        for i in range(n):
            t = 2 * math.pi * i / n
            for r, z in prof:
                pts.append((r * math.cos(t), r * math.sin(t), z))
        k = len(prof)
        counts, idx = [], []
        for i in range(n):
            j = (i + 1) % n
            for m in range(k):
                m2 = (m + 1) % k
                quad = [i * k + m, j * k + m, j * k + m2, i * k + m2]
                # skip the zero-width faces along the axis
                if prof[m][0] == 0.0 and prof[m2][0] == 0.0:
                    continue
                counts.append(4)
                idx += quad
        return pts, counts, idx

    def __init__(self, stage, root="/TrayInspection"):
        session = stage.GetSessionLayer()
        pts, counts, idx = self.build()
        with Usd.EditContext(stage, session):
            mat = UsdShade.Material(stage.GetPrimAtPath(f"{root}/_materials/tw_cap_yellow"))
            for p in stage.GetPrimAtPath(root).GetChildren():
                if not p.GetName().endswith("_cap"):
                    continue
                for m in p.GetChildren():
                    if m.IsA(UsdGeom.Mesh):
                        UsdGeom.Imageable(m).MakeInvisible()
                hat = UsdGeom.Mesh.Define(stage, p.GetPath().AppendChild("cap_hat"))
                hat.CreatePointsAttr([Gf.Vec3f(*q) for q in pts])
                hat.CreateFaceVertexCountsAttr(counts)
                hat.CreateFaceVertexIndicesAttr(idx)
                hat.CreateDoubleSidedAttr(True)
                hat.CreateSubdivisionSchemeAttr("none")
                if mat:
                    UsdShade.MaterialBindingAPI.Apply(hat.GetPrim()).Bind(mat)


class ScrapVials:
    """The scrap fixture (right tray) holds the twin's own S01-S05 vials. Most real photos show it empty, so by
    default hide them, and in a fraction of images show a random 1-5 of them."""

    def __init__(self, stage, root="/TrayInspection"):
        self.stage = stage
        self.session = stage.GetSessionLayer()
        self.groups = {}   # position -> prims (tube and cap, where they exist)
        for p in stage.GetPrimAtPath(root).GetChildren():
            n = p.GetName()
            if n.startswith("S") and (n.endswith("_tube") or n.endswith("_cap")):
                self.groups.setdefault(n.split("_")[0], []).append(p)

    def _show(self, positions):
        with Usd.EditContext(self.stage, self.session):
            for pos, prims in self.groups.items():
                for p in prims:
                    UsdGeom.Imageable(p).GetVisibilityAttr().Set("inherited" if pos in positions else "invisible")

    def apply(self, rng, prob):
        """With probability `prob`, show a random subset of the scrap vials. Returns how many are shown."""
        shown = set()
        if rng.random() < prob and self.groups:
            shown = set(rng.sample(sorted(self.groups), rng.randint(1, len(self.groups))))
        self._show(shown)
        return len(shown)


class LightJitter:
    """Sets the scene's lighting for each image: a brightness offset plus random variation (exposure, in stops)
    and a random warm-to-cool colour shift. Labels are unaffected.

    offset: constant added to every light's exposure (the twin is about 1 stop darker than the real camera).
    stops:  max random change on top of that. One global value per image plus a smaller per-light value.
    tint:   max colour shift in stops. The real images range from warm (tray_000001) to cooler (tray_000122);
            0 = the twin's original warm tone, 1 = red down and blue up by `tint` stops each.
    """

    def __init__(self, stage):
        self.stage = stage
        self.session = stage.GetSessionLayer()
        self.lights = []   # (light api, original exposure, original colour)
        for p in stage.Traverse():
            if p.HasAPI(UsdLux.LightAPI) or p.GetTypeName().endswith("Light"):
                api = UsdLux.LightAPI(p)
                e = api.GetExposureAttr().Get() if api.GetExposureAttr() else None
                c = api.GetColorAttr().Get() if api.GetColorAttr() else None
                self.lights.append((api, float(e) if e is not None else 0.0,
                                    tuple(c) if c is not None else (1.0, 1.0, 1.0)))

    def _set(self, api, exposure, color):
        with Usd.EditContext(self.stage, self.session):
            api.CreateExposureAttr().Set(float(exposure))
            api.CreateColorAttr().Set(Gf.Vec3f(*[float(x) for x in color]))

    def reset(self):
        for api, e, c in self.lights:
            self._set(api, e, c)

    def apply(self, rng, stops=0.0, offset=0.0, tint=0.0):
        """Returns (global exposure change in stops, tint position 0-1)."""
        g = offset + (rng.uniform(-stops, stops) if stops > 0 else 0.0)
        t = rng.uniform(0.0, 1.0) if tint > 0 else 0.0
        gain = (2 ** (-tint * t), 1.0, 2 ** (tint * t))
        for api, e, c in self.lights:
            local = rng.uniform(-stops / 2, stops / 2) if stops > 0 else 0.0
            self._set(api, e + g + local, [ci * k for ci, k in zip(c, gain)])
        return g, t


def sample_labels(positions, rng, weights):
    classes = list(weights)
    w = [weights[c] for c in classes]
    return {p: rng.choices(classes, w)[0] for p in positions}


def sample_labels_fill(positions, rng, defect_weights, partial_prob=0.7, min_filled=3, gap_prob=0.03):
    """Labels that follow how the robot actually fills the fixture.

    The robot fills G01, G02, ... in order (top right, down, then one column left). Most real photos are taken
    mid-run, so with probability `partial_prob` only the first n positions hold vials and the rest are empty
    (no_vial, waiting to be filled). Otherwise the fixture is full. Among filled positions, `gap_prob` is the
    chance of an empty hole inside the filled area (a missed placement); the others are drawn from
    `defect_weights` (pass and the vial defects).
    """
    n = len(positions)
    filled = rng.randint(min_filled, n - 1) if rng.random() < partial_prob else n
    classes = list(defect_weights)
    w = [defect_weights[c] for c in classes]
    labels = {}
    for i, pos in enumerate(positions):
        if i >= filled or rng.random() < gap_prob:
            labels[pos] = "no_vial"
        else:
            labels[pos] = rng.choices(classes, w)[0]
    return labels