"""Turn a CapSealCam render into the real cap-seal camera image (lens distortion + crop).

CapSealCam is a pinhole camera with 1.2x overscan. Render it at the camera JSON's `render_px`
(768x576), then run this to get the final 640x480 image with the real lens distortion.
The fill camera has no distortion and needs no post-processing.

    python tools/apply_lens.py render.png cameras/CapSealCam.json out.png

Needs numpy + opencv-python.
"""
import json, sys
from pathlib import Path
import numpy as np, cv2

src, cam_json, dst = sys.argv[1:4]
cam = json.load(open(cam_json))
W, H = cam["final_image_px"]
f, k1 = cam["focal_px_final"], cam["lens_k1"]
img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"cannot read image: {src}")
ph, pw = img.shape[:2]
if [pw, ph] != cam["render_px"]:
    raise ValueError(f"expected {cam['render_px']} input pixels, got {[pw, ph]}")

K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
D = np.array([k1, 0, 0, 0, 0.0])
u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
und = cv2.undistortPoints(np.stack([u.ravel(), v.ravel()], 1)[:, None, :], K, D).reshape(H, W, 2)
mapx = (und[..., 0] * f + pw / 2).astype(np.float32)          # distorted output pixel -> pinhole render pixel
mapy = (und[..., 1] * f + ph / 2).astype(np.float32)
Path(dst).resolve().parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(dst, cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)):
    raise OSError(f"could not write {dst}")
print(f"{src} ({pw}x{ph}) -> {dst} ({W}x{H})")
