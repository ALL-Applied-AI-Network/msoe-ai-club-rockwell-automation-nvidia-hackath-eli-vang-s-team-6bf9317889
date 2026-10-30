#!/usr/bin/env python3
"""Calibrate a new camera view: click 4 fixture corners once, and every photo from that camera can be scored.

    python app/calibrate.py path/to/one_photo.jpg            then open http://localhost:8001

Click the centres of positions G01, G05, G26 and G30 (the four corner vials) in that order, check that the 30
preview dots land on the vial centres, and press Save. It writes calibration.json next to the photo.
Then score with:  CALIBRATION=<that file> bash run_brev.sh inspect <photo folder>

Each photo is warped onto the golden fixture view (config/golden/scene.png) with the perspective transform
from the 4 clicks, so position, scale and rotation match what the model was trained on. Needs only Python 3.
Only valid while the camera stays where it was for the calibration photo; re-calibrate if it moves.
"""
import argparse
import base64
import csv
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORNERS = ("G01", "G05", "G26", "G30")


def solve(a, b):
    """Gaussian elimination with partial pivoting (stdlib only)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        if abs(m[c][c]) < 1e-12:
            raise ValueError("corner clicks are degenerate (in a line?)")
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def perspective(src, dst):
    """8 coefficients mapping src (golden view) points to dst (photo) points, in PIL's PERSPECTIVE order."""
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    return solve(a, b)


def apply(c, x, y):
    w = c[6] * x + c[7] * y + 1
    return (c[0] * x + c[1] * y + c[2]) / w, (c[3] * x + c[4] * y + c[5]) / w


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Fixture Calibration</title><style>
body{margin:0;background:#0f1216;color:#e6e9ee;font:14px system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid #2a313b;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
button{font:inherit;color:#e6e9ee;background:#1e232b;border:1px solid #2a313b;border-radius:6px;padding:6px 12px;cursor:pointer}
button.primary{background:#4aa3ff;border-color:#4aa3ff;color:#fff}#msg{color:#f2a93b}
#wrap{padding:12px}svg{width:100%;max-height:calc(100vh - 80px);background:#000;cursor:crosshair}
</style></head><body><header><b>Fixture calibration</b><span id="msg"></span><span style="flex:1"></span>
<button id="undo">Undo click</button><button id="save" class="primary" disabled>Save calibration</button></header>
<div id="wrap"><svg id="svg"></svg></div><script>
const D = __DATA__; const pts = []; const svg = document.getElementById("svg");
// size as the browser shows it (phone JPEGs are rotated by their EXIF tag; scene_inspect.py does the same)
const probe = new Image(); probe.onload = () => { D.width = probe.naturalWidth; D.height = probe.naturalHeight;
  svg.setAttribute("viewBox", `0 0 ${D.width} ${D.height}`); draw(); }; probe.src = D.image;
function msg(){ document.getElementById("msg").textContent = pts.length < 4
  ? `Click the centre of ${D.corners[pts.length]} (${pts.length + 1} of 4). Corners: G01 = first vial the robot places (top of the first column), G05 = bottom of that column, G26 = top of the last column, G30 = bottom of the last column.`
  : "Check the 30 dots sit on the vial centres, then Save."; }
async function draw(){
  let h = `<image href="${D.image}" width="${D.width}" height="${D.height}"/>`;
  pts.forEach((p,i)=> h += `<circle cx="${p[0]}" cy="${p[1]}" r="${D.width/120}" fill="#ef4b4b"/><text x="${p[0]+8}" y="${p[1]-8}" fill="#fff" font-size="${D.width/50}">${D.corners[i]}</text>`);
  if (pts.length === 4) {
    const r = await (await fetch("/preview", {method:"POST", body: JSON.stringify({clicks: pts})})).json();
    if (r.error) { document.getElementById("msg").textContent = r.error; }
    else r.points.forEach(p => h += `<circle cx="${p.cx}" cy="${p.cy}" r="${D.width/180}" fill="#2fbf71"/><text x="${p.cx+4}" y="${p.cy-4}" fill="#2fbf71" font-size="${D.width/70}">${p.position.slice(1)}</text>`);
  }
  svg.innerHTML = h; document.getElementById("save").disabled = pts.length !== 4; msg();
}
svg.onclick = e => { if (pts.length >= 4) return; const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
  const q = p.matrixTransform(svg.getScreenCTM().inverse()); pts.push([+q.x.toFixed(1), +q.y.toFixed(1)]); draw(); };
document.getElementById("undo").onclick = () => { pts.pop(); draw(); };
document.getElementById("save").onclick = async () => { const r = await (await fetch("/save", {method:"POST", body: JSON.stringify({clicks: pts, size: [D.width, D.height]})})).json();
  document.getElementById("msg").textContent = r.error || ("Saved " + r.path + ". You can close this window."); };
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", type=Path)
    ap.add_argument("--out", type=Path, help="default: calibration.json next to the photo")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    out = args.out or args.photo.with_name("calibration.json")
    with open(ROOT / "config" / "fixture_G.csv", newline="") as f:
        fixture = {r["position"]: (float(r["cx"]), float(r["cy"])) for r in csv.DictReader(f)}
    golden_size = (640, 480)
    data = args.photo.read_bytes()
    uri = f"data:{mimetypes.guess_type(args.photo.name)[0] or 'image/png'};base64,{base64.b64encode(data).decode()}"
    page = PAGE.replace("__DATA__", json.dumps({"image": uri, "corners": CORNERS}))

    def fit(clicks):
        coeffs = perspective([fixture[p] for p in CORNERS], clicks)
        return coeffs, [{"position": p, "cx": round(apply(coeffs, *xy)[0], 1), "cy": round(apply(coeffs, *xy)[1], 1)}
                        for p, xy in fixture.items()]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, body, ctype="application/json"):
            body = body.encode() if isinstance(body, str) else body
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(page, "text/html; charset=utf-8")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            clicks = body["clicks"]
            try:
                coeffs, points = fit(clicks)
            except ValueError as e:
                return self.reply(json.dumps({"error": str(e)}))
            if self.path == "/save":
                out.write_text(json.dumps({"photo": args.photo.name, "photo_size": body.get("size"),
                                           "golden_size": list(golden_size), "corners": dict(zip(CORNERS, clicks)),
                                           "perspective": coeffs, "positions_in_photo": points}, indent=1))
                print(f"saved {out}")
                return self.reply(json.dumps({"path": str(out)}))
            self.reply(json.dumps({"points": points}))

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Calibrating {args.photo} at {url}  (Ctrl+C to stop)")
    threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    main()
