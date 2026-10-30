#!/usr/bin/env python3
"""Operator review app: look at every fixture scene, check the model's calls, and flag the wrong ones.

    python app/review_app.py runs/<run>            then open http://localhost:8000
    python app/review_app.py run_<run>.zip         a zip from the GPU box: unpacked into runs/ first

Needs only Python 3 (no packages), so it runs on a Windows laptop with a run folder copied from the GPU box.
Every verdict the operator gives is appended to runs/<run>/flags.csv (latest verdict per position wins).
scripts/flags_to_deft.py turns that file into the next DEFT round's inputs.

Flag types (from the model's decision vs the operator's label):
  missed_defect   model said PASS, operator says defect        (the costly one)
  false_alarm     model said NO_PASS, operator says good
  review_good / review_defect   a REVIEW position the operator resolved
  wrong_class     operator corrected the defect class from the second opinion
  confirmed       operator agrees with the model
"""
import argparse
import csv
import json
import threading
import webbrowser
import zipfile
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
FLAG_FIELDS = ("time", "operator", "scene", "position", "object_name", "score", "model_call", "decision",
               "second_label", "disagreement", "operator_label", "flag_type", "note", "crop", "golden")
CLASSES = ("pass", "missing_cap", "crooked_cap", "no_vial", "damaged_vial")
lock = threading.Lock()


def flag_type(row, label):
    defect = label != "pass"
    if row["decision"] == "review":
        return "review_defect" if defect else "review_good"
    if row["decision"] == "pass":
        return "missed_defect" if defect else "confirmed"
    if not defect:
        return "false_alarm"
    second = row.get("second_label", "")
    return "wrong_class" if second and second != "no_pass" and second != label else "confirmed"


def read_flags(run):
    path = run / "flags.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {(r["scene"], r["position"]): r for r in csv.DictReader(f)}  # latest wins


def make_handler(run):
    with open(run / "decisions.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    index = {(r["scene"], r["position"]): r for r in rows}
    meta = json.loads((run / "decisions.json").read_text())

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send_json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path, ctype):
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path).path
            if url in ("/", "/index.html"):
                return self.send_file(HERE / "review.html", "text/html; charset=utf-8")
            if url == "/api/run":
                return self.send_json({"run": run.name, **meta, "rows": rows, "classes": CLASSES,
                                       "flags": list(read_flags(run).values())})
            if url.startswith("/files/"):
                path = (run / unquote(url[len("/files/"):])).resolve()
                if run.resolve() not in path.parents or not path.is_file():
                    return self.send_error(404)
                ctype = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                         ".bmp": "image/bmp"}.get(path.suffix.lower(), "application/octet-stream")
                return self.send_file(path, ctype)
            self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/flag":
                return self.send_error(404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            row = index.get((body.get("scene"), body.get("position")))
            label = body.get("label")
            if not row or label not in CLASSES:
                return self.send_json({"error": "unknown position or label"}, 400)
            flag = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "operator": str(body.get("operator", ""))[:40], "operator_label": label,
                    "flag_type": flag_type(row, label), "note": str(body.get("note", ""))[:500],
                    **{k: row.get(k, "") for k in ("scene", "position", "object_name", "score", "model_call",
                                                   "decision", "second_label", "disagreement", "crop", "golden")}}
            path = run / "flags.csv"
            with lock:
                new = not path.exists()
                with open(path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FLAG_FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(flag)
            self.send_json(flag)

    return Handler


def unpack(path):
    """run_<run>.zip (one top folder) -> runs/<run>; an existing folder is reused so flags are kept."""
    with zipfile.ZipFile(path) as z:
        top = {n.replace("\\", "/").split("/")[0] for n in z.namelist()}
        if len(top) != 1:
            raise SystemExit(f"{path} should contain one run folder, found {sorted(top)[:5]}")
        dest = RUNS / top.pop()
        if not dest.exists():
            RUNS.mkdir(exist_ok=True)
            z.extractall(RUNS)
            print(f"unpacked {path.name} into {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run folder with decisions.csv (from scene_inspect.py decide)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 lets other laptops on the network open it")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if args.run.suffix.lower() == ".zip":
        args.run = unpack(args.run)
    if not (args.run / "decisions.csv").exists():
        raise SystemExit(f"{args.run} has no decisions.csv; run scene_inspect.py decide first")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.run))
    url = f"http://localhost:{args.port}"
    print(f"Review app for {args.run} at {url}  (Ctrl+C to stop). Flags go to {args.run / 'flags.csv'}")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
