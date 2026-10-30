"""Pluggable second opinion for the disagreement detector (e.g. the team's in-context vision model).

A second opinion answers, per fixture position: is this vial good or a defect, and optionally which class.
Two ways to plug one in, chosen with --second-opinion on `scene_inspect.py decide`:

  csv:<path>              a CSV with columns scene,position,label[,confidence,reason]
                          (any tool can write it: a notebook, a VLM script, another model)
  py:<file.py>:<function> a Python function called once per position with a dict:
                          {scene, position, score, crop, golden, scene_image}  (paths are absolute)
                          returning {"label": ..., "confidence": ..., "reason": ...} or None (no opinion)

label is "pass" or a defect class (missing_cap, crooked_cap, no_vial, damaged_vial) or "no_pass".
See examples/second_opinion_example.py for the function shape.
"""
import csv
import importlib.util
from pathlib import Path

GOOD = {"pass", "good", "ok"}


def is_defect(label):
    return str(label).strip().lower() not in GOOD


def load(spec):
    """Return f(item) -> {label, confidence, reason} | None."""
    kind, _, rest = spec.partition(":")
    if kind == "csv":
        with open(rest, newline="") as f:
            table = {(r["scene"], r["position"]): r for r in csv.DictReader(f)}

        def from_csv(item):
            r = table.get((item["scene"], item["position"]))
            if not r or not r.get("label"):
                return None
            return {"label": r["label"].strip(), "confidence": r.get("confidence", ""), "reason": r.get("reason", "")}
        return from_csv
    if kind == "py":
        path, _, func = rest.rpartition(":")
        mod_spec = importlib.util.spec_from_file_location("second_opinion_plugin", Path(path).resolve())
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        return getattr(mod, func)
    raise SystemExit(f"--second-opinion must be csv:<file> or py:<file.py>:<function>, got {spec!r}")
