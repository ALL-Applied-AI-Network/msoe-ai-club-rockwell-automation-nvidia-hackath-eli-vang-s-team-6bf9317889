#!/usr/bin/env bash
# Visual ChangeNet on a Brev GPU instance (or any Linux box with docker + nvidia-container-toolkit).
#
#   export NGC_KEY=...                            # never write keys into files in this repo
#   bash run_brev.sh round1 ~/vial_dataset.zip    # setup + train iter1 and control side by side + both reports
#
# Single steps: setup <zip> | train <run> | report <run>      (runs: baseline, control, iter1)
#   baseline  specs/baseline_train.yaml, 50 epochs on train.csv (the first run)
#   control   same spec, 20 epochs: the fair comparison for iter1
#   iter1     specs/iter1_train.yaml, 20 epochs on train_hard.csv (DEFT round 1)
#
# Lab inspection with the deployed model (checkpoints/iter1.pth, config/deploy.json):
#   bash run_brev.sh lab-setup                    # container, backbone, checkpoint check (no dataset needed)
#   bash run_brev.sh inspect <scene image|folder> [labels.csv]   # -> runs/<run>/decisions.csv + ~/run_<run>.zip
#   env: RUN=<name> (default lab_<time>), LABEL_COORDS=1 (crop at labels.csv cx,cy), SECOND_OPINION=csv:<file>,
#        CALIBRATION=<calibration.json from app/calibrate.py> (photos from another camera),
#        GOLDEN=<photo of an all-good tray from the same camera> (real golden reference instead of the twin one),
#        CONDITION="<lighting, camera, fixture...>" (goes into trials/trial_log.csv with every run)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pinned by tao-skill-bank 7.3.0 versions.yaml. Needs NVIDIA driver >= 595.45: Brev's L40S image
# shipped 565 (held by apt) and had to be upgraded first (README, "Brev gotchas").
TAO_IMAGE="${TAO_IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:7.3.0-pyt}"
SKILL_BANK_TAG="7.3.0"
SKILL_BANK="$ROOT/third_party/tao-skill-bank"
VCN_SKILL="$SKILL_BANK/skills/models/tao-train-visual-changenet"
BACKBONE="$ROOT/pretrained_models/C-RADIOv2_B.safetensors"

log() { printf '\n== %s\n' "$*"; }

# Per-run spec and CLI overrides.
spec_for() { case "$1" in iter1) echo /workspace/specs/iter1_train.yaml ;; *) echo /workspace/specs/baseline_train.yaml ;; esac; }
overrides_for() {
  echo "results_dir=/workspace/results/$1"
  [[ "$1" == control ]] && echo "train.num_epochs=20"
  return 0
}

# Launch pattern from the Skill Bank's tao-run-on-docker skill: run as the host user (no root-owned
# results) with HOME and framework caches redirected to a writable folder.
docker_tao() {
  local home=/workspace/results/.tao-runtime/home
  mkdir -p "$ROOT/results/.tao-runtime/home/.cache"/{huggingface,torch,triton,torchinductor,matplotlib}
  docker run --rm --gpus all --shm-size=8g --ipc=host \
    --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
    -e HOME=$home -e XDG_CACHE_HOME=$home/.cache -e HF_HOME=$home/.cache/huggingface \
    -e TORCH_HOME=$home/.cache/torch -e TRITON_CACHE_DIR=$home/.cache/triton \
    -e TORCHINDUCTOR_CACHE_DIR=$home/.cache/torchinductor -e MPLCONFIGDIR=$home/.cache/matplotlib \
    -v "$ROOT":/workspace \
    -v "$BACKBONE":/data/pretrained_models/C-RADIOv2_B.safetensors:ro \
    "$TAO_IMAGE" "$@"
}
tao() { docker_tao visual_changenet "$@"; }

# Host python when it has Pillow (Brev's venv python has no pip), otherwise the TAO container's python.
py() {
  if python3 -c "import PIL, numpy" 2>/dev/null; then
    python3 "$@"
  else
    local args=() a
    for a in "$@"; do args+=("${a//$ROOT//workspace}"); done
    docker_tao python3 "${args[@]}"
  fi
}

# GPU driver, backbone and TAO container: everything training and inference need except data.
runtime_setup() {
  command -v docker >/dev/null || { echo "docker not found"; exit 1; }
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  local drv; drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
  (( drv >= 595 )) || { echo "NVIDIA driver $drv is too old for TAO 7.3 (needs 595+). See README, Brev gotchas."; exit 1; }
  [[ -n "${NGC_KEY:-}" ]] || { echo "export NGC_KEY first (NGC personal key, used for the nvcr.io pull)"; exit 1; }

  log "C-RADIOv2-B backbone"
  if [[ ! -s "$BACKBONE" ]]; then
    mkdir -p "$(dirname "$BACKBONE")"
    # plain download of the public HF file; some Brev images ship a venv python without pip
    curl -fL --retry 3 -o "$BACKBONE" https://huggingface.co/nvidia/C-RADIOv2-B/resolve/main/model.safetensors
  fi
  ls -lh "$BACKBONE"

  log "TAO container (first pull takes 10-20 min)"
  printf '%s' "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null
  docker pull -q "$TAO_IMAGE"
}

setup() {
  local zip="${1:?usage: run_brev.sh setup /path/to/vial_dataset.zip}"
  runtime_setup

  log "TAO Skill Bank $SKILL_BANK_TAG"
  [[ -d "$SKILL_BANK" ]] || git -c advice.detachedHead=false clone -q --depth 1 --branch "$SKILL_BANK_TAG" \
    https://github.com/NVIDIA-TAO/tao-skill-bank "$SKILL_BANK"

  log "Dataset"
  rm -rf "$ROOT/data/raw" && mkdir -p "$ROOT/data/raw"
  python3 -m zipfile -e "$zip" "$ROOT/data/raw"
  local src; src="$(cd "$(dirname "$(find "$ROOT/data/raw" -name labels.csv -path '*golden*' | head -1)")/.." && pwd)"
  py "$ROOT/scripts/prepare_vcn_dataset.py" --src "$src" --out "$ROOT/data/vcn"
  for split in train train_hard val val_stress test; do
    mode=train; [[ $split == test ]] && mode=evaluate
    python3 "$VCN_SKILL/scripts/validate_vcn_dataset.py" --csv "$ROOT/data/vcn/$split.csv" \
      --images-dir "$ROOT/data/vcn/images" --mode "$mode" --image-ext .png --batch-size 16 --num-gpus 1
  done
}

train() {
  local run="${1:?usage: run_brev.sh train <baseline|control|iter1>}"
  mkdir -p "$ROOT/results/$run/logs"
  log "Training $run (log: results/$run/logs/train.log)"
  # shellcheck disable=SC2046
  tao train -e "$(spec_for "$run")" $(overrides_for "$run") > "$ROOT/results/$run/logs/train.log" 2>&1
  tail -3 "$ROOT/results/$run/logs/train.log"
}

# Final epoch by default: on the baseline, val hit 0% FAR after epoch 0, so the val-selected
# "best" checkpoint was the epoch-0 one and scored worse on the brief metric. CKPT=<host path> overrides.
checkpoint_for() {
  local d="$ROOT/results/$1/train" ckpt="${CKPT:-}"
  [[ -n "$ckpt" ]] || ckpt="$d/changenet_model_classify_latest.pth"
  [[ -e "$ckpt" ]] || { echo "no checkpoint in $d" >&2; exit 1; }
  echo "${ckpt/#$ROOT//workspace}"
}

report() {
  local run="${1:?usage: run_brev.sh report <baseline|control|iter1>}"
  local res="$ROOT/results/$run" ckpt
  ckpt="$(checkpoint_for "$run")"
  mkdir -p "$res/logs"
  log "Scoring $run with $ckpt"
  for split in val val_stress test; do
    # shellcheck disable=SC2046
    tao inference -e "$(spec_for "$run")" $(overrides_for "$run") \
      inference.checkpoint="$ckpt" \
      inference.results_dir="/workspace/results/$run/inference_$split" \
      dataset.classify.infer_dataset.csv_path="/workspace/data/vcn/$split.csv" \
      > "$res/logs/inference_$split.log" 2>&1 || { tail -20 "$res/logs/inference_$split.log"; exit 1; }
  done
  py "$ROOT/scripts/score_report.py" --meta "$ROOT/data/vcn/meta.csv" --title "Visual ChangeNet $run" \
    --val "$(find "$res/inference_val" -name inference.csv | head -1)" \
    --stress "$(find "$res/inference_val_stress" -name inference.csv | head -1)" \
    --test "$(find "$res/inference_test" -name inference.csv | head -1)" \
    --out "$res/report"
  echo "$ckpt" > "$res/report/checkpoint.txt"
  cp "$res/report/report.md" "$HOME/report_$run.md"     # top level, easy to download from Jupyter
}

DEPLOY_CKPT="$ROOT/checkpoints/iter1.pth"

lab_setup() {
  runtime_setup
  log "Checkpoint"
  [[ -s "$DEPLOY_CKPT" ]] || { echo "Put the iter1 checkpoint at $DEPLOY_CKPT (1,847,715,953 bytes)"; exit 1; }
  ls -l "$DEPLOY_CKPT"
  log "Ready. Next: bash run_brev.sh inspect <scene image or folder>"
}

# Scenes -> crops -> TAO inference with iter1 -> PASS / NO_PASS / REVIEW per position (scripts/scene_inspect.py).
inspect_scenes() {
  local src="${1:?usage: run_brev.sh inspect <scene image or folder> [labels.csv]}" labels="${2:-}"
  local run="${RUN:-lab_$(date +%Y%m%d_%H%M%S)}"
  local dir="$ROOT/runs/$run" ws="/workspace/runs/$run" extra=()
  [[ -s "$DEPLOY_CKPT" ]] || { echo "missing $DEPLOY_CKPT; run lab-setup first"; exit 1; }
  [[ -s "$BACKBONE" ]] || { echo "missing backbone; run: bash run_brev.sh lab-setup"; exit 1; }
  rm -rf "$dir" && mkdir -p "$dir/input" "$dir/logs"
  # copy scenes into the run so the container (which only sees this folder) can read them
  find "$src" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) \
    -exec cp {} "$dir/input/" \;
  [[ -n "$(ls -A "$dir/input")" ]] || { echo "no images found in $src"; exit 1; }
  if [[ -n "$labels" ]]; then
    cp "$labels" "$dir/input_labels.csv"
    extra+=(--labels "$dir/input_labels.csv")
    [[ -n "${LABEL_COORDS:-}" ]] && extra+=(--use-label-coords)
  fi
  if [[ -n "${CALIBRATION:-}" ]]; then
    cp "$CALIBRATION" "$dir/calibration.json"
    extra+=(--calibration "$dir/calibration.json")
  fi
  if [[ -n "${GOLDEN:-}" ]]; then
    cp "$GOLDEN" "$dir/golden_photo.${GOLDEN##*.}"
    extra+=(--golden-photo "$dir/golden_photo.${GOLDEN##*.}")
  fi
  local t0; t0=$(date +%s)
  log "Cropping $(ls "$dir/input" | wc -l) scenes into runs/$run"
  py "$ROOT/scripts/scene_inspect.py" crop --run "$dir" --scenes "$dir/input" ${extra[@]+"${extra[@]}"}

  log "Scoring with iter1 (log: runs/$run/logs/inference.log)"
  local ds=() s
  for s in train_dataset validation_dataset test_dataset infer_dataset; do   # every split -> this run's pairs
    ds+=("dataset.classify.$s.csv_path=$ws/infer.csv" "dataset.classify.$s.images_dir=$ws/pairs")
  done
  tao inference -e /workspace/specs/iter1_train.yaml results_dir="$ws/tao" \
    inference.checkpoint=/workspace/checkpoints/iter1.pth inference.results_dir="$ws/tao/inference" "${ds[@]}" \
    > "$dir/logs/inference.log" 2>&1 || { tail -20 "$dir/logs/inference.log"; trial_log "$run" "$t0" failed; exit 1; }

  log "Decisions"
  local so=()
  [[ -n "${SECOND_OPINION:-}" ]] && so=(--second-opinion "$SECOND_OPINION")
  py "$ROOT/scripts/scene_inspect.py" decide --run "$dir" ${so[@]+"${so[@]}"}
  trial_log "$run" "$t0" ok
  (cd "$ROOT/runs" && python3 -m zipfile -c "$HOME/run_$run.zip" "$run")
  log "Done. Download ~/run_$run.zip, then on your laptop: python app\\review_app.py <path to run_$run.zip>"
}

# One row per inspect attempt, failed ones too (CSI rule: log every attempt with version, input, result, latency).
trial_log() {
  local run="$1" t0="$2" status="$3" log="$ROOT/trials/trial_log.csv" dec="$ROOT/runs/$1/decisions.json"
  mkdir -p "$ROOT/trials"
  [[ -s "$log" ]] || echo "time,run,status,version,scenes,pass,review,no_pass,disagreements,seconds_total,seconds_per_scene,condition,calibrated,second_opinion" > "$log"
  python3 - "$log" "$run" "$status" "$(( $(date +%s) - t0 ))" "$dec" "${CONDITION:-}" "${CALIBRATION:+yes}${GOLDEN:+ +real golden}" "${SECOND_OPINION:+yes}" \
    "$(ls "$ROOT/runs/$1/input" 2>/dev/null | wc -l)" "$ROOT/config/deploy.json" <<'PY'
import csv, json, sys, time
log, run, status, secs, dec, cond, cal, so, n, cfg = sys.argv[1:]
d = json.load(open(dec)) if status == "ok" else {"summary": {"decisions": {}, "disagreements": ""}}
c, n, secs = d["summary"]["decisions"], int(n), int(secs)
csv.writer(open(log, "a", newline="")).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), run, status, json.load(open(cfg))["version"],
    n, c.get("pass", ""), c.get("review", ""), c.get("no_pass", ""), d["summary"]["disagreements"], secs,
    round(secs / n, 1) if n else "", cond, cal or "no", so or "no"])
PY
  echo "trial logged in trials/trial_log.csv"
}

round1() {
  setup "${1:-}"
  log "Training iter1 and control side by side on one GPU (about 1.5-2 h)"
  local p1 p2
  train iter1 & p1=$!
  train control & p2=$!
  wait "$p1" && wait "$p2"
  report iter1
  report control
  log "Done. Download ~/report_iter1.md and ~/report_control.md, then stop the instance."
}

case "${1:-}" in
  setup) setup "${2:-}" ;;
  train) train "${2:-}" ;;
  report) report "${2:-}" ;;
  round1) round1 "${2:-}" ;;
  lab-setup) lab_setup ;;
  inspect) inspect_scenes "${2:-}" "${3:-}" ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
