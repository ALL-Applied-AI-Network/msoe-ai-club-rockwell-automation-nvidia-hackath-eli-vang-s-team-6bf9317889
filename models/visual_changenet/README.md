# Visual ChangeNet baseline (vial fixture)

TAO Visual ChangeNet **classify** compares each fixture-position crop with the golden crop of the
same position and outputs a distance score. High score = different from golden = NO_PASS.
Built from the [NVIDIA TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) Visual ChangeNet
skill, pinned to release **7.3.0** (container `nvcr.io/nvidia/tao/tao-toolkit:7.3.0-pyt`, backbone C-RADIOv2-B).

## Folder layout

Put this folder in the project as `models/visual_changenet/`.

```
visual_changenet/
  README.md
  run_brev.sh                    round1 | setup | train <run> | report <run> | lab-setup | inspect <scenes>
  specs/baseline_train.yaml      TAO spec (Skill Bank template, vial changes marked "# vial:")
  specs/iter1_train.yaml         DEFT round 1 spec (changes marked "# iter1:")
  scripts/prepare_vcn_dataset.py vial_dataset.zip -> TAO pairs + CSVs, scene-level val split, train_hard, val_stress
  scripts/score_report.py        scores -> per-class misses, false alarm rate, report.md
  scripts/stress_cutoff.py       deployment cutoff = catch every stressed-val defect, applied to val/stress/test
  scripts/scene_inspect.py       full scenes -> crops -> PASS / NO_PASS / REVIEW per position (+ disagreement detector)
  scripts/second_opinion.py      plug-in point for the in-context model (CSV or Python function)
  scripts/flags_to_deft.py       operator flags -> RCA, twin targets and KPI gate set for the next DEFT round
  app/review_app.py, review.html operator review app (plain Python, no packages): view calls, flag misses/false alarms
  app/calibrate.py               click 4 corner vials once per camera view -> calibration.json (plain Python)
  docs/csi_lab_kit.md            CSI visit: rules, checklist, collection sheet, test procedure, Slack drafts
  docs/runbook.md                CSI lab day runbook: roles, timeline, step-by-step commands, troubleshooting
  docs/trial_log_manual.csv      manual log for physical attempts (scoring runs log themselves to trials/)
  trials/trial_log.csv           one row per inspect run, failed ones too (created on first run)
  config/deploy.json             deployed model, cutoff 0.2327, review band, fixture map
  config/fixture_G.csv, golden/  golden position centres, 30 golden crops and the golden scene
  examples/demo_run.zip          10 test scenes already scored by iter1: try the review app with no GPU
  config/calibration_csi_view_*.json   ready calibrations for the organisers' real 1920x1080 tray photos (from the
                                 vial-labeler's calibration.json, good-tray grid; view_001 and view_122)
  examples/second_opinion_example.py   template for the in-context model hook
  checkpoints/iter1.pth          the deployed model (not in git, 1.85 GB)
  runs/<run>/                    one inspection run: scenes, crops, decisions.csv, flags.csv (not in git)
  .gitignore                     keeps data/, results/, weights and third_party/ out of git
  # created on the GPU box, not committed:
  third_party/tao-skill-bank/    Skill Bank 7.3.0 (validator, backbone stager, DEFT AOI)
  pretrained_models/             C-RADIOv2_B.safetensors
  data/raw/  data/vcn/           extracted zip, converted dataset
  results/<run>/                 train/ (checkpoints, tensorboard), inference_*/, report/, logs/ (baseline, control, iter1)
```

## Data decisions

| Decision | Why |
|---|---|
| Golden = `golden/crops/pass/img_0000_GNN.png` for the same position | ChangeNet needs a reference per pair; the golden fixture has all 30 positions |
| `pass` -> `PASS`, `missing_cap`, `crooked_cap`, `no_vial` -> NO_PASS (names kept) | TAO collapses non-PASS to one class; keeping names lets the report give per-class misses |
| `no_vial` counts as NO_PASS | An empty target position is something the operator must see; the fixture-state engine decides later whether it is "waiting" or a missed placement |
| Val = 40 of the 200 train scenes, whole scenes (seed 42) | Neighbouring vials appear in crops, so splitting by scene stops leakage |
| Test (50 scenes) untouched until the final report | Brief rule: never tune on test |
| Scored checkpoint = final epoch (best `val_loss` also kept) | On the baseline val hit 0% false alarms after epoch 0, so val_far could not pick between epochs |
| Threshold = middle of the val gap between good vials and defects (`--rule edge` for the old catch-all edge) | The edge sat at 0.9998 and missed 4 test crooked caps; the val gap was 0.061 to 0.9998 |

## Run it on Brev

1. Create a Brev instance (L40S or bigger, **VM Mode**, Jupyter on). It bills while running, so **stop it when done**.
2. Click **Open Notebook**, open a terminal and check `nvidia-smi | head -4`. If the driver is below 595, run the
   driver fix under "Brev gotchas" first (about 10 minutes including the reboot).
3. Upload `visual_changenet.zip` and `vial_dataset.zip` with Jupyter's Upload button (single zip files, never folders).
4. In the terminal:
   ```bash
   python3 -m zipfile -e visual_changenet.zip . && cd visual_changenet
   export NGC_KEY='nvapi-...'                 # type it in the shell only, never save it in this folder
   tmux new -s vcn                            # keeps it running if the browser closes (back in: tmux a -t vcn)
   bash run_brev.sh round1 ~/vial_dataset.zip
   ```
   `round1` = setup (Skill Bank, backbone, container pull, data prep + validation), then trains **iter1** and
   **control** side by side on the one GPU, then scores both on val, stressed val and test. Logs are in
   `results/<run>/logs/`. When it prints "Done", download `~/report_iter1.md` and `~/report_control.md`.
5. **Stop the instance.** Stopping keeps the disk; deleting it loses the trained models.

Single steps: `bash run_brev.sh setup <zip>`, `bash run_brev.sh train <baseline|control|iter1>`,
`bash run_brev.sh report <run>`.

## What the report gives you

`results/baseline/report/report.md` and `metrics.json`:

- Threshold picked on val, and val false alarm rate at it.
- On test, at that threshold: missed defects per class (crooked_cap, missing_cap, no_vial), overall
  recall, false alarm rate on good vials, AUROC, and how many fixture scenes had any miss.
- The brief's metric: false alarm rate at the threshold that catches every test defect (reported, not used for tuning).
- `test_misses.csv` and `test_false_alarms.csv`, the input for the DEFT AOI iteration.

## DEFT round 1 (iter1)

Baseline result (see `results/baseline/report.md`): test AUROC 1.0 and 0% false alarms when every defect is caught,
but val was already perfect after one epoch, so val could not rank models, and the weakest val defects were all
crooked caps, mostly in the left columns (G19 to G30) where the fixture edge darkens the background.

| Change | Where | Why (val evidence only) |
|---|---|---|
| Repeat crooked caps x3 and positions G19-G30 x2 in training (`train_hard.csv`, 8,416 rows) | `prepare_vcn_dataset.py` (`HARD_*`), `specs/iter1_train.yaml` | Lowest val defect scores were crooked caps in those columns |
| Stressed val set: the 1,200 val crops with harsher lighting (0.55-1.45x), contrast, colour, blur and +-3 px shift | `val_stress.csv` | Clean val is saturated; the stressed copy can tell models apart and shows robustness beyond the twin's randomization range |
| Control run: baseline spec for the same 20 epochs | `run_brev.sh` (`control`) | Fair comparison, so any gain is from the data change |
| Threshold = middle of the val gap; final-epoch checkpoint | `score_report.py`, `run_brev.sh` | The edge threshold missed 4 test crooked caps; val_far tied every epoch |

Pick the winner on **stressed val** (fewer misses at the val threshold, then lower false alarm rate at catch-all).
Test is reported for both but is not used to choose. This is the "RCA, targeted data, retrain, gate" part of the
Skill Bank's `tao-run-deft-aoi` loop done by hand; the full workflow (synthetic anomalies with Cosmos, k-NN mining,
state tracking) lives in `third_party/tao-skill-bank/skills/applications/tao-run-deft-aoi/` and needs a KPI set
the model actually fails on, which the stressed val now provides.

### Round 1 result: iter1 is the model, cutoff 0.2327

| | iter1 | control |
|---|---|---|
| Stressed val: FAR when every defect is caught | **8.07%** | 12.99% |
| Stressed val AUROC | **0.9998** | 0.9885 |
| Test missed defects / false alarms (val threshold) | **0 / 0%** | 2 crooked caps / 0% |

Deployment cutoff for iter1 = **0.2327**, the lowest stressed-val defect score (set without test). At it, test has
0 of 919 defects missed and 0 of 581 good vials flagged; stressed val flags 8% of good vials for review.
Details and the decision note: `results/round1/summary.md`. Checkpoint: `checkpoints/iter1.pth`
(= `model_epoch_019_step_10520.pth`, 1,847,715,953 bytes; the Brev instance is deleted).

## Lab inspection (deployed model)

Model: iter1 checkpoint at `checkpoints/iter1.pth`. Decision per fixture position (`config/deploy.json`, set on val only):

| ChangeNet score | Decision |
|---|---|
| >= 0.2327 | **NO_PASS** (the round-1 cutoff: catches every stressed-val defect) |
| 0.10 to 0.2327 | **REVIEW**: a buffer for real-camera shift, a person decides |
| < 0.10 | **PASS** (clean val good vials score at most 0.055) |
| second opinion disagrees with the PASS/NO_PASS call | **REVIEW** (a disagreement never auto-passes) |

Each scene is aligned to the golden fixture by phase correlation (max 1 px error on the 50 test scenes), cut
into 30 crops of 40x40 and paired with the golden crop of the same position. If lab images are a different size
or framing (a phone, another camera), calibrate once per camera view on the laptop:
`python app/calibrate.py one_photo.jpg`, click G01, G05, G26 and G30, Save, then pass `CALIBRATION=calibration.json`
to `inspect`. Each photo is warped onto the golden view, so scale and rotation match training (on a simulated
tilted 1600x1200 photo the crops matched the originals closely; real-camera scores still need checking at the lab).
Every `inspect` run, failed or not, adds a row to `trials/trial_log.csv` (version, counts, seconds per scene,
`CONDITION`). The CSI visit plan is in `docs/csi_lab_kit.md`.

**Real photos.** The organisers' public tray photos (vial-labeler package) are 1920x1080 from another view;
`config/calibration_csi_view_001.json` (and `_view_122`) are built from the labeler's own good-tray grid, so no
clicking is needed for that view. The twin's cap-seal camera matches the real 640x480 cap-seal camera, so frames from
that camera need no calibration. Real photos look very different from twin renders (lighting, tall clear tubes in
perspective), so compare them with a **real** golden: `GOLDEN=<photo of a tray with all 30 good-tray positions
confirmed good>` uses that photo's crops as the reference (inference only, nothing is trained on it):
```bash
CALIBRATION=config/calibration_csi_view_001.json GOLDEN=~/good_tray.jpg CONDITION="public photos" \
  bash run_brev.sh inspect ~/real_photos
```
Only the good tray (G01-G30) is scored; the scrap tray has a different plate and no golden yet. How well the
twin-trained model transfers to real photos is exactly what the CSI test measures: expect more REVIEW calls at first.

On the GPU box (driver 595+, docker, `NGC_KEY` exported, `iter1.pth` in `checkpoints/`):
```bash
bash run_brev.sh lab-setup                         # container + backbone + checkpoint check
bash run_brev.sh inspect ~/lab_scenes              # folder or single image; -> ~/run_lab_<time>.zip
RUN=day1 bash run_brev.sh inspect ~/lab_scenes     # name the run
```
Scoring a labelled set (e.g. the test split) prints missed defects and false alarms at the deployed rule:
`bash run_brev.sh inspect <images folder> <labels.csv>`. Replaying the round-1 iter1 test scores through the
decision rule gives 0 of 919 defects missed and 0 of 581 good vials flagged on all 50 test scenes. The `inspect`
TAO call itself was tested with a stand-in for the container only; run it on the test images once on the GPU box
to confirm (the aligned crops match the dataset crops to within 1 px).

Review on any laptop (plain Python 3, no packages): unzip the run into `runs/`, then
```bash
python app/review_app.py runs/<run>                # opens http://localhost:8000
python app/review_app.py examples/demo_run.zip     # demo, no GPU needed (unpacks into runs/demo_run)
python app/review_app.py ~/Downloads/run_<run>.zip # a run zip from the GPU box, unpacked into runs/
```
Click any position (or use the queue: arrow keys), compare the crop with the golden crop, and press **G** for a
good vial or **1-4** for the defect class. Every answer goes to `runs/<run>/flags.csv` and is classified as
`missed_defect`, `false_alarm`, `review_good` / `review_defect`, `wrong_class` or `confirmed`.

### Disagreement detector (in-context model hook)

Re-run only the decision step when the in-context model's answers exist (no GPU needed):
```bash
python scripts/scene_inspect.py decide --run runs/<run> --second-opinion csv:answers.csv
python scripts/scene_inspect.py decide --run runs/<run> --second-opinion py:examples/second_opinion_example.py:judge
```
`answers.csv` columns: `scene,position,label[,confidence,reason]` with label `pass` or a defect class. Any position
where the two models disagree goes to REVIEW and is shown dashed purple in the app. With a class from the second
opinion the app also reports fixture state (empty / occupied_good / occupied_bad) and missed placements (an empty
position before a filled one in fill order).

### Flags -> next DEFT round
```bash
python scripts/flags_to_deft.py --runs runs/day1 --out deft/round2
```
Writes `rca.md` (misses and false alarms by class and position, which model won each disagreement),
`targets.csv` (position x class to render more of in the twin or oversample) and `kpi_set.csv` (every checked
position as a TAO pair with the operator's label: the gate the next model must pass). The brief says train on
twin data, so real crops only go into training with `--allow-real-train` after the organisers OK it.

## Brev gotchas (from the first run, 2026-10-26)

- **Upload single zips through Jupyter**, never folders (folder uploads silently drop files). Check the dataset with
  `md5sum vial_dataset.zip` (`20e72c3d278d6aa64c4c774fdab3aa12` for the original).
- **NVIDIA driver must be 595.45 or newer** for TAO 7.2/7.3 (`nvidia-smi`). Brev's L40S image shipped 565 with every
  NVIDIA package held. Fix used on that instance:
  ```bash
  distro=ubuntu$(. /etc/os-release; echo ${VERSION_ID//./})
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
  sudo apt-mark unhold $(apt-mark showhold | grep -iE "nvidia|cuda|libnvidia")
  # only if cuda-drivers reports Conflicts: purge the old branch (565 on the Isaac image, 580 on a Shadeform L40S)
  sudo apt-get purge -y '^nvidia-.*565.*' '^libnvidia-.*565.*' 'cuda-drivers-565' 'cuda-drivers-fabricmanager-565'
  sudo apt-get autoremove -y && sudo apt-get install -y cuda-drivers && sudo reboot
  ```
- Training took about 2.5 min/epoch on an L40S (50 epochs, about 2 hours).
- **Copy results off before deleting an instance.** Stopping keeps the disk; deleting loses the trained model.

## Brief rule to keep in mind

The brief says to **train on digital-twin data** and use real data only for looking/testing. The
per-image metadata in this dataset (`light_stops`, `tint`, `shift_x/y`, `roll_deg`) looks like
twin domain randomization; if the team confirms these are twin renders, this baseline is
compliant. If they are real captures, treat this as a reference baseline and retrain on twin output.
