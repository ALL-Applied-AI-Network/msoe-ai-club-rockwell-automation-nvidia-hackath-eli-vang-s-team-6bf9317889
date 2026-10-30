# CSI lab kit: vial inspection (Dataset 1)

Frozen version for testing: **vcn-iter1-v1** (Visual ChangeNet iter1, `checkpoints/iter1.pth`, cutoff 0.2327,
`config/deploy.json`). Do not change code or the model during a test session; write down any change as a new version.

Step-by-step order of work on the day, with troubleshooting: [`runbook.md`](runbook.md).

## 1. Rules on site (from the organisers' "Collect and Test Your Own Data" page)

- Observe first. Touch equipment only when the plant manager explicitly approves.
- Never move, re-aim, disconnect or reconfigure an existing camera. Never change a PLC, robot controller,
  FactoryTalk configuration, safety system, recipe or program.
- No hardware on the industrial network without approval. Nothing where a robot could hit it.
- Own camera only after CSI reviews the setup. No photos of people or anything not needed for the project.
- Our system only reads images. It never sends anything to the line, so it cannot interfere with controls.

## 2. Before you go (checklist)

- [ ] Slack request posted in `#data-and-testbed-requests`, Shamar Webster tagged (draft in section 6), time confirmed
- [ ] Laptop: `models/visual_changenet` with this package, Python 3, review app opens on `examples/demo_run.zip`
- [ ] Model file `checkpoints/iter1.pth` (1,847,715,953 bytes) on the laptop and on a USB stick
- [ ] GPU plan: Brev instance (L40S, driver 595+) started and `bash run_brev.sh lab-setup` already passed,
      or the CSI workstation if CSI allows it. Scoring takes about a minute per batch; the review app needs no GPU.
- [ ] Failure handling tested on saved data (section 4, step 0)
- [ ] Empty folder for today's raw data: `data/csi/raw/<date>/` (never edit files in it)
- [ ] Printed or open copy of this sheet and the manual trial log (`docs/trial_log_manual.csv`)

## 3. Collection sheet

| | Plan |
|---|---|
| Target | Vial state at each of the 30 fixture positions: pass, missing_cap, crooked_cap, damaged_vial, no_vial |
| Question it answers | Does the twin-trained model catch every real defect without flagging too many good vials? |
| Variation | Defect type, defect position (especially left columns G19-G30, our weakest area), fill level (partly filled fixtures), room lighting |
| Constants | Overhead camera fixed (never touched), same fixture, same resolution |
| Ground truth | A teammate writes the true class per position at the time of capture, before looking at model output |
| Identifiers | File name `<date>_<run>_<trial>.png`; row per position in `labels.csv`: `image,position,class` |
| Splits | Every real image is test-only (brief: train on twin data). Keep at least one full run unseen by anyone tuning thresholds |
| Storage | `data/csi/raw/<date>/` raw, `data/csi/labels/<date>/labels.csv` labels, `data/csi/notes.md` who/when/conditions. Not in git |
| Pilot | First 3 images: open them, run calibration, run inspect, check boxes land on the vials, then continue |

## 4. Test procedure (start / stop)

0. **Dry run on saved data first** (proves failure handling): `bash run_brev.sh inspect ~/visual_changenet/runs/demo_run/scenes`
   (after `python3 -m zipfile -e examples/demo_run.zip runs/`), and once on the 11 `full_test` twin scenes with
   their labels (`... inspect ~/full_test/images ~/full_test/labels.csv`): it prints missed defects and false alarms
   should finish and log a row in `trials/trial_log.csv`. Try a folder with no images: it must stop with
   "no images found" and change nothing.
1. **Calibrate once per camera view.** Frames from the real 640x480 cap-seal camera match the twin view: no
   calibration. The organisers' 1920x1080 public view: use `config/calibration_csi_view_001.json`. Anything else
   (own camera, phone): copy one photo to the laptop, `python app\calibrate.py data\csi\raw\<date>\first.jpg`,
   click G01, G05, G26, G30, check the 30 dots, Save.
   **Take one golden photo per view**: a tray with all 30 good-tray positions confirmed good by a person. Pass it as
   `GOLDEN=...` so real photos are compared with a real reference.
2. **Score.** Upload the photo folder and `calibration.json` to the GPU box, then:
   ```bash
   CALIBRATION=~/calibration.json GOLDEN=~/good_tray.jpg CONDITION="CSI overhead cam, room lights" \
     bash run_brev.sh inspect ~/csi_photos
   ```
   With true labels: add the labels file at the end (`... inspect ~/csi_photos ~/labels.csv`) to get missed defects
   and false alarms printed.
3. **Review.** Download `~/run_<run>.zip`, then `python app\review_app.py <path to run_<run>.zip>` (it unpacks into `runs\`). The operator
   goes through every REVIEW and NO_PASS position and spot-checks PASS ones; every answer goes to `flags.csv`.
4. **Stop.** Ctrl+C in the review app. Download `trials/trial_log.csv`. **Stop or delete the Brev instance**
   (after copying `runs/` off it).

## 5. What we record (metrics)

Automatic, per run (`trials/trial_log.csv`): time, run, status (ok / failed), version, scenes, pass / review /
no_pass counts, disagreements, total and per-scene seconds, condition, calibrated, second opinion.
Per position (`runs/<run>/decisions.csv`): score, decision, reasons, second opinion.
Operator truth (`runs/<run>/flags.csv`): label, flag type (missed_defect, false_alarm, review_good, ...).
Manual, for physical attempts that are not a scoring run (`docs/trial_log_manual.csv`): every attempt, also failed ones.

Report afterwards: missed defects per class, false alarm rate on good vials, share sent to REVIEW, seconds per scene.
Then `python scripts\flags_to_deft.py --runs runs\<run> --out deft\round2` for the next round's targets.

## 6. Slack drafts (`#data-and-testbed-requests`, tag Shamar)

**Data request**
> Team <name>, Dataset 1 (vial inspection). Question: does our twin-trained ChangeNet catch real defects on the
> G fixture without too many false alarms? Please record, with the overhead camera fixed: 10 completed good
> fixtures, and 10 fixtures each with 2-3 intentional defects (crooked cap, missing cap, empty position), at least
> half of them in the left columns (G19-G30). Full-resolution stills, one per fixture. For each image a CSV row per
> defect position: image, position, class. Include the placement count or completion signal if available. We need
> it by <date>. Success = every image opens, labels match the photos, and our pipeline crops all 30 positions.

**Live test request**
> Team <name>, Dataset 1. We'd like <45 min> of supervised time on <date> to test our frozen system (vcn-iter1-v1)
> against the physical vial fixture. It only reads still images from the overhead camera (or our own reviewed
> camera); nothing connects to the line or controls. Plan: calibrate once, score fixtures, operator review, every
> attempt logged (version, input, prediction, expected, latency). Attending: <names>. Test plan attached.

## 7. Data handling

Raw files stay unchanged in `raw/`. Labels, cropped pairs, runs and generated data live in separate folders.
Note who collected each set and the sharing terms in `data/csi/notes.md`. No datasets, checkpoints, keys or
photos in git: commit only this package, a few sample images and the README explaining how teammates get the rest.
