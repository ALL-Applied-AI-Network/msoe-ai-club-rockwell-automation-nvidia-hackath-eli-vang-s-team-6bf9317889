# CSI lab day runbook: vial inspection (vcn-iter1-v1)

Step-by-step for the supervised test session at the CSI testbed. Site rules, the collection plan and the Slack
drafts are in [`csi_lab_kit.md`](csi_lab_kit.md); this page is the order of work on the day and what to do when a
step fails. Keep it open on the laptop.

**Frozen version: `vcn-iter1-v1`** (`checkpoints/iter1.pth`, NO_PASS >= 0.2327, REVIEW 0.10 to 0.2327,
`config/deploy.json`). Do not change code, thresholds or the model on site. If something has to change, stop,
write it in `docs/trial_log_manual.csv` and call it a new version.

Our system only reads still images. It never connects to the line. Touch nothing at CSI without the plant
manager's explicit OK.

## Roles

| Role | Does | Must not |
|---|---|---|
| **Driver** | GPU box: upload, `inspect`, download run zips | Change specs, config or code |
| **Labeler** | Writes the true class of all 30 positions per photo into `labels.csv` at capture time | Look at model output before the labels are written |
| **Operator** | Review app: resolves every REVIEW and NO_PASS, spot-checks PASS | Edit `flags.csv` by hand |
| **Logger / liaison** | Talks to CSI staff, keeps `docs/trial_log_manual.csv` and `data/csi/notes.md`, watches the clock | Touch equipment |

With fewer people, one person can do two roles, but the Labeler can never also be the Driver or the Operator.

## Timeline

| When | What | Section |
|---|---|---|
| Day before | Pre-flight on saved data, pack | [A](#a-day-before-pre-flight) |
| T-30 min | Start Brev, `lab-setup`, open laptop tools | [B](#b-t-30-min-bring-up) |
| 0-10 min | Calibrate, golden photo, pilot of 3 photos | [C](#c-on-site-0-10-min-calibrate-and-pilot) |
| 10-35 min | Capture, label, score, review in batches | [D](#d-on-site-10-35-min-capture-score-review) |
| 35-45 min | Wrap up, copy everything off, stop Brev | [E](#e-on-site-35-45-min-wrap-up) |
| After | Metrics and the next DEFT round | [F](#f-after-the-visit) |

Laptop commands are PowerShell, run from `models\visual_changenet`. GPU box commands are bash, run from
`~/visual_changenet`.

---

## A. Day before: pre-flight

Do all of this at home. If any step fails, fix it before the visit, not on site.

1. **Laptop review app.**
   ```powershell
   python app\review_app.py examples\demo_run.zip
   ```
   Expect: `Review app for runs\demo_run at http://localhost:8000`, and a browser tab with 10 scenes and colored
   boxes. Press Ctrl+C to stop.
2. **Model file.** Check `checkpoints\iter1.pth` is exactly **1,847,715,953 bytes**, on the laptop and on a USB stick:
   ```powershell
   (Get-Item checkpoints\iter1.pth).Length
   ```
3. **GPU box dry run** (Brev L40S, VM Mode, Jupyter on). Upload the package zip and `iter1.pth`, then:
   ```bash
   nvidia-smi | head -4                       # driver must be 595 or newer; if not, see README "Brev gotchas"
   export NGC_KEY='nvapi-...'                 # type it here only, never save it in a file
   tmux new -s vcn                            # back in after a disconnect: tmux a -t vcn
   bash run_brev.sh lab-setup                 # expect "Ready. Next: bash run_brev.sh inspect ..."
   python3 -m zipfile -e examples/demo_run.zip runs/
   RUN=dry_demo bash run_brev.sh inspect runs/demo_run/scenes
   RUN=dry_test bash run_brev.sh inspect ~/full_test/images ~/full_test/labels.csv
   ```
   Expect: each run ends with `Done. Download ~/run_<run>.zip` and adds a row to `trials/trial_log.csv`.
   `dry_test` prints a `vs_labels` block with missed defects and false alarms (twin scenes: expect 0 missed).
4. **Failure handling.** `mkdir ~/empty && bash run_brev.sh inspect ~/empty` must stop with `no images found`.
5. **Round trip.** Download `~/run_dry_test.zip` through Jupyter, open it on the laptop with
   `python app\review_app.py <path to run_dry_test.zip>`.
6. **Stop the Brev instance** (stopping keeps the disk). Note the instance name.
7. **Pack:** laptop and charger, USB stick with `iter1.pth`, printed [`csi_lab_kit.md`](csi_lab_kit.md) and this page,
   a blank `labels.csv` template (header `image,position,class`), empty folders `data\csi\raw\<date>\` and
   `data\csi\labels\<date>\`.

**Go / no-go:** go only if steps 1 to 5 passed. Otherwise, still go, but only capture data (section D, steps 1 and 2)
and score it afterwards.

## B. T-30 min: bring up

1. Start the Brev instance, open a terminal, `tmux a -t vcn` (or `tmux new -s vcn`).
2. `export NGC_KEY='nvapi-...'` and `bash run_brev.sh lab-setup`. Expect `Ready.`
3. On the laptop, open this page, `docs\trial_log_manual.csv` and a blank `labels.csv`.
4. Logger: write today's date, attendees, CSI contact and room lighting into `data\csi\notes.md`.

## C. On site, 0-10 min: calibrate and pilot

1. **Find out which camera view you are using.** The answer decides the calibration:

   | Source | Calibration |
   |---|---|
   | Real 640x480 cap-seal camera (same view as the twin) | none |
   | Same 1920x1080 view as the organisers' public photos | `config/calibration_csi_view_001.json` |
   | Anything else (our own camera after CSI review, a phone) | make one (step 2) |

2. **Calibrate (only if needed).** Copy one photo into `data\csi\raw\<date>\`, then:
   ```powershell
   python app\calibrate.py data\csi\raw\<date>\first.jpg
   ```
   On http://localhost:8001, click the centres of **G01, G05, G26, G30** in that order. Check that all 30 dots sit on
   vial centres, then press Save. That writes `calibration.json` next to the photo. If the camera moves later,
   calibrate again.
3. **Golden photo.** Capture one tray with all 30 good-tray positions filled, which a person has checked as good.
   Save it as `data\csi\raw\<date>\good_tray.jpg`. With a non-640x480 camera, the golden photo needs the calibration
   from step 2.
4. **Pilot of 3 photos.** Capture 3 fixtures into their own batch folder `data\csi\raw\<date>\b0\`, label them,
   zip the folder and upload the zip (single zip files only, never folders):
   ```powershell
   Compress-Archive data\csi\raw\<date>\b0\* csi_b0.zip
   ```
   Upload `csi_b0.zip`, `calibration.json`, `good_tray.jpg` and `labels.csv` to the box. Then:
   ```bash
   python3 -m zipfile -e ~/csi_b0.zip ~/csi_b0
   RUN=pilot CALIBRATION=~/calibration.json GOLDEN=~/good_tray.jpg CONDITION="CSI <camera>, room lights" \
     bash run_brev.sh inspect ~/csi_b0 ~/labels.csv
   ```
   Leave out `CALIBRATION=` for the cap-seal camera. Download `~/run_pilot.zip` and open it in the review app.
   **Check that the boxes land on the vials in every scene.** If they don't, fix that first (troubleshooting
   below) and don't continue capturing.

## D. On site, 10-35 min: capture, score, review

Work in batches of about 5 fixtures, so a problem shows up after 5 photos and not 20.

1. **Capture** into a new folder per batch: `data\csi\raw\<date>\b1\`, `b2\`, and so on. Name files
   `<date>_<run>_<trial>.png`. Vary the defect type and position (at least half of the
   defects in G19 to G30) and the fill level, as listed in the collection sheet in the lab kit. Never edit files in
   `raw\`.
2. **Label.** For each photo, the Labeler writes **one row per position, all 30**, before anyone looks at model
   output:
   ```
   image,position,class
   20261031_r1_01.png,G01,pass
   20261031_r1_01.png,G02,crooked_cap
   ```
   Classes: `pass`, `missing_cap`, `crooked_cap`, `no_vial`, `damaged_vial`. The `image` column is matched by
   file name without the extension. A position missing from the file is left out of the miss and false-alarm
   counts, so don't list only the defects.
3. **Score.** Zip and upload the batch and the updated `labels.csv` as in C4 (`csi_b1.zip`, unpacked to `~/csi_b1`),
   then:
   ```bash
   RUN=b1 CALIBRATION=~/calibration.json GOLDEN=~/good_tray.jpg CONDITION="<what changed: lights, fill level>" \
     bash run_brev.sh inspect ~/csi_b1 ~/labels.csv
   ```
   About a minute per batch. Read the `vs_labels` block it prints. **Any `missed (decision pass)` above 0 is the
   headline finding of the day:** tell the team and write it down.
4. **Review.** Operator: download `~/run_b1.zip` and run `python app\review_app.py <path to run_b1.zip>`. Enter
   your name in the operator box. Go through the queue with the arrow keys, compare each crop with its golden crop,
   and press **G** for a good vial or **1** missing cap, **2** crooked cap, **3** no vial, **4** damaged vial.
   Resolve every REVIEW and NO_PASS, plus about 5 PASS per scene. Answers are saved to `runs\b1\flags.csv` as you go.
5. **Log** every physical attempt that was not a scoring run (bad photo, re-shoot, fixture reset) in
   `docs\trial_log_manual.csv`. Scoring runs log themselves to `trials/trial_log.csv`, failed ones included.

**Stop and ask CSI staff** if anything on the line looks different from what was agreed, or if anyone asks you to
touch equipment. **Stop scoring** (keep capturing and labeling) if the box keeps failing: score later from
`data\csi\raw\`.

## E. On site, 35-45 min: wrap up

1. Ctrl+C in the review app.
2. On the box, zip everything the session produced and download it:
   ```bash
   cd ~/visual_changenet && python3 -m zipfile -c ~/csi_session_<date>.zip runs trials
   ```
3. Check the laptop has: `data\csi\raw\<date>\` (all photos), `data\csi\labels\<date>\labels.csv`, every
   `run_*.zip`, `csi_session_<date>.zip`, `calibration.json`, `good_tray.jpg`, `notes.md`.
4. **Stop the Brev instance.** Delete it only after step 3 is checked; deleting loses the disk.
5. Thank CSI staff, leave the station as you found it, and post a short thank-you in `#data-and-testbed-requests`.

Nothing from this session goes into git: no photos, labels, runs or keys.

## F. After the visit

```powershell
python scripts\flags_to_deft.py --runs runs\b1 runs\b2 --out deft\round2
```
Report: missed defects per class, false alarm rate on good vials, share sent to REVIEW, seconds per scene (from
`trials\trial_log.csv`), then fill in the "Real-camera results" line in the top-level README. Real crops stay
test-only unless the organisers OK `--allow-real-train`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NVIDIA driver NNN is too old for TAO 7.3` | Brev image ships 565/580 | Driver fix in README "Brev gotchas" (about 10 min with reboot). Do this the day before |
| `export NGC_KEY first` | New shell or tmux window | `export NGC_KEY='nvapi-...'` again |
| `missing .../checkpoints/iter1.pth` | Checkpoint not uploaded | Upload from the USB stick into `checkpoints/`, check 1,847,715,953 bytes |
| `no images found in ...` | Wrong folder, or the zip unpacked into a subfolder | `ls` the folder; point `inspect` at the folder that holds the images |
| `X.jpg is (W, H), the fixture map is for (640, 480): calibrate this camera` | Not the cap-seal view and no calibration | Section C2, then pass `CALIBRATION=` |
| `--golden-photo needs --calibration` | `GOLDEN=` with a non-640x480 camera | Add `CALIBRATION=` from the same camera |
| `WARNING ... is (W, H), calibration photo was ...` | Photo resolution changed since calibration | Set the camera back to the calibration resolution, or calibrate again |
| `WARNING ...: fixture looks shifted by (dx, dy) px` | Fixture or camera moved | Check the boxes in the review app; if they are off, re-seat the fixture (ask CSI) or calibrate again |
| Boxes miss the vials in the review app | Bad calibration clicks or the camera moved | Calibrate again, clicking the cap centres carefully in the order G01, G05, G26, G30 |
| Almost everything is REVIEW or NO_PASS on good trays | Twin golden used on real photos, or a bad golden photo | Pass `GOLDEN=` with a person-checked all-good tray from the same camera and lighting |
| `inspect` fails after "Scoring with iter1" | TAO error | `tail -40 runs/<run>/logs/inference.log`; the attempt is logged as failed. Retry once, then switch to capture-only |
| Review app: `has no decisions.csv` | Opened a failed or partial run | Open a run that ended with `Done.` |
| Port 8000 / 8001 already in use | Another app instance still running | Ctrl+C the old one, or add `--port 8010` |
| Browser disconnects from Brev | Jupyter session dropped | Reopen the terminal, `tmux a -t vcn`; the run carries on inside tmux |
| `vs_labels` counts look too small | `labels.csv` doesn't cover all 30 positions, or the file names don't match | One row per position; `image` must match the photo file name |
