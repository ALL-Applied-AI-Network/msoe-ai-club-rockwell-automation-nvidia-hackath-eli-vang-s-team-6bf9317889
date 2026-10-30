# Vial Inspection: twin-trained ChangeNet with an operator review loop

Rockwell Automation + NVIDIA Innovation Lab, Fall 2026 · Dataset 1, Final Vial Inspection

After the robot fills a fixture, every one of the 30 good-tray positions has to be checked for a missing cap, a
crooked cap, a damaged vial or an unexpected empty position. A missed defect costs more than a false alarm. So we
built a system that never passes a vial on its own when it is unsure: every position comes out **PASS**, **NO_PASS**
or **REVIEW**, and an operator's corrections go straight back into the next training round.

## How it works

1. **Visual ChangeNet (NVIDIA TAO 7.3, C-RADIOv2-B backbone)** compares each position's crop with the golden crop
   of the same position and gives a change score. It is trained on digital-twin data only, as the brief asks.
2. **Risk-aware decision**: score >= 0.2327 is NO_PASS, 0.10 to 0.2327 is REVIEW, below 0.10 is PASS. Both lines
   were set on validation data only, never on test.
3. **Disagreement detector**: a second opinion (our in-context vision model, plugged in as a CSV or one Python
   function) is compared with ChangeNet. Any disagreement goes to REVIEW.
4. **Operator review app**: shows every scene with colored boxes, the crop next to its golden reference, and records
   the operator's verdict (good / defect class) as missed defect, false alarm, review resolved, or confirmed.
5. **DEFT loop**: the flags turn into a root-cause report, a list of positions and classes to generate more twin data
   for, and a gate set the next model must pass. Round 1 of this loop already gave us the current model.
6. **Fixture state**: with a class from the second opinion, empty positions before a filled one are reported as
   possible missed placements (fill order: top right, down, then one column left).

## Results (twin test set: 50 fixtures, 1,500 positions; never used for tuning)

| Model | Missed defects | False alarms | Fixtures fully correct |
|---|---|---|---|
| Baseline (50 epochs, val threshold) | 4 of 919 (crooked caps) | 0 of 581 | 47 of 50 |
| **iter1, DEFT round 1 (deployed)** | **0 of 919** | **0 of 581** | **50 of 50** |

DEFT round 1: the baseline's weakest validation defects were crooked caps in the left columns (G19 to G30), so iter1
oversamples those. On a harsher stressed validation set iter1 flags 8.1% of good vials when catching every defect,
against 13.0% for the same recipe without the targeted data. Details: `models/visual_changenet/results/round1/summary.md`.

Real-camera results from the CSI testbed: _to be added after the lab test (trial log in
`models/visual_changenet/trials/trial_log.csv`)._

## Run it

Everything is in [`models/visual_changenet`](models/visual_changenet/README.md). The model file
(`checkpoints/iter1.pth`, 1.85 GB) is not in git: ask the team for it (GitHub Release `vcn-iter1-v1`).

```bash
# Review app demo, any laptop with Python 3, no GPU:
cd models/visual_changenet
python app/review_app.py examples/demo_run.zip

# Score fixture photos (Linux + NVIDIA GPU, driver 595+, docker, NGC key):
bash run_brev.sh lab-setup
bash run_brev.sh inspect <photo folder> [labels.csv]
```

CSI testbed plan (rules, checklist, collection sheet, test procedure): `models/visual_changenet/docs/csi_lab_kit.md`.

## Repository layout

```
models/visual_changenet/      the inspection system: training specs, scoring, review app, results, docs
csi_digital_twin_v1/          organisers' digital twin of the CSI line (Isaac Sim)
vial-labeler-participants-*/  organisers' manual labeling tool and public geometry references
full_test/                    extra twin test scenes (not in git)
```

## Data and credits

- Data: MSOE AI Club / Rockwell Automation / UWM Connected Systems Institute vial dataset and digital twin (event terms).
- Model: NVIDIA TAO Toolkit 7.3 Visual ChangeNet, NVIDIA C-RADIOv2-B backbone, recipe from the NVIDIA TAO Skill Bank.
- Compute: NVIDIA Brev (L40S).
- Code written with help from Claude (Anthropic).
- Team: _names and roles_.

---

## Hackathon info

- Event guide: <https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026>
- Team workspace: <https://dashboard.all-ai-network.org/j/L36HEH>
- **Final submission due Sun Nov 1, 11:59 PM CT** ([submit here](https://dashboard.all-ai-network.org/s/V28NHK)):
  a ~5 minute video, a one-page summary, a link to this repository, references and credits.
- Finals: Thu Nov 5, 5:00 to 8:00 PM, MSOE Diercks Hall.
- Don't commit large datasets or model weights; `.gitignore` keeps them out.
