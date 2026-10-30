# Visual ChangeNet baseline

Run: TAO 7.3.0-pyt, C-RADIOv2-B, 50 epochs on Brev L40S (2026-10-26). Checkpoint scored: model_best_000.pth
(kept by lowest val_far; val was already perfectly separated after epoch 0, so later epochs could only tie).

Threshold chosen on **val** (catches all 692 val defects): `0.446707`
Val false alarm rate at that threshold: **0.00%**, AUROC 1.0000

## Test (never used for tuning)

- Missed defects at the val threshold: **3 / 919** (recall 99.67%)
- False alarm rate at the val threshold: **0.00%** (0 of 581 good vials)
- Brief metric, false alarm rate at the threshold that catches every test defect: **14.46%**
- AUROC: 0.9998

| Class | Test samples | Missed at val threshold | Miss rate |
|---|---|---|---|
| crooked_cap | 212 | 3 | 1.42% |
| missing_cap | 195 | 0 | 0.00% |
| no_vial | 512 | 0 | 0.00% |

Scenes: 50 fixture images, 2 with at least one missed defect, 48 with every position correct.
