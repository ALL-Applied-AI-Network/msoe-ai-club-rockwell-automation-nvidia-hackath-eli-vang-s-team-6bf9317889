# Visual ChangeNet baseline (final epoch, recommended)

Run: TAO 7.3.0-pyt, C-RADIOv2-B, 50 epochs on Brev L40S (2026-10-26). Checkpoint: changenet_model_classify_latest.pth (epoch 50).
Comparison with the epoch-0 checkpoint is in report_epoch000.md.

Threshold chosen on **val** (catches all 692 val defects): `0.999822`
Val false alarm rate at that threshold: **0.00%**, AUROC 1.0000

## Test (never used for tuning)

- Missed defects at the val threshold: **4 / 919** (recall 99.56%)
- False alarm rate at the val threshold: **0.00%** (0 of 581 good vials)
- Brief metric, false alarm rate at the threshold that catches every test defect: **0.00%**
- AUROC: 1.0000

| Class | Test samples | Missed at val threshold | Miss rate |
|---|---|---|---|
| crooked_cap | 212 | 4 | 1.89% |
| missing_cap | 195 | 0 | 0.00% |
| no_vial | 512 | 0 | 0.00% |

Scenes: 50 fixture images, 3 with at least one missed defect, 47 with every position correct.

## Reading it

- Good and defective test vials are perfectly separated (AUROC 1.0, 0% false alarms when every defect is caught).
- The 4 misses come from the threshold rule, not the model: scores are squashed near 1.0, and "highest threshold
  that still catches every val defect" (0.999822) sits right at the edge of the val defects, so a few test crooked
  caps fall just under it. Next iteration: pick the threshold with a safety margin inside the val gap, and make the
  val set harder so it can tell checkpoints apart. (We now know test is separable, so any rule change must be
  justified on val alone and confirmed on fresh twin data.)
- Checkpoint selection by val_far did not work here: val hit 0% after epoch 0, so every later epoch tied.

## Margin threshold (val only)

Val score gap for the final-epoch model (`val_inference_final_epoch.csv`): good vials score at most **0.0609**
(median 0.055); defects score at least **0.99982**. The edge rule put the threshold at the very top of that gap.
The margin rule (midpoint, now the default in `score_report.py`) gives **0.530**.

The 4 test crooked caps missed at the edge threshold score 0.842, 0.9958, 0.9961 and 0.99982
(`test_misses_edge_threshold.csv`), so all four are above 0.530 and would be caught. The test false alarm count at
0.530 still needs a rerun of `bash run_brev.sh report` with the updated script (0 is expected, since every test
good vial scored below 0.842).
