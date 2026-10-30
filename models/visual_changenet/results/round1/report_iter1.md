# Visual ChangeNet iter1

Threshold chosen on **val** (midpoint of the val gap between good vials and defects): `0.527309`
Val false alarm rate at that threshold: **0.00%**, AUROC 1.0000

## Stressed val (val scenes, harsher lighting, blur and shift; used to compare models)

- At the val threshold: missed **1 / 692**, false alarms **5.31%**
- False alarm rate when every stressed defect is caught: **8.07%**, AUROC 0.9998
- Missed per class: crooked_cap 1/154, missing_cap 0/179, no_vial 0/359

## Test (never used for tuning)

- Missed defects at the val threshold: **0 / 919** (recall 100.00%)
- False alarm rate at the val threshold: **0.00%** (0 of 581 good vials)
- Brief metric, false alarm rate at the threshold that catches every test defect: **0.00%**
- AUROC: 1.0000

| Class | Test samples | Missed at val threshold | Miss rate |
|---|---|---|---|
| crooked_cap | 212 | 0 | 0.00% |
| missing_cap | 195 | 0 | 0.00% |
| no_vial | 512 | 0 | 0.00% |

Scenes: 50 fixture images, 0 with at least one missed defect, 50 with every position correct.
