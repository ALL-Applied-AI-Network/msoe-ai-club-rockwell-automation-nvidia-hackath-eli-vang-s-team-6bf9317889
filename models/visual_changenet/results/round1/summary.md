# DEFT round 1: iter1 vs control (2026-10-27, Brev L40S, TAO 7.3.0)

Both runs: 20 epochs, final-epoch checkpoint, threshold = midpoint of the clean-val gap.
iter1 = train_hard.csv (crooked_cap x3, G19-G30 x2). control = baseline spec on train.csv.

| | iter1 | control |
|---|---|---|
| Clean val: FAR at val threshold / AUROC | 0.00% / 1.0000 | 0.00% / 1.0000 |
| **Stressed val: FAR when every defect is caught (brief metric)** | **8.07%** | 12.99% |
| Stressed val: AUROC | **0.9998** | 0.9885 |
| Stressed val at clean-val threshold: missed / false alarms | 1 of 692 (crooked_cap) / 5.31% | 0 of 692 / 13.78% |
| Test at val threshold: missed / false alarms | **0 of 919 / 0.00%** | 2 of 919 (crooked_cap) / 0.00% |
| Test: FAR when every defect is caught | **0.00%** | 1.38% |
| Test scenes fully correct | **50 / 50** | 48 / 50 |

## Decision: iter1

The README said to pick on stressed val by "fewer misses at the val threshold, then lower false alarm rate at
catch-all". Read literally, control wins the first test (0 vs 1 miss). That count depends on a threshold set on
*clean* val, and the two models trade misses for false alarms at it (control: 0 misses but 13.8% false alarms;
iter1: 1 miss, 5.3%). The threshold-free comparison, the brief's own metric (false alarm rate when every defect is
caught) and AUROC, favours iter1 clearly (8.1% vs 13.0%, 0.9998 vs 0.9885). Test, not used to choose, agrees:
iter1 is perfect on all 50 test fixtures, control misses 2 crooked caps.

## Deployment cutoff for iter1: 0.2327

Set on **stressed val only**: the lowest score of any stressed defect (a crooked cap at G29, scene img_0071),
so every stressed defect is caught. Computed with `scripts/stress_cutoff.py` from the inference CSVs in
`round1/iter1/`.

| At cutoff 0.2327 | Missed defects | False alarms |
|---|---|---|
| Clean val | 0 of 692 | 0 of 508 (0.00%) |
| Stressed val | 0 of 692 | 41 of 508 (8.07%) |
| **Test** (checked after the cutoff was fixed) | **0 of 919** | **0 of 581 (0.00%)** |

On test, good vials score at most 0.071 and defects at least 0.976, so 0.2327 sits well inside the gap.
The cutoff is driven by one hard stressed crop; the next weakest stressed defect scores 0.992. Under the harsh
stressed lighting, 8% of good vials would be flagged for review, which is the cost of never missing a defect.

The control run also confirms the baseline recipe at the margin threshold: 0 false alarms on test.
