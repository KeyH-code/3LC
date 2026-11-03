# D060 AP50 cycle study

| run | FP32 val AP50 | AP50–95 | best epoch | training seconds |
|---|---:|---:|---:|---:|
| old_SS04 | 0.845501972043 | 0.678893386691 | 28 | 2337.75 |
| R00 | 0.851600222014 | 0.677386640876 | 25 | 2114.85 |
| R01 | 0.864186476003 | 0.703183242411 | 59 | 1099.76 |

AP50 deltas (percentage points):
- R01_minus_R00: +1.258625398842
- R00_minus_old_SS04: +0.609824997179
- R01_minus_old_SS04: +1.868450396021

Target val AP50 > .90: False

Long-cycle net strategy effect, not isolated extra epochs: stretched LR decay, mosaic closes at epoch71 vs19, more checkpoint selection opportunities.
Single paired seed only; native selection and standalone FP32 evaluation differ. Old SS04 used AP95 selection.
Per-class AP and fixed-point P/R/F1/counts: per_class.csv; F1=2TP/(2TP+FP+FN), zero denominator maps to zero.
No hidden-test use, new submission, automatic follow-up, or competition-compliance claim.

Both runs were interrupted and resumed: R00 after epoch4 OOM, R01 after epoch63 service restart. Not equivalent to uninterrupted runs; native resume uses EMA-derived weights and does not restore all RNG/loader state. R00 artifacts were reused without retraining. CSV time is segment-local, not continuous cumulative time.
R00 workers including failed segment: 2843.924s. R01 pre-interruption CSV training time: 6803.010s (not full worker cost); new recovery worker: 1099.757s. No unobserved time is fabricated.
