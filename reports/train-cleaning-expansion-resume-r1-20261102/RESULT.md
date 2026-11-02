# SS04 three-band cumulative cleaning: complete results

D080 resumed from the interrupted epoch-4 last.pt to 28 total epochs; D070/D060 trained from scratch. D080 resume=true is an explicitly approved protocol deviation: RNG/DataLoader streams and FP16 serialized checkpoint state need not match uninterrupted training. Its initial-state hash describes original provenance, NOT a newly randomized or uninterrupted recovery run. Twelve uniform FP32 evaluations; prior A/B reused. Technical inspection/tool retries are preserved separately; no claim that only three training processes were attempted.
All labels/proposals frozen before these runs. val966 is a development/best-selection set, not an independent test.

| Model | best epoch | val AP50 | val AP50-95 | delta vs B (pp) | delta previous (pp) |
|---|---:|---:|---:|---:|---:|
| A | 28 | 0.809510 | 0.656022 | -1.371 | +0.000 |
| B | 27 | 0.828407 | 0.669736 | +0.000 | +1.371 |
| D080 | 26 | 0.824848 | 0.662334 | -0.740 | -0.740 |
| D070 | 26 | 0.854866 | 0.678405 | +0.867 | +1.607 |
| D060 | 28 | 0.845502 | 0.678893 | +0.916 | +0.049 |

## Hypothesis checks (descriptive, one seed)

- Evidence classification: `{"H1_strict_monotonic": "not_supported", "H1_all_new_above_B": "not_supported", "H2_observed_diminishing_or_decline": "supported_descriptively", "H3_fixed_reference": {"D0": "supported_descriptively", "D085": "supported_descriptively"}, "H4_gap_shrink_vs_B": {"D080": {"D0": "not_supported", "D085": "not_supported"}, "D070": {"D0": "supported_descriptively", "D085": "supported_descriptively"}, "D060": {"D0": "supported_descriptively", "D085": "supported_descriptively"}}, "cross_seed_stability_and_causal_mechanism": "insufficient_evidence"}`.
- H1 strict B -> D080 -> D070 -> D060 val improvement: False.
- All new models above B: False; adjacent decline exists: True.
- H3 fixed-reference train down-then-up observed: {'D0': True, 'D085': True} (self scores are NOT used for this check).
- H4 absolute train-val gap changes vs B: {'D080': {'D0': 0.00682910140376547, 'D085': 0.006324841692062488}, 'D070': {'D0': -0.01629907420044352, 'D085': -0.016426172728540966}, 'D060': {'D0': -0.010350202404861064, 'D085': -0.010439021157173656}}.
- No significance, cross-seed stability, causal mechanism or label-perfection claim. A decline does not identify a root cause.
- AP/PR at native best-F1 points and common conf>=.25 / IoU=.50 rematched metrics are distinct; see per_class.csv.
- H2 diminishing refers to raw adjacent AP increments, not a linear per-action effect; candidate composition also changes.
- Class gains/losses and common-threshold FP/FN deltas are reported without hiding regressions in class_deltas.csv.

## Fixed-reference train/val AP50-95

| Model | D0 train | D085 train | self train | val | D0 gap | D085 gap |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.812964 | 0.812333 | 0.812964 | 0.656022 | +0.156941 | +0.156311 |
| B | 0.799106 | 0.806349 | 0.806349 | 0.669736 | +0.129370 | +0.136613 |
| D080 | 0.798533 | 0.805272 | 0.804741 | 0.662334 | +0.136199 | +0.142938 |
| D070 | 0.791475 | 0.798591 | 0.798582 | 0.678405 | +0.113071 | +0.120186 |
| D060 | 0.797913 | 0.805067 | 0.802796 | 0.678893 | +0.119019 | +0.126174 |

## Cost / provenance

- Shared initial state SHA256: `b29d1a75cc221bee6f6f2b3722dbc7c65a266fad8cdea8295e938c975d78f8c0`.
- Train seconds: D080=2338.0, D070=1902.9, D060=2337.8
- Evaluation seconds (12 new): 1212.5013147999998
- Original sources and runtime copies verified before final success marker; no network/3LC; technical retry amendment preserved in retry_amendment.json.
- AI directly implemented/executed the specifically authorized workflow. Experiment choices were set by the learner.

## New weights

- D080: `E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\train-cleaning-expansion-resume-r1-20261102\D080\resumed\weights\best.pt` SHA256 `e256f52bcba0e9e33b169db8d1cc82bade2d06060f321a5e1ccd3418052efe21`
- D070: `E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\train-cleaning-expansion-resume-r1-20261102\D070\yolov8n\weights\best.pt` SHA256 `9b3dc2079e712df944de2b9d78acee9dcc2e97c227a97c1fc0ffdbc35ab9a835`
- D060: `E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\train-cleaning-expansion-resume-r1-20261102\D060\yolov8n\weights\best.pt` SHA256 `b36f72c507a13cc134092f3589d181c25b2014ae8f6845113b662ce654ed1eac`

## Recovery provenance and limitations

D080 resumed from the interrupted epoch-4 last.pt to 28 total epochs; D070/D060 trained from scratch. D080 resume=true is an explicitly approved protocol deviation: RNG/DataLoader streams and FP16 serialized checkpoint state need not match uninterrupted training. Its initial-state hash describes original provenance, NOT a newly randomized or uninterrupted recovery run. Twelve uniform FP32 evaluations; prior A/B reused. Technical inspection/tool retries are preserved separately; no claim that only three training processes were attempted.

D080 epoch 1-4 traces and historical checkpoint hashes are retained with the original interrupted attempt; epochs 5-28 are the resumed segment. Recovery elapsed times exclude prior interrupted training; see interrupted_attempt.json.
