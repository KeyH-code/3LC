# P3 label application readiness

- User approval: `LABEL-APPLICATION-APPROVAL-20261101.json`, original message `可以应用到标签`; three exact sealed proposal hashes only. Training remains unauthorized.
- New mechanical tool: `tools/cleaning_expansion_apply.py`; old sealed application/training sources remain unchanged.
- Focused tests: 10 passed. Full synthetic regression: 351 passed in 30.334s before application.
- Read-only code review session: `ses_e698cd3f0ffe3tns7xXUFMvCoP`. One concrete preflight blocker was reproduced: unchanged val labels can contain original edge/outside-image boxes. Corrected the new tool to hash/copy val bytes without geometry repair. Failing regression added, then passed. Follow-up read-only review found no blockers in the correction.
- Original A/B train order checked against 7852 image ID/name/hash tuples. D085 manifest row order differed at same-frame duplicates; lexical image filename order exactly matches both A/B arms. Output manifests use that order, with no membership or image-byte change.
- Fixed val966 source: all image/label hashes verified, 11487 boxes. No clipping, filtering or rewriting of val labels.
- D085 in-memory expected totals: D08073002, D07074080, D06075634 boxes; changed labels298/1128/2070. These were preflight values, not a claim of completed application.
- Storage: independent full train/val copies, no hardlinks, same existing copy strategy. Three copies1453986051 source bytes; available77999693824 bytes at initial measurement. No deletion of history or scratch directories.
- Application command was started once with an exclusive new destination `datasets-20261101`; completion and final validation are recorded separately. Partial output is preserved on failure, never silently overwritten.

External resource used: previously user-specified `E:\anaconda\envs\dl-3lc\python.exe`. No other project-external local files were searched/read.
