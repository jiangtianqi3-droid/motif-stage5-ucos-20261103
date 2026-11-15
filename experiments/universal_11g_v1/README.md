# Universal 11-graph evaluation

This experiment evaluates pure structural and structural plus lexical residuals against frozen MOTIF Stage-7 asymmetry baselines. Both variants use five folds of masked observation facts to select weights. The B2 chain averages metrics across seeds 1024, 2026, and 3407.

## Run the evaluation

The launcher uses WSL, a CUDA GPU, and the historical asset layout under `/mnt/d`. Prepare these assets before running.

| Asset | Expected location or source |
| --- | --- |
| Shared Python, CUDA headers, GCC, and PyTorch runtime | `/mnt/d/ultra-phase1-env` |
| Stage-7 asymmetric implementation, runtime, manifests, and baseline ranks | `/mnt/d/motif-phase4-work/motif-stage7-asymmetric-20261106` |
| Frozen observation and evaluation arrays | `/mnt/d/motif-phase4-work/motif-stage6-entity-degree-p0p1-v2/data` |
| Observation vocabularies, raw splits, and B1/B2 checkpoints | `/mnt/d/motif-phase4-work/topology-semantic-stage4-20261102` |
| Official B0 checkpoint | `/mnt/d/motif-phase3-work/MOTIF/ckpts/motif_3g.pth` |
| Prior ConceptNet adapter, text dependencies, and verified embeddings | [Sibling ConceptNet experiment](../conceptnet_zeroshot_v1/README.md), including its local `cache/deps` and `cache/conceptnet_text_emb.pt` |
| Frozen text encoder | Local BGE-small-en-v1.5 snapshot referenced by `semantic_calibrator.py` in the sibling experiment |

From this directory, audit the assets and run the two-graph smoke test before the full evaluation.

```bash
bash scripts/run.sh scripts/audit_assets.py
bash scripts/run.sh scripts/check_invariants.py
bash scripts/run.sh scripts/run_universal_11g.py --smoke
bash scripts/run.sh scripts/run_universal_11g.py
```

On Windows, `scripts/launch_background.ps1` starts the full run in WSL and returns its process ID. Double-click `monitor/LaunchProgress.vbs` to open the progress window. The worker writes `results/status.json` and `logs/full.log`. Completion generates `report_universal_11g.md` and a run directory with the summary, provenance, per-graph ranks, and checksums. Git ignores these generated files.

Run identities hash the implementation and frozen asset manifest. Rerunning the same command reuses valid completed graphs and available score shards for that identity. `--resume-from <fingerprint>` requires a compatible full run with local provenance and frozen calibration files. The diagnostic in `monitor/check_replay.py` requires the original ConceptNet recovery cache.

## Evaluation protocol

- M0 is official B0, M1 is B0+D*, and M2 is B2+D*. M3 uses pure structural self-supervision, M4 uses fixed dual-mode weights, M5 uses self-supervised dual-mode weights, and M6 uses validation calibration.
- GraIL test sets retain the Stage-7 merged `valid_ind` and `test_ind` convention. M6 uses `valid` from the independent training graph for these datasets. Ingram and ConceptNet use their disjoint validation splits. The original anchors stay fixed, and A0/A3/A5 report the adaptive backbone ablations separately.
- The five-fold fallback depends on observation reconstruction. All eleven graphs have prior test exposure. Opaque identifiers bypass the lexical encoder, and readable local NLIngram names activate it.
- The evaluation streams 256-query blocks. ConceptNet uses smaller feature shards. Final baseline scoring replays the original 16-fact batches and releases earlier activations to respect the GPU budget. The final anchor ranks are compared with the frozen ranks query by query.
- Intervals use 2,000 paired bootstrap draws, binding both directions and the seed results by fact. Group macro averages weight each graph equally.
