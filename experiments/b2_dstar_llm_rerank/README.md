# Frozen B2+D* local-evidence LLM reranking

This isolated experiment never trains the backbone or edits historical files.
All generated caches, prompts, scores, ranks and reports belong to this directory.

## Frozen protocol

- Eleven historical Stage-7 targets; seeds 1024, 2026 and 3407 are averaged at the metric level, never score-ensembled.
- B2 checkpoint identity is verified against Stage-7 SHA256 manifests. D-Sender uses gamma=.75, alpha=.20, edge-mean anchoring. Adaptive variants are disabled.
- GraIL preserves the historical merged valid_ind+test_ind test set. Its validation comes from the independent training graph. Other datasets use disjoint target validation.
- Validation facts are grouped into connected components sharing either directional public query; selection and confirmation never split a sibling component. All settings are fixed a priori, and the optional high-margin gate is disabled.
- Only observation positives are excluded from Top-30. Private test truth is applied in offline evaluation only. Every test row remains in the denominator.
- All four LLM ablations use the same 5+3 pool and common no-evidence gate. This intentionally controls pool selection even for the names/signatures-only ablation; it is not a wholly retrieval-free system.
- At most three diverse paths per candidate are selected from a bounded 512-prefix beam. `search_pruned` explicitly records pruning; absence of retrieved paths is not proof of graph disconnection. PPR-guided ranking retains discovered cross-community bridge witnesses.
- Anonymous entities use directional Top-5 relation-frequency signatures. Numeric IDs are references into a readable dictionary, not names. Opaque relation types are explicitly identified as anonymous.
- Prompt degradation removes third, then second paths, then motif detail; it never deletes a candidate or strongest path. Unfit mandatory content produces an explicit identity fallback. cl100k_base is an estimator with 20% headroom, not the provider's exact tokenizer; actual API prompt_tokens is checked before accepting a response.
- Strict JSON, boolean, confidence, summary-length and candidate-bijection validation; abstentions, invalid outputs, timeouts and budget failures retain the original ranking exactly.
- Remote challengers enter the front only when ranked in the LLM top five. Others preserve tail order. Original full scores stay immutable. Filtered ranks preserve historical pessimistic ties on unchanged outputs.
- At-most-once API dispatch uses a persistent in-flight journal and file locks. An interrupted ambiguous request is retained as a fallback, not automatically retried or billed twice. This is not a claim of distributed exactly-once delivery.

## Run

From this directory in WSL:

```bash
bash scripts/run.sh scripts/fetch_lexical.py
bash scripts/run.sh scripts/audit.py
bash scripts/run.sh -m pytest tests/test_rerank_isolation.py -q -s --junitxml=results/tests.xml
bash scripts/run.sh scripts/record_tests.py
bash scripts/run.sh scripts/smoke.py
bash scripts/run.sh scripts/preflight_large.py
bash scripts/run.sh scripts/run_experiment.py
```

Windows: `scripts/launch_background.ps1` starts a hidden WSL worker. Double-click `monitor/LaunchProgress.cmd` for the native live progress window. The native viewer opens status with ReadWrite and Delete sharing so it cannot block the worker's atomic status replacement. The older PowerShell status readers are superseded; do not use them while the worker is running. GPU batches and query-mode work also have tqdm bars in `logs/full.log`. `status.json` records progress, elapsed time, ETA, token use, calls and a conservative USD cost estimate.

Credentials remain in the user's existing credential document or `LLM_API_KEY`; they are never copied into this repository. Defaults reuse the previous DeepSeek endpoint/model. Override with `LLM_API_BASE_URL` and `LLM_MODEL_NAME`; an override requires fresh smoke qualification.

## Audit / reuse map

| Asset | Reuse and changes |
|---|---|
| main universal_11g_v1/src/backbone.py | Imported unchanged; frozen model loading and 16-fact batches for ordinary graphs |
| Stage-7 engine, calibration, gated_model, topology | Imported read-only; hashes recorded; no training or mathematical operator replacement |
| main protocol_data / regime_inspector | Read-only observation loading, vocabulary alignment and historical split protocol |
| historical 11-graph M5 rankings | Hash-checked and matched against exact query order and three Stage-7 seed baselines |
| query-community community_detection.py / local_ppr.py | Vendored byte-for-byte from the pinned branch; provenance commit in manifest |
| query-community path_extractor / edge_weighter | Audited, not reused directly: new sender-only path weighting and explicit inverse direction required |
| query-community verbalizer | Replaced opaque-ID fallback with genuine public mapping or observed profiles; shared dictionaries avoid repetition |
| query-community llm_reranker | Reused transport/credential convention; replaced partial-candidate validation and retry behavior with bijection validation and durable at-most-once dispatch |

Full results are not available until `status.json.state` is `completed`. No claim of superiority to M5 is justified before paired statistical analysis.
