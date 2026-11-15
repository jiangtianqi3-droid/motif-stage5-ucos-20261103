import json
from regime_inspector import ROOT, digest, save_json

hashes = json.loads((ROOT / 'results/historical_hashes.json').read_text())
changed = [p for p, expected in hashes.items() if digest(p) != expected]
audit = json.loads((ROOT / 'results/asset_audit.json').read_text())
changed += [v['path'] for v in audit['checkpoints'].values() if digest(v['path']) != v['sha256']]
assert not changed, changed
smoke = json.loads((ROOT / 'results/smoke_status.json').read_text())
assert smoke['stage'] == 'smoke_passed'
save_json(ROOT / 'results/preflight_verification.json', dict(historical_files_unchanged=True,
          verified_files=len(hashes) + len(audit['checkpoints']), smoke=smoke))
save_json(ROOT / 'results/status.json', dict(stage='awaiting_protocol_choice', full_run_started=False,
          smoke_passed=True, pending=['P1_GRAIL', 'ADAPTIVE_ANCHOR'],
          implementation_status='preflight and smoke complete; full experiment integration pending protocol choice'))
report = ROOT / 'preflight_review.md'
with report.open('a', encoding='utf-8') as f:
    f.write('\n\n## 冒烟已通过\n\nNELLInductive_v1、WKIngram_25 每图 32 条最终查询；B0+D* 与 B2 三种子排名逐条匹配阶段七。5 折各掩蔽 41/64 条正向边，冒烟每折评估 8 条双向伪查询。\n\n')
    f.write(f'峰值 RSS={smoke["peak_rss_bytes"] / 1024**3:.3f} GiB，GPU reserved={smoke["peak_gpu_reserved_bytes"] / 1024**3:.3f} GiB。历史受保护资产哈希未变化。\n')
print('Preflight verified; full evaluation has not started.')
