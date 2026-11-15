"""Asset and protocol audit; test labels are not used for selecting parameters."""
import json
from regime_inspector import ROOT, STAGE7, HISTORY, PREVIOUS, records, digest, save_json

def main():
    manifest = json.loads((STAGE7 / 'model_manifest.json').read_text())
    checks = {}
    for name, item in manifest.items():
        path = HISTORY / item['checkpoint']
        actual = digest(path)
        assert actual == item['checkpoint_sha256']
        checks[name] = dict(path=str(path), sha256=actual)
    official = STAGE7.parent.parent / 'motif-phase3-work/MOTIF/ckpts/motif_3g.pth'
    checks['B0'] = dict(path=str(official), sha256=digest(official))
    original = {}
    for folder in [PREVIOUS / 'src', PREVIOUS / 'scripts', STAGE7]:
        for path in folder.glob('*.py'):
            original[str(path)] = digest(path)
    cache = PREVIOUS / 'cache/conceptnet_text_emb.pt'
    original[str(cache)] = digest(cache)
    for name in ['frozen.json', 'inputs.json', 'model_manifest.json', 'results/summary.json']:
        path = STAGE7 / name
        original[str(path)] = digest(path)
    save_json(ROOT / 'results/historical_hashes.json', original)
    macro = json.loads((STAGE7 / 'results/summary.json').read_text())['macro']
    reference = [row for row in macro if row['method'] in ['B0', 'Dstar', 'B2+Dstar']]
    conflicts = [
        dict(id='P1_GRAIL', detail='Stage-7 GraIL test = valid_ind + test_ind. Calibrating on valid_ind leaks into the same test table.',
             recommended='Keep Stage-7 test unchanged; calibrate M6 on disjoint training-graph valid, label as cross-graph calibration.',
             alternative='Report M6 only on test_ind after valid_ind calibration, in a separate non-comparable table.'),
        dict(id='ADAPTIVE_ANCHOR', detail='Changing ConceptNet backbone to beta_tri=.15 and D*-Plus changes its zero-residual scores. It cannot simultaneously equal unmodified Stage-7 B2+D*.',
             recommended='Keep M2 and zero-residual anchor unchanged; report adaptive backbone separately and compare fallback against the unchanged anchor.'),
        dict(id='BASELINE_NAMES', detail='ConceptNet Stage-7 B0+D* MRR is 0.02427372636117352; 0.03544914826615365 is the mean of three B2+D* seed metrics, not B0+D* and not a score ensemble.',
             recommended='Preserve three-seed metric averaging for direct baseline alignment; separately label any ensemble.'),
    ]
    save_json(ROOT / 'results/asset_audit.json', dict(checkpoints=checks, baseline_reference=reference, protocol_conflicts=conflicts,
              target_graphs=len(records()), previous_embedding_cache=str(cache), previous_metrics_reuse='Do not merge metrics: previous ConceptNet models and clipping differ'))
    lines = ['# 11 图实验预检与协议待决事项', '',
             '本文件是预检记录，不是全量实验结果。尚未启动全量后台评测。', '',
             '已创建 experiment/universal-11graphs-boost；新增内容仅在 experiments/universal_11g_v1。历史文件只读。', '',
             '已完成 11 图观察图诊断、B0/B1/B2 检查点核验，以及零权重保真、纯结构隔离、微掩码和匿名文本防护的基础测试。', '',
             '## 需要明确的评估口径', '']
    for item in conflicts:
        lines.extend([f'### {item["id"]}', item['detail'], '', '建议：' + item['recommended'], ''])
    lines += ['## 本地模态与资产', '',
              'NLIngram_25/50 原始文件含可读 NELL 名称；不能预设为匿名图。WKIngram_25 含 Q/P 标识符，禁止文本编码。',
              'FB 系列为 Freebase MID、WN18RR 为 WordNet offset；所核查历史 data 目录未发现对应名称映射。',
              'ConceptNet 的既有句向量可按词表身份校验后复用，但旧结果不能直接充当新消融结果。', '',
              '## 当前实现范围', '',
              'src/regime_inspector.py：11 图只读诊断及原始观察边名称对齐。',
              'src/context_features.py：稀疏可达、拓扑关系画像、观察图经验平移和纯结构/双模共识。',
              'src/micro_probe_calibrator.py：零权重原分数旁路、5 折微掩码、坐标搜索与回退。',
              'src/backbone.py：阶段七冻结骨干与掩码后拓扑重算。',
              'scripts/run_smoke.py：两图 32 条最终查询、5 折探针与 B0/B2 三种子基准核对。', '',
              '未完成：体制切换与原始锚点的协议整合、完整 M6、11 图后台调度与汇总 Bootstrap 报告。',
              '冒烟是否通过以 results/smoke_status.json 为准；不以此文预先宣称通过。']
    (ROOT / 'preflight_review.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(verified_checkpoints=list(checks), protocol_conflicts=conflicts), ensure_ascii=False))

if __name__ == '__main__':
    main()
