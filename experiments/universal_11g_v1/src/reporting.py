"""Three-seed metric means; fact-paired, graph-stratified macro bootstrap."""
import json
import numpy as np
from regime_inspector import ROOT, save_json

METRICS = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']

def values(ranks):
    # seeds x (facts * 2) -> facts x metrics, preserving both directions together.
    r = np.asarray(ranks).reshape(len(ranks), -1, 2)
    return np.stack([1 / r, r <= 1, r <= 3, r <= 10], axis=-1).mean(axis=(0, 2))

def build_report(run, recs, smoke=False):
    all_values, tables, distributions = {}, {}, {}
    rng = np.random.default_rng(20261114)
    for rec in recs:
        key = rec['key']
        folder = run / 'per_graph' / key
        summary = {}
        with np.load(folder / 'rankings.npz') as data:
            items = {name: values(data[name]) for name in data.files if name != 'queries'}
        all_values[key] = items
        distributions[key] = {}
        for name, val in items.items():
            family = name.split('/')[0]
            anchor = items[family + ('/M2' if family == 'B2' else '/M1')]
            dist, delta = [], []
            # Shared random indices per comparison ensure pairing; both directions and seeds are bound.
            local = np.random.default_rng(20261114)
            for _ in range(2000):
                ix = local.integers(len(val), size=len(val))
                dist.append(val[ix].mean(0))
                delta.append((val - anchor)[ix].mean(0))
            dist, delta = np.asarray(dist), np.asarray(delta)
            distributions[key][name] = (dist, delta)
            summary[name] = dict(metrics=dict(zip(METRICS, val.mean(0).tolist())),
                                 ci95=np.quantile(dist, [.025, .975], axis=0).T.tolist(),
                                 paired_delta= (val-anchor).mean(0).tolist(),
                                 paired_delta_ci95=np.quantile(delta, [.025, .975], axis=0).T.tolist())
        save_json(folder / 'metrics.json', summary)
        tables[key] = summary
    groups = {}
    for group in ['locked', 'historical', 'external', 'All-11']:
        members = [r['key'] for r in recs if group == 'All-11' or r['group'] == group]
        if not members:
            continue
        groups[group] = {}
        for name in tables[members[0]]:
            point = np.mean([all_values[k][name].mean(0) for k in members], axis=0)
            # Independently permuting each graph's draws avoids cross-graph index correlation.
            samples, deltas = [], []
            for k in members:
                a, b = distributions[k][name]
                order = rng.permutation(2000)
                samples.append(a[order]); deltas.append(b[order])
            groups[group][name] = dict(metrics=dict(zip(METRICS, point.tolist())),
                ci95=np.quantile(np.mean(samples, axis=0), [.025, .975], axis=0).T.tolist(),
                paired_delta_ci95=np.quantile(np.mean(deltas, axis=0), [.025, .975], axis=0).T.tolist(), graphs=len(members))
    save_json(run / 'summary.json', dict(per_graph=tables, groups=groups, bootstrap_replicates=2000,
              seed_aggregation='Mean metrics over 1024,2026,3407; not score ensemble', bootstrap_unit='fact, both directions bound; graphs fixed and equally weighted'))
    lines = ['# 11 图零样本结构与双模评测', '',
        '冒烟集成报告，不能用于效果结论。' if smoke else '全量评测完成。所有正负结果保留，不把实验目标表述为既成结论。', '',
        'M0=官方 B0；M1=B0+D*；M2=阶段七 B2+D* 三种子指标均值。B0 表中的 M2 是同一 B2 对照，B0 的 M3–M6 以 M1 为锚点。',
        'M3=纯结构 P0-SelfSup；M4=双模固定先验；M5=双模 P0-SelfSup；M6=P1 验证校准。',
        'A0=体制自适应主干；A3/A5=自适应纯结构/双模探针。普通图与 M2（B0 表为 M1）及 M3/M5 相同；ConceptNet 单列，不替换原始锚点。A3/A5 探针回退仍指向原始锚点。',
        'GraIL 保留阶段七 valid_ind+test_ind 评估集；M6 用独立训练图 valid。Ingram 和 ConceptNet 用各自独立目标 valid。M6 不属于 P0，且跨图校准不保证是性能上界。',
        'P0 仅使用观察图微掩码伪标签；主干及全部辅助特征逐折重算。每图 P0 权重在读取 P1/最终测试标签前冻结。',
        '文本编码器冻结；匿名 MID、WordNet offset、Q/P 和纯数字标识符不编码。NLIngram 原始名称可读，实际开启文本。M3 的共识项独立于词面分支。',
        '本轮所有图均有历史测试暴露，locked 仅为历史分组名称，不声称首次盲测。',
        '95% 区间为 2000 次事实配对 Bootstrap；双向与训练种子绑定，宏平均在每图内独立抽样。不覆盖调参及训练种子总体不确定性。',
        '推理逻辑查询块=256。ConceptNet 特征微块=32，校准与自适应 GNN 微批次为 2，原始主干在最终测试中按阶段七使用 16 条事实的批次并释放不再使用的传播层结果。原始锚点按全量查询逐条核对。', '',
        '## 分组与逐图结果', '']
    for key, table in list(groups.items()) + list(tables.items()):
        lines += [f'### {key}', '', '|模型|MRR|Hits@1|Hits@3|Hits@10|MRR 95% CI|相对锚点 MRR 差值 95% CI|', '|---|---:|---:|---:|---:|---|---|']
        for name, row in table.items():
            metric = row['metrics']
            lines.append('|' + name + '|' + '|'.join(f'{metric[m]:.6f}' for m in METRICS) + '|' + str(row['ci95'][0]) + '|' + str(row['paired_delta_ci95'][0]) + '|')
        lines.append('')
    lines += ['## 是否全面超越', '']
    for family in ['B0', 'B2']:
        for model in ['M3', 'M5', 'A3', 'A5']:
            anchor = family + ('/M2' if family == 'B2' else '/M1')
            comparisons = {g: t[family + '/' + model]['metrics']['MRR'] - t[anchor]['metrics']['MRR'] for g, t in groups.items()}
            lines.append(f'- {family}/{model}：各组 MRR 差值 {comparisons}；全部三个组严格提升：{all(comparisons.get(g, 0) > 0 for g in ["locked", "historical", "external"])}。')
    lines += ['', '完整四指标置信区间、逐图权重、每折掩码、逐查询排名及基准一致性记录见本次 run 目录。',
              '原始观察图、数据标签、冻结模型与历史代码只读；资源上限触发时写 failed 状态并退出，不报告未完成的模型。',
              '源训练继承既有 B2 检查点与其协议，本轮无主干参数更新；未声称消除原历史源训练协议的局限。']
    report = run / 'report_universal_11g.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    if not smoke:
        (ROOT / 'report_universal_11g.md').write_text(report.read_text(), encoding='utf-8')
    return report
