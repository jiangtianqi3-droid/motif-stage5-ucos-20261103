from collections import defaultdict
from importlib.metadata import version
import sys
from statistics import mean

from src.io_utils import ROOT, RESULTS, PRIMARY, atomic, read, write_csv


def recovery_report():
    r = read(RESULTS / 'recovery/recovery.json')
    lines = ['# 冻结候选排名恢复报告', '', '按阶段七每批 16 条事实重算两个方向。推理入口只接收方向化的 (锚点, 关系) 和全实体枚举，过滤标签仅用于独立核验。', '',
             '未过滤矩阵的元素为 1/(60+悲观并列名次)。排序时同值按实体 ID 升序。阶段七 ZIP 的完整矩阵条目数为 '+str(len(r['archive_rrf_entries']))+'。', '',
             '| 图谱 | 主干 | 查询方向数 | 实体数 | 完整矩阵 SHA256 | 过滤名次 | 秒 | GPU 峰值字节 |',
             '|---|---|---:|---:|---|---|---:|---:|']
    for x in r['records']:
        seconds = f"{x['seconds']:.2f}" if x['seconds'] is not None else '复用'
        lines.append(f"| {x['dataset']} | {x['model']} | {x['directional_queries']} | {x['entities']} | `{x['cache_sha256']}` | 逐条一致 | {seconds} | {x['gpu_peak_bytes'] or '复用'} |")
    lines.extend(['', '历史 B0.npz 的字段和形状如下。'])
    for x in r['records']:
        if x['model'] == 'B0':
            lines.append(f"- {x['dataset']} `{x['historical_npz_schema']}`。ranks 仅存真实目标在两个方向的过滤名次。")
    lines += ['', 'WN18RR 采用历史 GraIL 合并 valid_ind/test_ind 协议。FBIngram 使用 test 查询和包括独立 valid 的过滤集合。两图均逐一核验观察边及其逆边。', '',
              '下游主干固定为 B2+D* 种子 1024。B0 单独恢复用于历史连续性核对。M5 为另一种报告基线。']
    atomic(RESULTS/'recovery/ranking_recovery_report.md', '\n'.join(lines)+'\n')


def aggregate(rows):
    result = []
    groups = defaultdict(list)
    for row in rows:
        for dataset in (row['dataset'], 'all'):
            groups[(dataset,row['arm'],row['fusion'])].append(row)
    for (dataset, arm, fusion), records in groups.items():
        old = [r['original_rank'] for r in records]
        new = [r['new_rank'] for r in records]
        data = dict(dataset=dataset, arm=arm, fusion=fusion, count=len(records))
        for metric, func in (('mrr',lambda x: 1/x), *[(f'hits{k}',lambda x,k=k: x<=k) for k in (1,3,10)]):
            a,b = mean(map(func,old)),mean(map(func,new))
            data.update({f'{metric}_baseline':a,metric:b,f'{metric}_delta':b-a,
                         f'{metric}_relative_pct':100*(b-a)/a if a else None})
        data.update(recall12=mean(r['top12'] for r in records), recall30=mean(r['top30'] for r in records),
                    positive=sum(r['positive'] for r in records), negative=sum(r['negative'] for r in records),
                    unchanged=sum(r['new_rank']==r['original_rank'] for r in records),
                    abstain=sum(r['status']=='abstain' for r in records),
                    invalid=sum(r['status']=='invalid' for r in records),
                    unavailable=sum(r['status'] in ('unavailable','uncertain') for r in records),
                    abstain_rate=mean(r['status']=='abstain' for r in records),
                    invalid_rate=mean(r['status']=='invalid' for r in records),
                    input_tokens=sum(r['input_tokens'] for r in records),
                    output_tokens=sum(r['output_tokens'] for r in records),
                    mean_latency=mean(r['seconds'] for r in records))
        result.append(data)
    return result


def report(state):
    lines = ['# MOTIF 拓扑感知 PCST 与 LLM API 可行性验证', '',
             f"运行状态为 {state['state']}。当前阶段为 {state['current_stage']}。",
             f"固定主干为 {PRIMARY}，一次冻结种子。实验将 WN18RR GraIL 合并划分和 FBIngram 原始测试分开核验。候选和方向使用阶段七原始实体 ID，提示仅传匿名观察事实。",
             f"运行环境为 WSL Ubuntu 24.04，Python {sys.version.split()[0]}，NumPy {version('numpy')}，PyTorch {version('torch')}，pcst_fast {version('pcst_fast')}。GPU 为 RTX 3060 Laptop。仓库版本及第三方文件位置见 [源码清单](THIRD_PARTY_PROVENANCE.md)。",
             'API 使用 DeepSeek 的 OpenAI 兼容 Chat Completions，deepseek-flash、temperature=0、关闭思考、JSON 模式，单次输出最多 512 Token。冒烟每图 2 条事实，正式每图另取 12 条，均评价两个方向。',
             'B0 和 B2+D* 的完整候选恢复、模型来源和 GPU 峰值见 [恢复报告](results/recovery/ranking_recovery_report.md)。',
             '普通 PCST 使用纯结构节点奖励和统一边成本。模体 PCST 在同一局部图上增加关系级模体和路径奖励、桥接与枢纽项，并先保留可达候选骨干。局部模体统计来自关系超图，不视为实体级实例数。候选覆盖和证据容量由分阶段算法控制。',
             '阶段七的过滤和悲观并列规则贯穿评价。外部重排将原 Top-12 的分数层级分配到新位置，同层排名仍按悲观规则计入。候选外维持原结构值。保守融合要求至少引用一条本方法实际选中的候选连接路径事实。', '',
             '所有检索组共享 300 节点、800 本地边、60 条事实和 3000 输入 Token 上限。提示上限以 UTF-8 字节加模板余量估算，账本优先记录服务商实报 Token。A5 在同一局部图中固定种子抽样，以 A4 证据条数为目标。', '',
             f"API 请求已登记 {state['api_requests_used']} 次，输入 {state['api_input_budget_used']}，输出 {state['api_output_budget_used']}。账本用量包括未知请求的保守预留。"]
    if state.get('error'):
        lines.append('当前中断原因为 '+state['error']+'。')
    path = RESULTS / 'retrieval_summary.json'
    if path.is_file():
        lines += ['', '| 图谱 | 方法 | 查询 | 候选覆盖/可达 | 无路径比例 | 节点/事实 | 平均模体支持 | 枢纽比例 | 冗余率 | 输入上界 | 毫秒 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        retrieval = read(path)
        for r in retrieval:
            lines.append(f"| {r['dataset']} | {r['arm']} | {r['queries']} | {r['coverage']:.1%} | {r['no_evidence']:.1%} | {r['nodes']:.1f}/{r['edges']:.1f} | {r['motif_support']:.3f} | {r['hub_ratio']:.1%} | {r['duplication']:.1%} | {r['input_bound']:.0f} | {r['milliseconds']:.1f} |")
        lines += ['', '覆盖率分母为当前查询集中的可达候选数。无路径指选中证据中没有连到候选的 1–3 跳有向路径。详表保留路径长度及原始边 ID。模体支持为启发式指标。',
                  'PPR 的局部候选图也优先保留粗检索阶段的候选路径。A4 与 A3 同时改变多种机制，须结合单因素消融判断原因。']
        for dataset in sorted({r['dataset'] for r in retrieval}):
            rows = {r['arm']:r for r in retrieval if r['dataset']==dataset}
            a,ppr,base,off = (rows[k] for k in ('A4','A2','A3','A4-no-motif'))
            lines.append(f"- {dataset} 的 A4 相比 A2 覆盖率变化 {100*(a['coverage']-ppr['coverage']):+.1f} 个百分点，"
                         f"平均事实数变化 {a['edges']-ppr['edges']:+.1f}。相比 A3 覆盖变化 "
                         f"{100*(a['coverage']-base['coverage']):+.1f} 个百分点。只关闭模体奖励时覆盖变化 "
                         f"{100*(off['coverage']-a['coverage']):+.1f} 个百分点，平均模体支持变化 "
                         f"{off['motif_support']-a['motif_support']:+.3f}。")
    for stage in ('smoke', 'pilot'):
        path = RESULTS / f'{stage}_summary.json'
        if not path.is_file():
            continue
        lines += ['', f'## {stage}', '', '| 图谱 | 方法 | 融合 | 查询 | MRR | ΔMRR | 相对变化 | Hits@1/3/10 | 正/负 | 弃权/非法 | 输入/输出 |',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        metrics = read(path)
        for r in metrics:
            relative = f"{r['mrr_relative_pct']:+.2f}%" if r['mrr_relative_pct'] is not None else '—'
            lines.append(f"| {r['dataset']} | {r['arm']} | {r['fusion']} | {r['count']} | {r['mrr']:.4f} | {r['mrr_delta']:+.4f} | {relative} | {r['hits1']:.3f}/{r['hits3']:.3f}/{r['hits10']:.3f} | {r['positive']}/{r['negative']} | {r['abstain']}/{r['invalid']} | {r['input_tokens']}/{r['output_tokens']} |")
        lines += ['', '分组 MRR 绝对变化使用原始 0–1 标度。逐组 Hits 的绝对和相对变化、Recall@12/30、失败率和延迟见指标 CSV。两种融合方式共享 API 请求，不能把两行成本相加。']
        direct = {r['arm']:r for r in metrics if r['dataset']=='all' and r['fusion']=='direct'}
        baseline = next((r for r in metrics if r['dataset']=='all' and r['arm']=='A0'),None)
        if baseline and {'A2','A3','A4','A5'} <= direct.keys() and len({baseline['count'],*(r['count'] for r in direct.values())}) == 1:
            a4 = direct['A4']
            comparisons = '，'.join(f"相对 {arm} 为 {a4['mrr']-direct[arm]['mrr']:+.4f}" for arm in ('A2','A3','A5'))
            lines.append(f"同样本直接重排的 A4 相对 A0 的 MRR 变化为 {a4['mrr']-baseline['mrr']:+.4f}，"
                         f"{comparisons}。正向翻转 {a4['positive']} 条，负向翻转 {a4['negative']} 条。")
        cases = RESULTS / f'{stage}_cases.md'
        if cases.is_file():
            lines += ['', cases.read_text(encoding='utf-8')]
    lines += ['', '检索覆盖、结构支持和 API 端预测成效分别依据上述实测数据。小样本结果只用于描述，尚不能据此断定接入 KRLM 会提升全实体预测。']
    atomic(ROOT/'report_poc_v1.md', '\n'.join(lines)+'\n')
