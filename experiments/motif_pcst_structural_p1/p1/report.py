from collections import defaultdict
from statistics import mean

from p1.config import ROOT
from src.io_utils import atomic, read


def pct(value):
    return 'N/A' if value is None else f'{value:.1%}'


def dec(value,digits=3):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def paired_cases(folder, summary):
    dataset, phase = summary['dataset'], summary['phase']
    marks = read(folder/'runs'/phase/dataset/'diagnostics.json')
    by_query = defaultdict(dict)
    for r in marks:
        if r['k']==12 and r['budget']==32:
            by_query[r['query_id']][r['method']]=r
    checks = [
        ('模体支持提高且正确实体名次下降','R2','R3',lambda b,a:a['support']>b['support'] and a['fixed']>b['fixed']),
        ('覆盖改善且固定融合预测名次未改善','R1','R2',lambda b,a:a['coverage']>b['coverage'] and a['fixed']>=b['fixed']),
        ('枢纽控制去除正确候选路径','R3','R4',lambda b,a:b['target_path'] and not a['target_path']),
        ('R5 证据融合导致名次下降','R5','R5',lambda b,a:a['fixed']>a['baseline']),
        ('R5 无正确候选连接路径','R5','R5',lambda b,a:not a['target_path'])]
    result = []
    for label,u,v,condition in checks:
        pair = next(((row[u],row[v]) for row in by_query.values() if u in row and v in row and condition(row[u],row[v])),None)
        if pair is None:
            result.append(f'- {label}在该样本中未出现。')
        else:
            b,a = pair
            result.append(f"- {label}见 {a['query_id']}。原始排名 {a['baseline']}，{u}/{v} 的固定融合排名为 "
                          f"{b['fixed']}/{a['fixed']}，正确目标为 {a['target']}，对应路径事实 ID 为 "
                          f"{b['path']}/{a['path']}，两组覆盖候选数为 {b['coverage']}/{a['coverage']}。")
    return result


def make_report(folder,state):
    lines = ['# P1 候选判别型拓扑 PCST 与纯结构关联预测', '',
             f"运行状态为 {state['state']}，阶段为 {state['current_phase']}。P1 LLM API 请求 {state['llm_api_requests']} 次。",
             f"已完成 {', '.join(state['completed_experiments']) or '无'}。运行 ID 为 `{state['run_id']}`。"]
    if state['errors']:
        lines.extend([f"- {e['phase']}：{e['error']}" for e in state['errors']])
    if (folder/'audit.json').is_file():
        audit = read(folder/'audit.json')
        cp = audit['checkpoint']
        lines.extend([f"代码提交 {audit['commit']}。B2 检查点为 {cp['checkpoint']}，SHA256 为 `{cp['checkpoint_sha256']}`。D* 固定为阶段七 D-Sender（gamma 0.75，alpha 0.20，均值锚定）。",
                      f"WSL Ubuntu 24.04，RTX 3060 Laptop，本次已登记的 CUDA 张量峰值 {state['gpu_peak_memory_mb']:.1f} MiB。上一阶段 PoC 代码为 {audit['poc']}。"])
    lines.extend(['', '上一阶段正式 API 报告中 A4 相对冻结主干的 MRR 变化为 −0.2985，正向 4 条、负向 24 条。P1 独立使用结构数据。',
                  'R0–R5 使用相同冻结 B2+D* 1024 候选、PPR 局部图、定向事实和预算。R1–R5 使用同一有根 PCST。R2 加入固定候选路径骨干，R3 加入来自六层实际冻结 B2 T3 门控的奖励，R4 加入发送端成本，R5 加入路径可信度、候选排他度和扩展去冗余。后者在快速验证中有三个单独移除的对照。',
                  '关系超图角色与归一化沿用阶段七，T3 门控读取按模体类型标准化的第 4 个通道（排他性）。P1 对包含查询关系的模体按候选边关系投影，逐层压缩 gate/(1+gate) 并用计数加一收缩，六层取平均。该项为全图关系先验。实际有向实体路径及其模式另行计算。',
                  '发送端使用历史 Triton 的 edge_index[1] 约定。P1 的枢纽控制是检索成本，并未在 PPR 的逐行归一化里加入会抵消的度数常数。PCST 求解先预留候选骨干，再收缩和扩展，是分阶段近似。虚拟边还原为原始事实，观察三元组重复记录只保留一个 ID，不同关系保留各自身份。',
                  '每候选最多保留四种关系路径实例，使用单条最佳长度和枢纽折减后的源域可信度。无路径记中性加分。排序代理从 1/(60+悲观原始名次) 还原为秩层级，共用一套刻度。在 Top-K 位置重新分配原分数层级，保留完整过滤集合、其他实体顺序和悲观并列。',
                  '历史 M5 使用相同测试行和过滤协议，按原三种子分别计算指标再平均。M5 可使用目标观察图内校准和适用的词法输入，因此作为完整历史对照独立展示。P1 路径估计和融合选择只来自源域，目标观察图只用于单次推理所需的结构特征。四图均有此前历史测试暴露，迁移判断限于 P1 的冻结训练和调参。'])
    calibration = folder/'source_calibration.json'
    if calibration.is_file():
        source = read(calibration)
        p = source['mask_diagnostics']
        lines += ['', '## 源域遮蔽与融合校准', '',
                  f"在 {source['source']} 观察图中一次移除预固定的拟合和诊断事实，以及全部相应逆向记录。重新构图并计算度数与 1–3 跳路径。每个正候选匹配最多三名可达但在原观察图中无该关系事实的候选。后者是未观察控制，可能包含假负例。",
                  'N(p) 为该模式下每个候选选一条最短路径后的暴露次数，N(p,r) 为被遮蔽观察正事实的暴露次数。模式编码路径长度、关系与查询同向或互逆、其余边的规范方向和中间节点度数区间。先用 Beta(1,1) 平滑粗模式，再给细模式加入权重 4 的粗模式先验。没有跨图关系 ID 映射。遮蔽诊断只用重算的路径统计，不复用未遮蔽 B2 门控。',
                  f"独立遮蔽诊断正例 {p['positives']}，未观察控制 {p['nonedges']}。正例/控制有路径比例为 {pct(p['positive_path_rate'])}/{pct(p['nonedge_path_rate'])}，平均可信度为 {dec(p['mean_positive_support'])}/{dec(p['mean_nonedge_support'])}。描述性 Brier 为 {dec(p['proxy_brier'])}，实验采样先验对照为 {dec(p['rate_baseline_brier'])}。",
                  f"另用 {source['tuning_queries']} 条历史源域验证方向查询选择融合系数，候选值与 MRR 为 {source['source_grid']}，最终系数 {source['lambda_selected']}。固定 0.15 的融合仅做诊断。"]
    results = [read(path) for path in sorted((folder/'runs').glob('*/*/summary.json'))]
    phases = ('quick','expanded','transfer','sensitivity')
    for phase in phases:
        group = [r for r in results if r['phase']==phase]
        if not group:
            lines += ['', f'## {phase}', '', '未执行。']
            continue
        for item in group:
            key = item['dataset']
            lines += ['',f'## {phase} · {key}', '',
              f"运行 {item['total_queries']} 个查询与预算组合，用时 {item['wall_seconds']:.1f} 秒。观察图 {item['graph']['nodes']} 个节点、{item['graph']['edges']} 条有向记录、{item['graph']['relations']} 个含逆方向的关系类型，出度 P90 为 {item['graph']['degree_p90']:.0f}。",
              'WN18RR、NELL 和 FB15k237 沿用 GraIL 合并划分。FBIngram 使用原始测试集，验证事实只参加离线过滤。覆盖率同时报告全部候选和可达候选两种分母。',
              '', '| 方法 | K | 预算 | 全部覆盖 | 可达覆盖 | 节点/事实 | 模体 | 路径可信度 | 枢纽/冗余 | 利用率 | 毫秒 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for r in item['retrieval']:
                lines.append(f"| {r['method']} | {r['k']} | {r['budget']} | {pct(r['candidate_coverage_all'])} | "
                             f"{pct(r['coverage_reachable'])} | {dec(r['selected_nodes'],1)}/{dec(r['selected_observed_edges'],1)} | "
                             f"{dec(r['motif_support'])} | {dec(r['path_reliability'])} | {pct(r['hub_ratio'])}/{pct(r['evidence_redundancy'])} | "
                             f"{pct(r['budget_utilization'])} | {dec(r['retrieval_latency_ms'],1)} |")
            lines += ['', '| 方法 | 融合 | K/预算 | MRR | ΔMRR | 95% 配对区间 | Hits@1/3/10 | 正/负 |',
                      '|---|---|---:|---:|---:|---:|---:|---:|']
            for p in item['prediction']:
                iv = '/'.join(f'{x:+.4f}' for x in p['ci'])
                lines.append(f"| {p['method']} | {p['fusion']} | {p['k']}/{p['budget']} | {p['mrr']:.4f} | "
                             f"{p['delta_mrr']:+.4f} | {iv} | {pct(p['hits1'])}/{pct(p['hits3'])}/{pct(p['hits10'])} | "
                             f"{p['positive']:g}/{p['negative']:g} |")
            if phase!='sensitivity':
                k,b = 12,32
                rows = {r['method']:r for r in item['retrieval'] if r['k']==k and r['budget']==b}
                pred = {r['method']:r for r in item['prediction'] if r['k']==k and r['budget']==b and r['fusion']=='fixed_015'}
                lines += ['', '机制对照以相同查询配对，固定 λ=0.15 作为诊断。']
                for a,b in [('R1','R2'),('R2','R3'),('R3','R4'),('R4','R5')]:
                    ra,rb = rows[a],rows[b]
                    lines.append(f"- {b} 相对 {a}：实际事实 {rb['selected_observed_edges']-ra['selected_observed_edges']:+.2f}，"
                                 f"正确候选有路径比例 {100*(rb['target_path']-ra['target_path']):+.2f} 个百分点，"
                                 f"固定融合 MRR {pred[b]['mrr']-pred[a]['mrr']:+.4f}。")
                lines.extend(paired_cases(folder,item))
    preferred = [r for r in results if r['phase'] in ('expanded','transfer')]
    if preferred:
        lines += ['', '## 跨图判断', '']
        rows = [p for r in preferred for p in r['prediction'] if p['k']==12 and p['budget']==32]
        for method in ('frozen_B2+Dstar','B0','M5_three_seed','R0','R1','R2','R3','R4','R5'):
            selected = [p for p in rows if p['method']==method and p['fusion'] in ('fusion','baseline')]
            if selected:
                detail = '，'.join(f"{p['dataset']} {p['mrr']:.4f} ({p['delta_mrr']:+.4f})" for p in selected)
                lines.append(f"- {method} 图谱宏平均 MRR {mean(p['mrr'] for p in selected):.4f}。分图为 {detail}。")
        primary = [p for p in rows if p['method']=='R5' and p['fusion']=='fusion']
        robust = [p['dataset'] for p in primary if p['ci'][0]>0]
        negative = [p['dataset'] for p in primary if p['delta_mrr']<0]
        lines += ['', f"源域锁定的 R5 融合正向区间图谱为 {robust or '无'}，负向点估计图谱为 {negative or '无'}。",
                  '进入 KRLM 前需出现跨图一致的增量预测证据。覆盖提升、门控奖励和路径分数分别按配对消融判断。11 图和轻量 MLP 本轮未安排，待本轮独立实验结果决定。']
    atomic(ROOT/'report_structural_p1.md','\n'.join(lines)+'\n')
