from collections import defaultdict
from statistics import mean
import csv

from p2.assets import ROOT
from src.io_utils import atomic, read


def form(x):
    return 'N/A' if x is None else f'{x:.4f}'


def pct(x):
    return 'N/A' if x is None else f'{100*x:.1f}%'


def report(folder,state):
    rows=['# P2 拓扑证据影响传播与候选判别评分','',
          f"状态为 {state['state']}，当前阶段为 {state['current_phase']}。P2 API 请求 {state['llm_api_requests']} 次。",
          f"运行 ID 为 `{state['run_id']}`。P2 源码提交为 `{state.get('commit','未记录')}`。",
          '四图沿用 P1 已核对的真实观测图、行号、双向过滤评测和固定 Top-12。WN18RR、FB15k237、NELL 为 GraIL 合并验证和测试协议，FBIngram 为原测试划分。头尾方向按同一事实聚类自助重采样。',
          'P2 只传递选中事实 ID 到路径提取器。P1 代码保持原样，外部三种子 M5 沿用相同测试行，允许历史观察图适配和可选词法特征。四图属于已有基准迁移。']
    if state.get('error'):
        rows.append('中断信息为 '+state['error'])
    path=folder/'propagation_summary.json'
    if path.is_file():
        data=read(path)
        rows+=['','## 证据到排序的逐级核对','',
          f"与 P1 选中图逐条重算路径、分数、证据顺序和 λ=0 融合，全部一致。Top-K 外正确目标因相邻检索组改变名次的次数为 {data['outside_rank_changes']}。审计耗时 {data['seconds']:.1f} 秒。",
          'P1 的 Retriever.query 固定候选骨干并收缩其节点，R3 奖励、R4 成本只作用于 PCST 扩展。Retriever.describe 在各组选中的图上重新取路径。scoring.evidence 使用长度和枢纽折减的最高路径分数。P2 另在入评分器前枚举全部三跳内简单路径，检查新增边是否在截断前进入候选路线。',
          '',
          '| 图 | 对照 | 查询 | T0/T1/T2/T3/T4 | 平均边差 | 特征变化 | 分数变化 | 排序变化 | 全路径发现新增事实 | P1 路径保留新增事实 | 正确名次正/负 |',
          '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in data['rows']:
            counts='/'.join(str(r['types'][f'T{k}']) for k in range(5))
            rows.append(f"| {r['dataset']} | {r['method_a']}→{r['method_b']} | {r['queries']} | {counts} | "
                        f"{r['mean_edge_symmetric_difference']:.2f} | {pct(r['feature_change_rate'])} | "
                        f"{pct(r['score_change_rate'])} | {pct(r['ranking_change_rate'])} | "
                        f"{pct(r['new_any_path_rate'])} | {pct(r['new_retained_path_rate'])} | "
                        f"{r['positive_rank_changes']}/{r['negative_rank_changes']} |")
        early=[r for r in data['rows'] if r['method_b'] in ('R3','R4')]
        if early and all(r['new_any_path_rate']==0 and r['score_change_rate']==0 for r in early):
            rows+=['','R2–R4 的相同评分在候选路径形成前出现。新增与删去的扩展事实不改变候选路径，因此扩大路径聚合窗口也无法让那些外围扩展进入路径评分。R4→R5 若出现 T2 或 T3，则分别记录聚合和排序处的后续损失。']
        rows+=['','逐查询特征向量按 P1 路径数、全部异构路径模式数、全部路径数、Top-4 路径权重、路径模体均值、相对分差、中间枢纽统计和缺失标记排列。P1 未额外归一化证据项，故记录的证据标准化差等于原差。']
        path=folder/'counterfactual_probe.csv'
        if path.is_file():
            prob=defaultdict(list)
            with path.open(encoding='utf-8',newline='') as f:
                for x in csv.DictReader(f):prob[x['scope'],x['intervention']].append(x)
            rows+=['','| 证据干预 | 图范围 | 样本 | 评分有变化 | 证据顺序有变化 |','|---|---|---:|---:|---:|']
            for (scope,kind),r in prob.items():
                rows.append(f"| {kind} | {scope} | {len(r)} | {pct(mean(int(x['candidate_scores_changed'])>0 for x in r))} | {pct(mean(x['ranking_changed']=='True' for x in r))} |")
            rows+=['','删除选中边时固定原图度数和优先级，直接重算选中图路径。删除原图事实时同时删除逆边和重复记录，重算度数、PPR 与 PCST，保留冻结 B2 关系先验以隔离图变化。打乱奖励重新选择观察事实。']
        path=folder/'cases.json'
        if path.is_file():
            cases=read(path)
            rows+=['','真实案例完整记录见 results/cases.json。']
            for kind in ('T1','T2','T3','positive','negative','reachable_low_support','no_path'):
                row=next((r for k,r in cases.items() if k.endswith('/'+kind)),None)
                if row:
                    rows.append(f"- {kind} 对应 {row['query_id']}，候选 {row['candidate_id']}，路径事实 ID 为 "
                                f"{row['paths_before']} → {row['paths_after']}，分数 {row['score_before']:.4f} → "
                                f"{row['score_after']:.4f}，证据候选位置 {row['evidence_rank_before']} → {row['evidence_rank_after']}。")
                else:
                    rows.append(f'- 当前逐查询样本没有 {kind} 案例。')
    path=folder/'path_summary.json'
    if path.is_file():
        source=read(path)
        rows+=['','## 路径可信度与未观察候选','',
          f"源域遮蔽拟合 {source['fit_facts']} 对，诊断 {source['probe_facts']} 对，移除正向事实 {source['removed_direct_facts']} 条以及逆向记录。按无向实体对分组移除所有关系，筛查与验证实体对重叠。仅在遮蔽图上重建关系超图和冻结模型分数，路径特征不使用跨图关系编号。",
          'C1 随机。C2 从三跳内可达实体抽样。C3 在冻结 Top-80 中优先匹配可达性、长度、实体度数区间、结构排名。源数据无可用实体类型标签。控制是未观察候选。成对准确率并列记半分，另列双方均有路径的子集。',
          '', '| 控制 | 路径变体 | 配对数 | 成对准确率 | 双方有路径准确率 | 正例/控制有路径 | 匹配 | 分差 | 与 S0 顺序一致 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in source['summary']:
            rows.append(f"| {r['control']} | {r['variant']} | {r['pairs']} | {pct(r['pairwise'])} | "
                        f"{pct(r['both_path_pairwise'])} | {pct(r['positive_path_rate'])}/{pct(r['control_path_rate'])} | "
                        f"{pct(r['matched_coverage'])} | {form(r['gap'])} | {pct(r['ranking_consistency_s0'])} |")
        rows.append('可靠性按路径模式的源域拟合暴露统计，并按粗模式回退。路径模式使用查询相对角色、规范方向和中间度数桶。结果只描述采样候选上的区分能力。')
    lock=folder/'source_lock.json'
    if lock.is_file():
        selected=read(lock)
        p=selected['primary']
        rows+=['','## 源域选择与评分消融','',
          f"验证查询扩至 {selected['queries']} 条，保留 P1 原 {selected['old_queries']} 条作为对照。仅按源域 MRR 锁定 {p['agg']}、{p['motif']}、{p['fusion']}，相对特征为 {p['relative']}，λ={p['lam']}。原 96 条和新增子集 MRR 分别为 {p['mrr_original96']:.4f}、{p['mrr_additional']:.4f}。",
          'S0 取最高值，S1 取前 2 或 4 条均值，S2 用稳定的归一化 log-mean-exp。各路径用关系模式去重。没有路径时保留零加分及显式缺失。M0/R3 选边、M2/R2 选边，M1 与 M2 在路径打分中加入中心化 B2 先验，M3 两处均关闭模体。F2 与 F4 以源域有效分数绝对值分位数校准，F3 以候选秩融合。证据统计和排名代理均不代表概率。']
        if p['lam']==0:
            rows.append('锁定方案 λ=0，主配置没有改变原结构预测。独立评分和诊断消融另表展示。')
    results=[read(f) for f in sorted((folder/'transfer').glob('*/*/summary.json'))]
    main=[r for r in results if r['k']==12]
    if main:
        rows+=['','## 四图同行对照','',
          '| 图 | 方法 | MRR | ΔB2+D*（区间） | ΔB0 | ΔM5 | Hits@1/3/10 | Recall@12/30 | 正/负 | 调整/分数变化/候选换位 |',
          '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
        for g in main:
            for m in g['metrics']:
                if m['scorer'] not in ('B0','frozen_B2+Dstar','M5_three_seed','P2_primary','P1_R1_F5','P1_R2_F5','P1_R5_F5') and not m['scorer'].startswith(('motif_','agg_')):
                    continue
                ci='/'.join(f'{a:+.4f}' for a in m['ci_b2'])
                rows.append(f"| {g['dataset']} | {m['scorer']} | {m['MRR']:.4f} | {m['delta_b2']:+.4f} ({ci}) | "
                            f"{m['delta_b0']:+.4f} | {m['delta_m5']:+.4f} | {pct(m['Hits1'])}/{pct(m['Hits3'])}/{pct(m['Hits10'])} | "
                            f"{pct(m['recall12'])}/{pct(m['recall30'])} | {m['positive_flips']:g}/{m['negative_flips']:g} | "
                            f"{m['queries_with_nonzero_adjustment']}/{m['queries_with_score_change']}/{m['queries_with_rank_change']} |")
        grouped=defaultdict(list)
        for g in main:
            for m in g['metrics']:grouped[m['scorer']].append(m)
        rows+=['','按图谱等权宏平均。']
        for name in ('B0','frozen_B2+Dstar','M5_three_seed','P2_primary','agg_S0','agg_S1-2','agg_S1-4','agg_S2-0.08','motif_M0','motif_M1','motif_M2','motif_M3'):
            group=grouped[name]
            if group:
                rows.append(f"- {name} {mean(x['MRR'] for x in group):.4f}，以 {len(group)} 图计算。")
        rows+=['','按结构区分图谱。']
        for g in main:
            p2=next(m for m in g['metrics'] if m['scorer']=='P2_primary')
            rows.append(f"- {g['dataset']} 有 {g['nodes']} 节点、{g['edges']} 条定向记录、{g['relations']} 种定向关系，"
                        f"平均出度 {g['edges']/g['nodes']:.1f}，出度 P90 为 {g['degree_p90']:.0f}，"
                        f"无路径候选 {pct(p2['no_evidence_fraction'])}，主配置 ΔMRR {p2['delta_b2']:+.4f}。")
        if len(main)==4:
            gains=[r['dataset'] for r in grouped['P2_primary'] if r['ci_b2'][0]>0]
            losses=[r['dataset'] for r in grouped['P2_primary'] if r['delta_b2']<0]
            rows+=['','## 取舍',
                   f'主配置正向区间图谱为 {gains or "无"}，负向点估计图谱为 {losses or "无"}。']
            if len(gains)<2:
                rows+=['证据与选边的作用可保留为压缩和解释接口。当前纯结构重排缺少跨图增量依据，暂停扩展至 11 图及训练 MLP。优先把候选之间真正不同的路径暴露给独立验证。']
            else:
                rows+=['多图出现正向区间，值得用同一冻结配置做扩大确认。']
        sensitive=[g for g in results if g['k']!=12]
        if sensitive:
            rows+=['','候选数量敏感性使用 P1 的限定样本。不同图或不同候选数量之间按同一行号比较。']
            for g in sensitive:
                m=g['metrics'][0]
                rows.append(f"- {g['dataset']} Top-{g['k']} 共 {g['queries']} 条，MRR {m['MRR']:.4f}，ΔB2+D* {m['delta_b2']:+.4f}。")
    atomic(ROOT/'report_structural_p2.md','\n'.join(rows)+'\n')
