"""Report every method and seed; do not equate positive point estimates with proof."""
import csv
import numpy as np
from .util import read,dump,ROOT,STAGE5,now
from .statistics import holm

def report(run,datasets,families,smoke=False):
    rows=[];comparisons=[];summaries=[]
    lines=['# 隔离修复后的实验报告','',f'生成时间：{now()}；运行：`{run.name}`。','',
           '本轮为历史暴露数据上的修复复验，不能称为首次盲测。保留全部负结果；没有通过修改指标或筛掉困难查询保证提升。','',
           '训练负采样只用源图 train；源验证过滤只用 train+valid。三个种子沿用相同架构和每种子 500 步预算，源验证选择检查点。','',
           '目标验证事实按共享公开查询的连通组分配至结构选参与 LLM 校准，两个角色不共享事实或查询。结构仅三档残差，使用查询等权和配对 one-SE 简单模型规则。LLM 随机抽取去重查询，少于 32 个不同查询/提示时回退为零。','',
           '主要比较为 R4−R1、R4−R3、R5−R4；主指标 MRR，同时报告 Hits@10。配对符号翻转检验以事实为单位，全部主比较统一 Holm 校正；关系/查询分组区间仅作依赖敏感性分析。','',
           '|图谱|模型/种子|方法|测试事实|MRR|Hits@1|Hits@3|Hits@10|','|---|---|---|---:|---:|---:|---:|---:|']
    if smoke:lines.insert(2,'**本报告仅为小样本执行检查，不构成有效性证据。**')
    for key in datasets:
        for family in families:
            work=run/key/family;s=read(work/'summary.json');summaries.append(s)
            for method,m in s['metrics'].items():
                rows.append(dict(dataset=key,family=family,method=method,test_facts=s['data_counts']['test_facts'],**m))
                lines.append(f"|{key}|{family}|{method}|{s['data_counts']['test_facts']}|{m['MRR']:.6f}|{m['Hits1']:.6f}|{m['Hits3']:.6f}|{m['Hits10']:.6f}|")
            for name,c in s['primary_comparisons'].items():
                comparisons.append(dict(dataset=key,family=family,comparison=name,delta_MRR=c['delta_MRR'],delta_Hits10=c['delta_Hits10'],
                    p=c['fact_sign_flip']['p'],changed_facts=c['changed_facts'],relation_lo=c['relation_clusters']['lo'],relation_hi=c['relation_clusters']['hi']))
    adjusted=holm([c['p'] for c in comparisons])
    for c,p in zip(comparisons,adjusted):c['p_holm']=p
    lines+=['','## 主要归因比较','','|图谱|模型|比较|ΔMRR|ΔHits@10|变化事实数|原始 p|Holm p|','|---|---|---|---:|---:|---:|---:|---:|']
    for c in comparisons:
        lines.append(f"|{c['dataset']}|{c['family']}|{c['comparison']}|{c['delta_MRR']:+.6f}|{c['delta_Hits10']:+.6f}|{c['changed_facts']}|{c['p']:.6f}|{c['p_holm']:.6f}|")
    seed_rows=[]
    lines+=['','## 三种子稳定性（描述性）','','|图谱|方法|种子数|MRR均值|种子标准差|最小|最大|','|---|---|---:|---:|---:|---:|---:|']
    for key in datasets:
        for method in sorted({r['method'] for r in rows}):
            vals=[r['MRR'] for r in rows if r['dataset']==key and r['method']==method and r['family'].startswith('T-4D-s')]
            if vals:
                r=dict(dataset=key,method=method,seeds=len(vals),MRR_mean=float(np.mean(vals)),MRR_seed_std=float(np.std(vals,ddof=1)) if len(vals)>1 else None,minimum=min(vals),maximum=max(vals));seed_rows.append(r)
                lines.append(f"|{key}|{method}|{len(vals)}|{r['MRR_mean']:.6f}|{r['MRR_seed_std']}|{r['minimum']:.6f}|{r['maximum']:.6f}|")
    lines+=['','## 样本量与校准限制','']
    for s in summaries:
        w=run/s['dataset']/s['family'];p=read(w/'validation_partition.json');llm=read(w/'selected_llm.json') if (w/'selected_llm.json').exists() else {}
        lines.append(f"- {s['dataset']}/{s['family']}：{s['data_counts']}；校准不同查询 {p['unique_queries']['calibration']}；LLM强度 {llm.get('strength','未运行')}；不同提示 {llm.get('unique_prompts','不适用')}。")
    lines+=['','## 结果解释','',
        'ReverseThenM0 是无需测试标签的捷径对照：优先观察图中已存在相同关系反向边的实体，再按原模型分数排序。完整分层指标在各组 summary.json 的 stratified 中，同时报告验证已见/未见查询、有/无反向关系线索。','',
        '正向 MRR 点差不自动构成新边权或 LLM 的独立贡献；主比较需结合校正后的 p、Hits@10 变化、跨图和跨种子一致性。符号翻转检验仍有配对差值交换性假设，图中相关事实和少数关系使跨图泛化推断仍有限。','',
        '源验证过滤口径已经修正，因此新旧源验证 MRR 不能直接相减。目标图测试口径与原路线 A 相同，可比较同图的冻结基线，但新选参规则与原规则不同，方法总差异并非单一因素因果效应。','',
        '历史测试暴露与现有小数据集无法被代码修复。尚须新增未参与路线设计的数据/来源，预先锁定协议后一次性评估；当前报告不会把已有图或新种子包装成新的独立数据。']
    for name,data in [('main_metrics.csv',rows),('primary_comparisons.csv',comparisons),('seed_robustness.csv',seed_rows)]:
        if data:
            with (run/name).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    dump(run/'multiplicity.json',dict(method='Holm',family_size=len(comparisons),comparisons=comparisons))
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not smoke:(ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
