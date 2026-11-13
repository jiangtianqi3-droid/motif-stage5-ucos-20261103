"""Chinese report generated from completed per-query measurements only."""
import csv
from .util import *

def report(run,datasets,families,smoke=False):
    title='路线 A 冒烟测试报告' if smoke else '路线 A 全量实验报告'
    lines=[f'# {title}', '', f'生成时间：{now()}。运行编号：`{run.name}`。', '',
           '本轮检验统一边权、结构扩召回与分差触发式大模型精排的实际效果。提升与下降均按同一批查询保留。', '',
           '## 数学修复与实验边界', '',
           r'统一边权为 $w_q(u,r,v)=\phi(\rho_q(r))(1+d_{out}(u))^{-0.25}(1+d_{r,out}(u))^{-0.5}(1+d_{out}(v))^{-0.25}$，再除以全图边权均值。沿用 E3a 的支持加权 RW 与排他度映射。', '',
           '发送端全局度因子在前向行归一化中依然会抵消，本轮不声称改变这一恒等式；新增的关系局部度与接收端度因子在邻边之间变化时可以保留在转移概率中。所有邻边因子恰好相同时，它们也会抵消。反向矩阵重新按目的端行归一化。', '',
           '结构证据分别对路径权重、转移概率乘积和 PPR 交汇分数做查询内对数 Min-Max 归一化，再加权并把最大综合支持锚定为 1；无证据为 0。原始 logit 在查询内线性归一化后添加残差。门控和榜首保护是经验约束，不能保证测试指标零退化。', '',
           '采用用户确认的答案隔离协议。验证候选只过滤观察事实，测试候选只过滤观察及验证事实。检索、结构评分、LLM 提示与触发均不读取测试答案。只在离线计分时过滤其他测试正例并恢复当前目标。报告中明确统计其与标准先按测试真值过滤再取 Top-30 的差异。', '',
           '最终排名通过重排候选所占原始 logit 槽位计算；完整保留悲观并列秩。候选之外的分数不变，原始候选分数多重集不变，并逐查询断言池外目标排名不变。这里的 Top-30 是答案隔离候选池，不能冒称精确 Filtered Top-30。', '',
           '每图最多固定无标签抽样 256 条验证事实，双向用于网格选择。LLM 在其中最多 64 个触发查询校准残差，选择完成后固定再评测测试集。Grail 历史协议合并了 inductive valid 与 test，本轮主表只汇报官方 test，因此不能把本表基准与历史合并主表直接当作同一测试集比较。', '',
           'B0 与 T-4D 分别运行 M0 和 R1–R5。R1 无权、R2 历史度权、R3 历史 E3a、R4 统一边权。FBIngram_25 跳过 LLM。置信区间按原始事实共同重采两方向，2,000 次；不按方向假设独立。', '',
           '## 标准全实体 Filtered 指标', '',
           '|图谱|冻结基线|方法|查询数|MRR|Hits@1|Hits@3|Hits@10|',
           '|---|---|---|---:|---:|---:|---:|---:|']
    rows=[];summaries=[]
    for key in datasets:
        for family in families:
            summary=read(run/key/family/'summary.json');summaries.append(summary)
            for method,m in summary['metrics'].items():
                row=dict(dataset=key,family=family,method=method,queries=summary['queries'],**m);rows.append(row)
                lines.append(f"|{key}|{family}|{method}|{summary['queries']}|{m['MRR']:.6f}|{m['Hits1']:.6f}|{m['Hits3']:.6f}|{m['Hits10']:.6f}|")
    lines+=['','## 相对各自冻结基线的配对差值及 95% 区间','','|图谱|基线|方法|指标|差值|下界|上界|','|---|---|---|---|---:|---:|---:|']
    interval_rows=[]
    for s in summaries:
        for method,intervals in s['paired_bootstrap'].items():
            for metric,v in intervals.items():
                interval_rows.append(dict(dataset=s['dataset'],family=s['family'],method=method,metric=metric,**v))
                lines.append(f"|{s['dataset']}|{s['family']}|{method}|{metric}|{v['delta']:+.6f}|{v['lo']:+.6f}|{v['hi']:+.6f}|")
    lines+=['','## Top 11 至 30 扩召回收益','','|图谱|基线|方法|原 Filtered 11–30 查询|进入前十|原前十跌出|Hits@10 净变化|','|---|---|---|---:|---:|---:|---:|']
    expansion_rows=[]
    for s in summaries:
        for m,v in s['expansion'].items():
            delta=s['metrics'][m]['Hits10']-s['metrics']['M0']['Hits10']
            expansion_rows.append(dict(dataset=s['dataset'],family=s['family'],method=m,**v,Hits10_delta=delta))
            lines.append(f"|{s['dataset']}|{s['family']}|{m}|{v['eligible']}|{v['promoted']}|{v['demoted']}|{delta:+.6f}|")
    lines+=['','候选池的排名与全图 Filtered 排名不同，可能存在原 Filtered 排名≤10但候选池排名>10的目标。主表 Hits@10 差值由完整排名直接计算，不能只用“11–30 提拔次数减去跌出次数”替代。','','## C D E3a F 的结构排序诊断','','|图谱|基线|方法|Top10 条件结构 MRR|池内11–30答案|严格超过Top10负例|','|---|---|---|---:|---:|---:|']
    for s in summaries:
        for m,v in s['structural_diagnostics'].items():
            val=v['top10_conditional_structural_MRR']
            lines.append(f"|{s['dataset']}|{s['family']}|{m}|{val if val is not None else 'NA'}|{v['rank11_30_answers']}|{v['strict_expansion_potential']}|")
    lines+=['','这些诊断使用各自新候选池和官方测试集，与历史未过滤候选、合并 valid+test 的条件指标存在口径差异。它们衡量结构证据排序，不是全图 Filtered MRR。','','## LLM 触发与正负翻转','','|图谱|基线|触发查询|触发率|真实响应数|拒答|拒答率|格式回退|正翻转|负翻转|总Token|平均延迟秒|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        v=s['llm']
        if v.get('skipped'):
            lines.append(f"|{s['dataset']}|{s['family']}|跳过||||||||")
        else:
            lines.append(f"|{s['dataset']}|{s['family']}|{v['eligible']}|{v['trigger_rate']:.2%}|{v['calls']}|{v['abstain']}|{v.get('abstain_rate',0):.2%}|{v['parse_failures']}|{v['positive_flips']}|{v['negative_flips']}|{v['total_tokens']}|{v.get('mean_latency_seconds',0):.3f}|")
    lines+=['','Token 与延迟按当前运行使用的真实响应汇总；命中历史响应缓存不增加网络请求。校准和连通性调用单独记录，未混入测试调用计数。','','## 实际图算子核验与内存峰值','']
    for key in datasets:
        for family in families:
            work=run/key/family;path=work/'operator_audit.json'
            if path.exists():
                audit=read(path)
                differences=[x['max_forward_difference_vs_E3a'] for x in audit]
                lines.append(f"- {key}/{family}：核验 {len(audit)} 个查询关系，全部前后向行归一化断言通过；与 E3a 的前向概率最大差异为 {max(differences,default=0):.6g}；发送端行常数抵消最大误差为 {max((x['sender_cancellation_error'] for x in audit),default=0):.3g}。")
            completion=read(work/'completion.json')
            if 'peak_rss_bytes' in completion:lines.append(f"  进程峰值 RSS：{completion['peak_rss_bytes']/1e9:.3f} GB，限制为 1.800 GB。")
    lines+=['','## 候选协议差异与数据完整性','']
    for s in summaries:
        v=s['filtered_candidate_difference']
        lines.append(f"- {s['dataset']} / {s['family']}：{v['candidate_pools_differ']}/{v['queries']} 条查询的候选池与标准先 Filtered 后取 Top-30 不同，合计缺少 {v['filtered_top30_missing_entities']} 个相应候选。")
    lines+=['','## 真实案例与文本覆盖','']
    for key in datasets:
        for family in families:
            work=run/key/family
            audit=work/'verbalizer_audit.json'
            if audit.exists():
                a=read(audit);lines.append(f"{key} / {family}：本地文本映射覆盖 {a['mapped']}/{a['entities']}。缺失时保留原始 ID 或 NELL 概念前缀，不编造实体名。")
            for case in sorted((work/'cases').glob('*.json'))[:4]:
                c=read(case);decision=c['response']['decision']
                lines.append(f"\n- {key}/{family}/{case.stem}：结构排名 {c['structural_filtered_rank']} → 最终排名 {c['final_filtered_rank']}；abstain={decision['abstain']}；模型说明：{decision['reasoning']}。完整原始路径及边权见 `{case.relative_to(run)}`。")
    lines+=['','## 结论与限制','']
    if smoke:
        lines.append('本文件仅证明小样本执行链路可运行，不构成方法有效性结论。没有用冒烟测试选择正式超参数。')
    else:
        for s in summaries:
            base=s['metrics']['M0']['MRR'];r4=s['metrics']['R4']['MRR']
            lines.append(f"- {s['dataset']} / {s['family']}：R4 相对 M0 的 MRR 差值为 {r4-base:+.6f}；R4 点估计{'提高' if r4>base else '降低' if r4<base else '持平'}。显著性以配对区间为准。")
    lines+=['','目标图谱已有历史研究暴露；本次新验证集选择是目标域开发协议，不能称为无目标选参的 P0 或首次盲测。保护机制不保证零退化。只运行一个固定模型种子；置信区间描述查询抽样不确定性，不涵盖训练种子和 LLM 服务版本变化。', '',
           'API 参数依据 [DeepSeek 官方 Chat Completions 文档](https://api-docs.deepseek.com/api/create-chat-completion/)；模型为 deepseek-flash，关闭 thinking，要求 JSON 输出。API 凭证只从环境或原始本地文档读入进程，不写入源码、报告或运行配置。']
    text='\n'.join(lines)+'\n'
    (run/'report_route_a_rerank.md').write_text(text,encoding='utf-8')
    if not smoke: (ROOT/'report_route_a_rerank.md').write_text(text,encoding='utf-8')
    for name,data in [('main_metrics.csv',rows),('paired_bootstrap.csv',interval_rows),('expansion_diagnostics.csv',expansion_rows)]:
        if data:
            with (run/name).open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
