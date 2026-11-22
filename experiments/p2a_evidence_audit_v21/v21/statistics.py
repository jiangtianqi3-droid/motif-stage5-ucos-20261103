"""Paired fact-cluster bootstrap, random-repeat aggregation, plots and Chinese report."""
from collections import defaultdict
from statistics import mean, pstdev
from pathlib import Path

import numpy as np

from v21.common import ROOT, read, dump, atomic, write_csv, verify
from v21.audit import AGGREGATORS


def confidence(values, fact_ids, cfg, tag):
    # One resample unit is the original test fact, containing both prediction directions.
    unique=sorted(set(fact_ids))
    grouped={f:[] for f in unique}
    for f,v in zip(fact_ids,values):
        grouped[f].append(float(v))
    n=cfg['audit']['bootstrap_repeats']
    seed=cfg['experiment']['seed']+sum(map(ord,tag))
    rng=np.random.default_rng(seed)
    # Cluster means equal the query mean for the paired-direction sample used here.
    x=np.array([mean(grouped[f]) for f in unique])
    blocks=[]
    for at in range(0,n,1000):
        ix=rng.integers(0,len(x),size=(min(1000,n-at),len(x)))
        blocks.append(x[ix].mean(axis=1))
    boot=np.concatenate(blocks)
    alpha=(1-cfg['audit']['bootstrap_confidence'])/2
    return np.quantile(boot,[alpha,1-alpha]).tolist()


def describe(values,fact_ids,cfg,tag):
    return dict(mean=mean(values),std=pstdev(values),ci95=confidence(values,fact_ids,cfg,tag))


def analyze(directory,records,cfg):
    facts=[r['fact_index'] for r in records]
    baseline=np.array([r['full']['rr'] for r in records])
    sufficient=[]
    necessity=[]
    summary_s=[]
    summary_n=[]
    random_details=[]
    groups=cfg['retrieval_groups']
    keep_names=['G_full','G_local',*groups,*['Random_'+g for g in groups]]
    rr_by_group={}
    nec_by_group={}
    for name in keep_names:
        random=name.startswith('Random_')
        group=name[7:] if random else name
        rows=[]
        for record in records:
            if random:
                rep=record['random_sufficiency'][group]
                value={k:mean(r[k] for r in rep) for k in ('rr','rank','hits1','hits10','top1_agreement','top10_overlap','unique_facts','directed_messages')}
                value['repeat_rr_std']=pstdev(r['rr'] for r in rep)
            else:
                value=dict(record['sufficiency'][group],repeat_rr_std=0.)
            row=dict(query_id=record['query_id'],fact_index=record['fact_index'],direction=record['direction'],
                     method=name,**{k:value[k] for k in ('rr','rank','hits1','hits10','top1_agreement','top10_overlap',
                     'unique_facts','directed_messages','repeat_rr_std')},
                     baseline_rank=record['full']['rank'],baseline_rr=record['full']['rr'],
                     rank_change=record['full']['rank']-value['rank'],rr_change=value['rr']-record['full']['rr'],
                     sufficiency_loss=record['full']['rr']-value['rr'])
            rows.append(row)
        sufficient+=rows
        values=np.array([r['rr'] for r in rows])
        rr_by_group[name]=values
        loss=baseline-values
        metrics=describe(loss.tolist(),facts,cfg,'suff_'+name)
        base_mrr=float(baseline.mean())
        result=dict(method=name,queries=len(records),unique_facts=mean(r['unique_facts'] for r in rows),
             directed_messages=mean(r['directed_messages'] for r in rows),MRR=float(values.mean()),
             MRR_std=pstdev(values),MRR_ci95=confidence(values,facts,cfg,'mrr_'+name),
             Hits1=mean(r['hits1'] for r in rows),Hits10=mean(r['hits10'] for r in rows),
             mean_rank=mean(r['rank'] for r in rows),top1_agreement=mean(r['top1_agreement'] for r in rows),
             top10_overlap=mean(r['top10_overlap'] for r in rows),sufficiency_loss=metrics['mean'],
             sufficiency_loss_std=metrics['std'],sufficiency_loss_ci95=metrics['ci95'],
             recovery=float(values.mean()/base_mrr) if base_mrr else None,baseline_MRR=base_mrr,
             random_rr_std=mean(r['repeat_rr_std'] for r in rows))
        if name in groups:
            random_rr=np.array([mean(x['rr'] for x in r['random_sufficiency'][name]) for r in records])
            gap=describe((values-random_rr).tolist(),facts,cfg,'suff_random_gap_'+name)
            result.update(sufficiency_vs_random=gap['mean'],sufficiency_vs_random_ci95=gap['ci95'])
        summary_s.append(result)
    for group in groups:
        rows=[]
        for record in records:
            item=record['necessity'][group]
            random=record['random_necessity'][group]
            random_comp=np.array([r['comp_rr'] for r in random])
            sampled=record['random_details']
            relax={key:sum(t['relaxations'].get(key,0) for t in sampled if t['group']==group)
                   for key in {k for t in sampled if t['group']==group for k in t['relaxations']}}
            result=dict(query_id=record['query_id'],fact_index=record['fact_index'],direction=record['direction'],
                method=group,baseline_rr=record['full']['rr'],rank_removed=item['rank'],rr_removed=item['rr'],
                selected_comp=item['comp_rr'],random_comp_mean=float(random_comp.mean()),
                random_comp_std=float(random_comp.std()),necessity_increment=item['comp_rr']-float(random_comp.mean()),
                unique_facts=item['unique_facts'],removed_messages=item['removed_messages'],
                random_repeats=len(random),relaxations=relax)
            rows.append(result)
        necessity+=rows
        inc=np.array([r['necessity_increment'] for r in rows])
        nec_by_group[group]=inc
        delta=describe(inc.tolist(),facts,cfg,'necessity_'+group)
        summary_n.append(dict(method=group,queries=len(records),unique_facts=mean(r['unique_facts'] for r in rows),
            removed_messages=mean(r['removed_messages'] for r in rows),
            selected_comp=mean(r['selected_comp'] for r in rows),
            selected_comp_ci95=confidence([r['selected_comp'] for r in rows],facts,cfg,'selected_drop_'+group),
            random_comp=mean(r['random_comp_mean'] for r in rows),
            random_comp_ci95=confidence([r['random_comp_mean'] for r in rows],facts,cfg,'random_drop_'+group),
            random_within_query_std=mean(r['random_comp_std'] for r in rows),
            necessity_increment=delta['mean'],necessity_increment_std=delta['std'],
            necessity_increment_ci95=delta['ci95'],
            positive_queries=sum(r['necessity_increment']>0 for r in rows),
            negative_queries=sum(r['necessity_increment']<0 for r in rows)))
    for record in records:
        for t in record['random_details']:
            group=t['group'];rep=t['repeat']
            outcome=record['random_necessity'][group][rep]
            random_details.append(dict(t,rank_removed=outcome['rank'],rr_removed=outcome['rr'],
                                      comp_rr=outcome['comp_rr']))
    write_csv(directory/'sufficiency/sufficiency_metrics.csv',summary_s)
    write_csv(directory/'sufficiency/sufficiency_per_query.csv',sufficient)
    write_csv(directory/'necessity/necessity_metrics.csv',summary_n)
    write_csv(directory/'necessity/necessity_per_query.csv',necessity)
    write_csv(directory/'necessity/random_deletion_details.csv',random_details)
    paired=[]
    for ix,a in enumerate(groups):
        for b in groups[ix+1:]:
            s=describe((rr_by_group[a]-rr_by_group[b]).tolist(),facts,cfg,'pair_suff_'+a+b)
            n=describe((nec_by_group[a]-nec_by_group[b]).tolist(),facts,cfg,'pair_nec_'+a+b)
            paired.append(dict(method_a=a,method_b=b,sufficiency_MRR_difference=s['mean'],
                     sufficiency_MRR_ci95=s['ci95'],necessity_increment_difference=n['mean'],
                     necessity_increment_ci95=n['ci95']))
    write_csv(directory/'paired_method_differences.csv',paired)
    summary=dict(sufficiency=summary_s,necessity=summary_n,paired=paired,
            bootstrap_repeats=cfg['audit']['bootstrap_repeats'],confidence=cfg['audit']['bootstrap_confidence'],
            resampling='original-fact clusters, both directions, aggregate 10 draws within query first',
            queries=len(records),independent_test_facts=len(set(facts)))
    dump(directory/'statistics.json',summary)
    return summary


def figures(directory,stats,trace,aggregation,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'axes.axisbelow':True,'grid.alpha':.20,'svg.fonttype':'none'})
    folder=directory/'figures'
    folder.mkdir(parents=True,exist_ok=True)
    sources=[]
    def save(fig,name,source):
        fig.tight_layout(rect=(0,.04,1,1))
        fig.text(.01,.005,'Source: '+source,fontsize=8)
        for ext in ('png','svg'):
            fig.savefig(folder/(name+'.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)
        sources.append(dict(figure=name,data=source))
    changes=[r for r in trace if r['method_a']=='R2']
    vals=[r['path_participation'] for r in changes if r['path_participation'] is not None]
    fig,ax=plt.subplots(figsize=(7,3.6))
    ax.hist(vals,bins=np.linspace(0,1,11),color='#0072B2',edgecolor='white')
    ax.set(xlabel='Added canonical facts participating in candidate paths (fraction)',ylabel='Queries',
           title='R3 added-fact path participation')
    ax.set_xlim(-.02,1.02)
    ax.text(.98,.95,f'No added facts (NA): {sum(r["path_participation"] is None for r in changes)}',
            ha='right',va='top',transform=ax.transAxes)
    save(fig,'01_path_participation','propagation/t0_t4_audit.csv')
    fig,ax=plt.subplots(figsize=(6.5,3.6))
    colors=['#999999','#E69F00','#56B4E9','#009E73','#D55E00']
    bottom=np.zeros(2)
    for k in range(5):
        values=np.array([mean(r['category']==f'T{k}' for r in trace if r['method_a']==a) for a in ('R2','R3')])
        ax.bar(['R2 to R3','R3 to R4'],values,bottom=bottom,color=colors[k],label=f'T{k}',hatch=['','/','x','.','+'][k])
        bottom+=values
    ax.set(ylabel='Fraction of queries',title='Evidence-to-score propagation',ylim=(0,1.08))
    ax.legend(ncol=5,fontsize=8,loc='upper center')
    save(fig,'02_t0_t4','propagation/t0_t4_audit.csv')
    fig,ax=plt.subplots(figsize=(7,3.6))
    x=np.arange(4)
    for off,method,col,hatch in ((-.18,'R2','#0072B2',''),(.18,'R3','#E69F00','/')):
        v=[next(r['MRR'] for r in aggregation if r['method']==method and r['aggregator']==a) for a in AGGREGATORS]
        ax.bar(x+off,v,.34,label=method,color=col,hatch=hatch)
    ax.set_xticks(x,['Max','Sum','LSE','Normalized LSE'])
    ax.set(ylabel='Filtered MRR',title='Frozen original-path aggregation')
    ax.set_ylim(bottom=0)
    ax.legend()
    save(fig,'03_aggregation','aggregation/aggregation_metrics.csv')
    s={r['method']:r for r in stats['sufficiency']}
    n={r['method']:r for r in stats['necessity']}
    groups=cfg['retrieval_groups']
    x=np.arange(len(groups))
    for name,key,title,ylabel in (
        ('04_recovery','recovery','Full-backbone performance recovery','MRR / full-graph MRR'),
        ('05_sufficiency_loss','sufficiency_loss','Sufficiency loss with full-graph D* fixed','Full MRR minus subgraph MRR')):
        fig,ax=plt.subplots(figsize=(7,3.8))
        for off,prefix,label_,col,hatch in ((-.18,'','Retrieved','#0072B2',''),(.18,'Random_','Local matched random','#E69F00','/')):
            vals=[s[prefix+g][key] for g in groups]
            ax.bar(x+off,vals,.34,label=label_,color=col,hatch=hatch)
        ax.set_xticks(x,groups)
        ax.set(ylabel=ylabel,title=title)
        ax.axhline(1 if key=='recovery' else 0,color='#333333',linestyle='--',linewidth=1)
        ax.set_ylim(bottom=min(0,ax.get_ylim()[0]))
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        save(fig,name,'sufficiency/sufficiency_metrics.csv')
    fig,ax=plt.subplots(figsize=(7,3.8))
    for off,key,label_,col in ((-.18,'selected_comp','Retrieve then delete','#0072B2'),(.18,'random_comp','Local matched delete','#E69F00')):
        vals=np.array([n[g][key] for g in groups])
        cis=np.array([n[g][key+'_ci95'] for g in groups])
        err=np.maximum(0,np.stack((vals-cis[:,0],cis[:,1]-vals)))
        ax.bar(x+off,vals,.34,label=label_,color=col,yerr=err,capsize=3)
    ax.set_xticks(x,groups)
    ax.set(title='Necessity versus local matched deletion',ylabel='Full MRR minus deletion MRR')
    ax.axhline(0,color='#333333',linewidth=1)
    ax.legend(fontsize=8)
    save(fig,'06_deletion_drop','necessity/necessity_metrics.csv; 95% fact-cluster bootstrap')
    fig,ax=plt.subplots(figsize=(6,4))
    for i,g in enumerate(groups):
        xx=s[g]['sufficiency_loss']; yy=n[g]['necessity_increment']
        ci=n[g]['necessity_increment_ci95']
        ax.errorbar(xx,yy,yerr=[[max(0,yy-ci[0])],[max(0,ci[1]-yy)]],fmt='o',capsize=3,
                    color=plt.get_cmap('tab10')(i))
        ax.annotate(g,(xx,yy),xytext=(5,5),textcoords='offset points')
    ax.axhline(0,color='#444444',linestyle='--')
    ax.set(xlabel='Sufficiency loss (lower is better)',ylabel='Necessity gain over matched deletion',
           title='Sufficiency and necessity of retrieved facts')
    save(fig,'07_fidelity_tradeoff','sufficiency/sufficiency_metrics.csv; necessity/necessity_metrics.csv')
    write_csv(folder/'figure_sources.csv',sources)


def fnum(x):
    return 'NA' if x is None else f'{x:.4f}'


def interval(ci):
    return '['+', '.join(f'{x:+.4f}' for x in ci)+']'


def make_report(directory,stats,records,trace,aggregation,cfg,state,smoke_time=None):
    provenance=read(directory/'provenance.json')
    proof=read(directory/'freeze_proof.json')
    n={x['method']:x for x in stats['necessity']}
    suff={x['method']:x for x in stats['sufficiency']}
    changes=[r for r in trace if r['method_a']=='R2']
    has_advantage=[g for g in cfg['retrieval_groups']
        if suff[g]['sufficiency_vs_random_ci95'][0]>0 or n[g]['necessity_increment_ci95'][0]>0]
    peripheral=any(r['added_facts']>0 and r['candidate_path_added_facts']==0 for r in changes)
    decision='Go' if has_advantage else 'Conditional Go' if peripheral else 'No-Go'
    rows=['# P2-A Stage 1 · V2.1 结构证据审计','',
       f"运行状态 {state['state']}，查询 {len(records)} 条，独立事实 {stats['independent_test_facts']} 条。API 请求 0 次。",
       '## 1. 背景','',
       'P1 中 R2–R4 选中的事实集合不同，但候选证据排序相同。本阶段逐级追踪事实、候选路径、原路径效用、分数及排名，并在冻结 B2+D* 上测试证据充分性与必要性。',
       '## 2. 环境与冻结配置','',
       f"CPU {provenance['cpu']}。GPU {provenance['gpu']}，显存 {provenance['gpu_bytes']/1024**3:.2f} GiB。"
       f"Python {provenance['python']}，PyTorch {provenance['torch']}，CUDA {provenance['cuda']}。",
       f"Git {provenance['git_commit']}。B2 检查点 SHA256 为 `{provenance['checkpoint_sha256']}`。"
       f"模型参数加载及结束哈希均为 `{proof['parameter_sha_before']}`。",
       f"全图关系表示文件哈希 `{proof['relation_embeddings_sha256']}`。D* 文件哈希 `{proof['dstar_cache_sha256']}`。",
       f"实际张量峰值 {proof['peak_allocated']/1024**2:.1f} MiB，CUDA 保留峰值 {proof['peak_reserved']/1024**2:.1f} MiB。"
       f"反事实期间关系编码器调用 {proof['counterfactual_relation_encoder_calls']} 次，D* 缓存逐次校验固定。"
       f"批次调整记录 {proof['batch_adjustments']}。",
       '## 3. 资产与协议','',
       f"P1 为 {provenance['p1_commit']}，运行 {provenance['p1_run_id']}。全部 R0–R5 实际记录已找到。"
       '完整候选与原批次全图评分逐实体的未过滤倒数秩、原过滤名次一致。B0.npz 的真实字段和形状保存在 historical_schema.json。'
       '证据测试是新执行的模型推理，原始资产核对见 asset_audit.md。',
       'FBIngram_25 沿用原测试划分、离线过滤集合、原始实体与关系编号。样本在 P1 固定查询资产内按种子 20261122 选择 50 条事实及两个方向。'
       '事实 ID 按正向三元组规范化，同一事实的逆向和重复记录计一次预算。路径审计使用原选择方向。保真度输入则对唯一事实恢复全图中的逆向与重复消息记录。',
       f"D* 原始全图均值 {proof['dstar_stats']['raw_edge_mean']:.8f}，均值锚定后 {proof['dstar_stats']['effective_edge_mean']:.8f}。"
       '历史 Triton 的发送端是 edge_index[1]。所有图复用全图 a/b/z、原 float32 锚点与固定查询 alpha，保持完整 4097 实体空间。'
       '关系编码器始终看到原全图关系超图，子图仅替换第二阶段 edge_index/edge_type。',
       '历史 Top-12 是未过滤生成；一致性指标以全实体未过滤 Top-1、Top-10 比较。最终指标用原过滤排名和悲观并列规则。',
       '## 4. R2–R4 影响传播','',
       '| 对比 | T0 | T1 | T2 | T3 | T4 | 新增唯一事实 | 路径参与事实 | 平均路径参与率 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a,b in (('R2','R3'),('R3','R4')):
        rr=[r for r in trace if r['method_a']==a]
        vals=[r['path_participation'] for r in rr if r['path_participation'] is not None]
        vals_text=fnum(mean(vals)) if vals else 'NA'
        rows.append(f"| {a}→{b} | "+' | '.join(f"{sum(r['category']==f'T{k}' for r in rr)} ({mean(r['category']==f'T{k}' for r in rr):.1%})" for k in range(5))+
                    f" | {sum(r['added_facts'] for r in rr)} | {sum(r['candidate_path_added_facts'] for r in rr)} | {vals_text} |")
    rows+=['','新增事实为零的查询路径参与率记录 NA。路径包含全部三跳内有向简单路径，Gq 中潜在参与率另列。'
           '“新增实体高阶模体实例”缺少历史对应定义，报告为 NA，本阶段未引入新接地算法。',
           '代码审计发现 P1 retrieval.py 的 choose_pcst 先保护候选骨干，将其中全部节点收缩到查询根，'
           '再跳过收缩后两端相同的边，最后按根连通性添加扩展。这一结构限制会偏向骨干外围的事实，'
           '上述路径参与数据用于判断该偏置在本批查询中是否发生。原 Max 评分的损失需在确实出现新候选路径后再判断。',
           '每组输入和再次执行的输入哈希、节点、关系、路径与效用保存于 input_consistency.csv / scoring_inputs.json。'
           '本阶段 T4 定义为候选分数或排名发生变化，与此前 P2 的分类规则分别保存。',
           '## 5. 固定路径聚合','',
           f"四组聚合共用 P1 原枚举路径和效用，温度固定 {cfg['audit']['aggregation_temperature']}，空路径统一回退 0。"
           'Sum 和非归一化 LSE 的路径数量偏置通过分数与路径数、度数相关系数检查。',
           '| 组 | 聚合 | MRR | H@1 | H@10 | 平均名次 | Top-1 改变率 | R2/R3 顺序一致 | 路径数相关 | 出度相关 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregation:
        rows.append(f"| {r['method']} | {r['aggregator']} | {r['MRR']:.4f} | {r['Hits1']:.4f} | {r['Hits10']:.4f} | "
                    f"{r['mean_rank']:.2f} | {r['top1_change_rate']:.4f} | {r['r2_r3_order_agreement']:.4f} | "
                    f"{fnum(r['spearman_paths'])} | {fnum(r['spearman_outdegree'])} |")
    rows+=['','## 6. 固定主干充分性','',
           '| 方法 | 唯一事实 | 消息边 | MRR | H@1 | H@10 | 平均名次 | Top-1 一致 | Top-10 重合 | 充分性损失 ±95% CI | 恢复比例 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in stats['sufficiency']:
        rows.append(f"| {r['method']} | {r['unique_facts']:.1f} | {r['directed_messages']:.1f} | {r['MRR']:.4f} | "
                    f"{r['Hits1']:.4f} | {r['Hits10']:.4f} | {r['mean_rank']:.2f} | {r['top1_agreement']:.4f} | "
                    f"{r['top10_overlap']:.4f} | {r['sufficiency_loss']:+.4f} {interval(r['sufficiency_loss_ci95'])} | {fnum(r['recovery'])} |")
    rows+=['','## 7. 局部匹配随机删除与必要性','',
           '随机事实从逆向闭包后的同一个 Gq 的唯一事实空间抽样，先排除检索事实。'
           '分层首先精确匹配关系与头部跳数，然后匹配跳数与关系全图频次三分位，依次放宽为同跳数、同频次、局部任意。'
           '仅在局部剩余预算不足时允许与检索事实重叠，并逐条记录。跳数根据局部图与头实体最短距离，超过三跳或不可达单列第 4 桶。'
           '每查询、每检索组 10 个固定种子控制。充分性和必要性共用相同随机样本。',
           '| 方法 | 删除后跌落 | 随机删除跌落 | 必要性增量 | 95% CI | 查询内随机标准差 | 正/负查询 |',
           '|---|---:|---:|---:|---|---:|---:|']
    for r in stats['necessity']:
        rows.append(f"| {r['method']} | {r['selected_comp']:+.4f} | {r['random_comp']:+.4f} | "
                    f"{r['necessity_increment']:+.4f} | {interval(r['necessity_increment_ci95'])} | "
                    f"{r['random_within_query_std']:.4f} | {r['positive_queries']}/{r['negative_queries']} |")
    rows+=['','## 8. 综合表与配对统计','',
           f"Bootstrap {cfg['audit']['bootstrap_repeats']} 次、95% 区间。先合并每查询 10 次随机结果，再以原始事实为聚类单元同时重采两个方向。均值、查询级标准差和方法间配对差异见 CSV。",
           '| 方法 | 事实数 | 全部选中事实路径参与率 | 子图 MRR | 充分性损失 | 恢复比例 | 必要性增量 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    inputs=read(directory/'propagation/scoring_inputs.json')
    # Ratio for every retrieval group based on exact selected paths, with canonical identity.
    from v21.graph import FactIndex
    from p2.assets import Target,p1_root
    view=Target(cfg['experiment']['dataset'],p1_root(),directory)
    index=FactIndex(view.graph)
    for g in cfg['retrieval_groups']:
        ratios=[]
        for item in inputs:
            if item['method']!=g:continue
            chosen=set(item['scoring_input']['canonical_fact_ids'])
            from p2.features import all_paths
            payload=item['scoring_input']
            q=next(r for r in records if r['query_id']==item['query_id'])
            i,d=q['fact_index'],q['direction']
            job,_=view.job(i,d)
            paths=all_paths(view.graph,job['head'],job['candidates'],payload['selected_ids'])
            used={int(index.directed_to_id[e]) for pp in paths.values() for p in pp for e in p}
            ratios.append(len(chosen&used)/len(chosen) if chosen else None)
        valid=[r for r in ratios if r is not None]
        s=suff[g];n0=n[g]
        rows.append(f"| {g} | {s['unique_facts']:.1f} | {fnum(mean(valid) if valid else None)} | {s['MRR']:.4f} | "
                    f"{s['sufficiency_loss']:+.4f} | {fnum(s['recovery'])} | {n0['necessity_increment']:+.4f} |")
        s=suff['Random_'+g]
        rows.append(f"| 匹配随机({g}) | {s['unique_facts']:.1f} | NA | {s['MRR']:.4f} | {s['sufficiency_loss']:+.4f} | {fnum(s['recovery'])} | NA |")
    rows+=['','## 9. 真实失败与变化案例','']
    criteria=[
        ('新增事实未进入候选路径',next((r for r in trace if r['method_a']=='R2' and r['added_facts']>=10 and r['candidate_path_added_facts']==0),None)),
        ('候选评分或顺序未变化',next((r for r in trace if r['category']=='T1'),None))]
    for label,r in criteria:
        if r:
            rows.append(f"- {label}见 {r['query_id']}，新增唯一事实 {r['added_facts']}，路径参与 {r['candidate_path_added_facts']}，"
                        f"特征差 {r['feature_delta']:.6f}，分数差 {r['score_delta']:.6f}，目标名次 {r['target_rank_before']}→{r['target_rank_after']}。")
        else:rows.append(f'- 当前样本没有“{label}”案例。')
    large=max(((rec,g) for rec in records for g in cfg['retrieval_groups']),
              key=lambda pair:pair[0]['necessity'][pair[1]]['comp_rr'])
    rec,g=large
    rows.append(f"- 最大检索删除跌落见 {rec['query_id']}、{g}，目标原名次 {rec['full']['rank']}，删除后 "
                f"{rec['necessity'][g]['rank']}，RR 跌落 {rec['necessity'][g]['comp_rr']:+.4f}。")
    similar=min(((rec,g) for rec in records for g in cfg['retrieval_groups']),
                key=lambda pair:abs(pair[0]['necessity'][pair[1]]['comp_rr']-mean(
                    x['comp_rr'] for x in pair[0]['random_necessity'][pair[1]])))
    rec,g=similar
    rows.append(f"- 删除影响相近见 {rec['query_id']}、{g}，检索删除跌落 {rec['necessity'][g]['comp_rr']:+.4f}，"
                f"随机平均 {mean(x['comp_rr'] for x in rec['random_necessity'][g]):+.4f}。原事实 ID 和随机种子见逐查询检查点及 random_deletion_details.csv。")
    rows+=['','## 10. Stage 1 Go / No-Go','',
           f"当前判断为 **{decision}**。超出局部随机对照且 95% 区间下界大于零的方法为 {has_advantage or '无'}。"
           '判断同时保留点估计和统计不确定性。若外围扩展存在明确偏置，允许按 Conditional Go 修正问题后再验证。',
           '## 11. 后续与可复现性','',
           '首先根据单图有限样本的充分性和必要性确定保留哪些证据结构。外围奖励偏置和路径聚合按本次传导结果处理，后续研究选择应基于对应的反事实增量。'
           '本阶段仅提供冻结模型评价，未训练、未实现实体级模体接地。'
           '自动入口 scripts/run_all.py，参数、代码哈希、查询列表和模型证明均已保存。']
    if smoke_time:
        rows+=['',f"10 条查询冒烟实际耗时 {smoke_time['seconds']:.1f} 秒，外推 100 条剩余耗时约 "
                f"{smoke_time['estimated_remaining_seconds']:.0f} 秒。该值仅按实测吞吐量外推，受初始化、缓存和负载影响。"]
    for fig,title in [('01_path_participation','新增事实路径参与率'),('02_t0_t4','影响传播'),('03_aggregation','路径聚合'),
         ('04_recovery','恢复比例'),('05_sufficiency_loss','充分性损失'),('06_deletion_drop','必要性'),('07_fidelity_tradeoff','二维保真度')]:
        rows.extend(['',f'![{title}](outputs/{directory.name}/figures/{fig}.png)'])
    atomic(ROOT/'report_p2a.md','\n'.join(rows)+'\n')
    atomic(directory/'aggregation/aggregation_analysis.md',
           '# 固定效用聚合诊断\n\n四种聚合读取同一 P1 路径和效用，温度固定。R2/R3 原始排名一致率和路径数量、度数相关见 aggregation_metrics.csv。'
           '新增实体级高阶模体计数为 NA，完整有向候选路径单独精确枚举。\n')
    return decision
