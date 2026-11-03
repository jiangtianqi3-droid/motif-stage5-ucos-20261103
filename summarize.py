import csv,json,hashlib
from pathlib import Path
import numpy as np
from common import ROOT,OLD,SOURCES,dump,sha,log
from evaluate import metrics
FOCUS=['B1-short','T-4D','T-4D-alt','T-4D+HubCal','U-Cos','R-Cos-Shuffle']
FAMILIES=['B0','B1','B2']+FOCUS
def csvout(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/'results'/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def bootstrap_matrix(diff,seed):
    n,c=diff.shape
    if n==0:return [(None,None,None)]*c
    if np.max(np.abs(diff),initial=0)==0:return [(0.,0.,0.)]*c
    rng=np.random.default_rng(seed);out=np.empty((5000,c));prob=np.full(n,1/n)
    for at in range(0,5000,128):
        count=rng.multinomial(n,prob,size=min(128,5000-at)).astype(float)
        out[at:at+len(count)]=(count@diff)/n
    lo,hi=np.quantile(out,[.025,.975],axis=0)
    return [(float(diff[:,i].mean()),float(lo[i]),float(hi[i])) for i in range(c)]
def paired(diff,key):
    return bootstrap_matrix(diff,int(hashlib.sha256(key.encode()).hexdigest()[:8],16)+20261105)
def comparisons():
    rows=[(f,b) for f in FOCUS for b in ['B0','B1','B2','B1-short','T-4D'] if f!=b]
    rows+=[('U-Cos','R-Cos-Shuffle'),('U-Cos','T-4D+HubCal')]
    return rows
def mdtable(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in rows)
def ci(row):
    if row['n']==0:return 'NA（空组）'
    return f"{row['delta']:+.6f} [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]"
def main():
    frozen=json.loads((ROOT/'frozen.json').read_text());choices=json.loads((ROOT/'analysis/source_choices.json').read_text());formal=frozen['formal']
    p0=[];pergraph=[];strata=[];intervals=[];bygraph={};ranks_bygraph={};cmps=comparisons()
    for target in frozen['targets']:
        key=target['key'];values={};hitvalues={};seedmetrics={};triple=None
        for family in FAMILIES:
            seeds=[1024] if family=='B0' else [1024,2026,3407];vv=[];sm=[];hh={k:[] for k in [1,3,10]}
            for seed in seeds:
                tag='B0' if family=='B0' else family+'_s'+str(seed)
                path=ROOT/'evaluation'/f'{key}__{tag}.npz';a=np.load(path);rr=a['ranks'];tt=a['triples']
                if triple is None:triple=tt
                assert np.array_equal(tt,triple)
                assert rr.shape==(len(tt),2) and (rr>=1).all()
                meta=json.loads(path.with_suffix('.json').read_text());recomputed=metrics(rr);assert abs(recomputed['mrr']-meta['mrr'])<1e-12
                p0.append(dict(key=key,group=target['group'],family=family,seed=seed,tag=tag,checkpoint_sha256=meta.get('checkpoint_sha256'),selected_step=meta.get('selected_step'),lambda_hub=meta.get('lambda_hub',0),**recomputed))
                vv.append((1/rr.astype(float)).mean(1));sm.append(recomputed)
                for k in hh:hh[k].append((rr<=k).mean(1))
            values[family]=np.stack(vv).mean(0);seedmetrics[family]=sm
            hitvalues[family]={k:np.stack(v).mean(0) for k,v in hh.items()}
            pergraph.append(dict(key=key,group=target['group'],family=family,mrr=float(values[family].mean()),seed_sd=float(np.std([r['mrr'] for r in sm],ddof=1)) if len(sm)>1 else 0.,seeds=len(seeds),**{'hits_at_'+str(k):float(np.mean([r['hits_at_'+str(k)] for r in sm])) for k in [1,3,10]}))
        qm=np.load(ROOT/'analysis'/f'query_masks_{key}.npz');assert qm['query_order_sha256'].item()==target['query_sha256']
        groups={'all':np.ones(len(triple),dtype=bool)}
        groups.update({k:qm[k] for k in qm.files if qm[k].dtype==bool})
        for qgroup,mask in groups.items():
            assert mask.shape==(len(triple),)
            n=int(mask.sum())
            for family in FAMILIES:strata.append(dict(key=key,group=target['group'],query_group=qgroup,family=family,n=n,mrr=float(values[family][mask].mean()) if n else None,**{'hits_at_'+str(k):float(v[mask].mean()) if n else None for k,v in hitvalues[family].items()}))
            diff=np.stack([values[a][mask]-values[b][mask] for a,b in cmps],1)
            cis=paired(diff,key+qgroup)
            for (a,b),(dd,lo,hi) in zip(cmps,cis):intervals.append(dict(key=key,group=target['group'],query_group=qgroup,method=a,baseline=b,n=n,delta=dd,ci_low=lo,ci_high=hi,bootstrap_replicates=5000,interval_scope='query-conditional paired original triple; seed variation reported separately'))
        bygraph[key]=values
        log('summarized_graph',key=key)
    macro=[]
    for group in ['locked','historical','external']:
        keys=[t['key'] for t in frozen['targets'] if t['group']==group]
        for family in FAMILIES:
            seeds=[1024] if family=='B0' else [1024,2026,3407]
            seedmacro=[np.mean([r['mrr'] for r in p0 if r['key'] in keys and r['family']==family and r['seed']==s]) for s in seeds]
            mean=float(np.mean(seedmacro));macro.append(dict(group=group,family=family,graphs=len(keys),mrr=mean,seed_sd=float(np.std(seedmacro,ddof=1)) if len(seeds)>1 else 0.,**{'delta_'+b:mean-float(np.mean([v[b].mean() for k,v in bygraph.items() if k in keys])) for b in ['B0','B1','B2','B1-short']}))
    source=[];source_ci=[];seed_source=[]
    for family,records in formal.items():
        for seed,rec in records.items():
            for sourcekey in SOURCES:
                a=np.load(ROOT/'curves'/f"{rec['tag']}_step{rec['best_step']}_{sourcekey}.npz")
                source.append(dict(family=family,seed=int(seed),source=sourcekey,selected_step=rec['best_step'],**metrics(a['ranks'])))
            initial=rec['validation'][0]['macro_mrr']
            seed_source.append(dict(family=family,seed=int(seed),selected_step=rec['best_step'],macro_mrr=rec['best_source_validation_mrr'],initial_macro_mrr=initial,delta_initial=rec['best_source_validation_mrr']-initial,checkpoint_sha256=rec['checkpoint_sha256'],gate_parameters=rec['gate_parameters'],semantic_parameters=rec['semantic_parameters'],backbone_trainable_parameters=rec['backbone_trainable_parameters']))
    for sourcekey in SOURCES:
        vs=[];anchor=None
        for family in ['U-Cos','R-Cos-Shuffle']:
            arr=[]
            for seed,rec in formal[family].items():
                a=np.load(ROOT/'curves'/f"{rec['tag']}_step{rec['best_step']}_{sourcekey}.npz")
                if anchor is None:anchor=a['triples']
                assert np.array_equal(anchor,a['triples'])
                arr.append((1/a['ranks'].astype(float)).mean(1))
            vs.append(np.mean(arr,axis=0))
        dd,lo,hi=paired((vs[0]-vs[1])[:,None],'source'+sourcekey)[0]
        source_ci.append(dict(source=sourcekey,method='U-Cos',baseline='R-Cos-Shuffle',n=len(vs[0]),delta=dd,ci_low=lo,ci_high=hi))
    screens=[dict(family=r['family'],alpha=r['config']['alpha'],freedom=r['config']['freedom'],selected_step=r['best_step'],source_mrr=r['best_source_validation_mrr'],initial_macro_mrr=r['validation'][0]['macro_mrr'],delta_initial=r['best_source_validation_mrr']-r['validation'][0]['macro_mrr']) for r in choices['screens']]
    for fn,rows in [('P0_metrics.csv',p0),('graph_seed_robustness.csv',pergraph),('macro_seed_robustness.csv',macro),('query_stratified_metrics.csv',strata),('paired_bootstrap.csv',[r for r in intervals if r['query_group']=='all']),('query_stratified_bootstrap.csv',intervals),('source_validation.csv',source),('source_selection.csv',seed_source),('source_shuffle_bootstrap.csv',source_ci),('topology_screen.csv',screens)]:csvout(fn,rows)
    dump(ROOT/'results/summary.json',dict(macro=macro,graph=pergraph,source_selection=seed_source,source_shuffle_ci=source_ci,source_screens=screens))
    dump(ROOT/'results/query_stratified_metrics.json',strata)
    dump(ROOT/'results/query_stratified_bootstrap.json',intervals)
    dump(ROOT/'results/source_validation.json',source)
    write_report(frozen,choices,p0,pergraph,macro,strata,intervals,source,seed_source,source_ci,screens)
def write_report(frozen,choices,p0,graphs,macro,strata,intervals,source,seeds,source_ci,screens):
    op=json.loads((ROOT/'results/operator_tests.json').read_text());mem=json.loads((ROOT/'results/memory_preflight.json').read_text())
    lookup={(r['key'],r['query_group'],r['method'],r['baseline']):r for r in intervals}
    src={f:float(np.mean([r['macro_mrr'] for r in seeds if r['family']==f])) for f in formal_families(frozen)}
    diff=src['U-Cos']-src['R-Cos-Shuffle'];shwins=sum(r['delta']<0 for r in intervals if r['method']=='U-Cos' and r['baseline']=='R-Cos-Shuffle' and r['query_group']=='all')
    hub_rows=[r for r in intervals if r['method']=='U-Cos' and r['baseline']=='T-4D' and r['query_group']=='high_degree_hub' and r['n']]
    hub_neg=sum(r['delta']<0 for r in hub_rows);hub_signeg=sum(r['ci_high']<0 for r in hub_rows)
    report=['# 新阶段五（修正版）：不确定性门控的拓扑—余弦核协同与双层枢纽校准\n',
    f"本报告由冻结后的原始双向 filtered 排名生成。训练种子为1024、2026、3407，目标图不更新参数、不使用目标验证集选模。源验证 U-Cos−R-Cos-Shuffle 的三源三种子宏MRR差为 **{diff:+.6f}**；目标11图中乱序点估计优于真实对应 **{shwins}图**。不能将移除坐标投影本身当作语义对应有效的实验证明。\n",
    f"U-Cos 相对匹配 T-4D 的 high_degree_hub 组：{len(hub_rows)}个非空目标图中{hub_neg}图点估计下降，其中{hub_signeg}图条件区间全负。最终是否同时保住枢纽和长尾，以第8节逐图表为准，不预设成功。\n",
    f"HubCal源选共享系数为 **{frozen['calibration']['lambda_hub']}**。若为0，T-4D+HubCal与T-4D等价，不能把相同的目标结果算成新增的双层校准收益，也不能据此断言所有非零实体度校准都无效。\n",
    '## 1. 输入复核、实验范围与诊断修正\n',
    '用户所称 report_2.md 未出现在本地文件清单；本轮读取已完成阶段四 report.md，其第8节与请求所引数值匹配。阶段四低频组的 NELL_v3 +0.016671、NLIngram_50 +0.046885、NLIngram_25 +0.026652 的95%区间均跨零；FBIngram_25低支持组 +0.015551 的区间全正。多数点估计为正不能改写为多数区间全正。查询分组重叠，不能把总损失唯一归因于枢纽。\n',
    '阶段四代码对目标向量应用同一套源图拟合的PCA基，而非逐图独立拟合PCA，因此不能将其问题表述为“各图PCA坐标轴任意错位”。压缩损失、分布迁移和身份捷径均是可能机制，乱序反超本身不能唯一证明查表记忆。本轮按请求直接移除PCA和可训练语义投影，但不把上述尚未分离的原因写成已证实因果。\n',
    '原四锁定图、历史六图和ConceptNet外部图均已在以前阶段暴露，本轮属于冻结P0复现，不称从未接触的盲测或11个独立知识库。旧阶段仅只读复用，所有新增缓存、代码和结果在独立目录。\n',
    '## 2. 坐标无关余弦核与不透明掩码\n',
    '只将规范化官方名称输入冻结 BAAI/bge-small-en-v1.5，正关系使用清洗名称，逆关系统一 inverse of <name>。使用归一化CLS 384维向量，无PCA、无语义线性投影、无Qwen生成描述。官方标签取既有公开映射；P54等未有公开名称的标签按固定正则判为不透明，不推测含义。\n',
    '为满足每层仅2个语义参数，把非自槽位的查询余弦均值与最大值取等权平均，作为一个特征，再加相邻槽位余弦均值。二者按观察图模体类型Z-score，查询特征逐查询标准化，无可用上下文或零方差时为0。r=q槽位不进入查询余弦，无额外自环可学习项。语义logit为theta_q*c_q+theta_in*c_in，每层仅2个标量，共12个。\n',
    '不透明槽位所在超边以及不透明查询的语义残差严格置零。此性质针对固定参数下的语义通路；语义联合训练可能改变拓扑参数，因此不能保证整个U-Cos在WKIngram上与独立训练T-4D排名相同，也不能承诺全模型零负迁移。\n',
    '内积在共同正交旋转下不变，去除了原阶段直接读取PCA坐标的身份查表通道。两标量和关系相似度几何仍可能包含身份相关统计，不能宣称信息论上杜绝一切ID推断。成对乱序保留正逆配对和向量边际分布；固定映射seed8111，不据目标选择映射。\n',
    '本轮是同一乱序映射下的三个训练种子，不是三个独立乱序映射；真实对应相对这一映射的输赢，不能直接概括为对所有乱序的普遍优势或排列显著性检验。\n',
    '## 3. 四维拓扑、残差与不确定性\n',
    '复用阶段一/二按删边观察图独立计算的[freq,RA-global,RW-support,slot-marginal exclusivity]四维标准化缓存。历史s_pmi名称指槽位边际排他支持比，非标准对数PMI；RW是未归一化支持权重和，非概率。查询拓扑交互沿用已验证TopologyGate，恢复全部4维。\n',
    '每种拓扑分支按目标关系×模体类型归一化后，w_final=1+alpha_topo*(w_topo−1)+alpha_sem*(w_sem−1)。测试alpha=.15/.20/.25与图自适应；自适应严格为.15+.10*CV/(1+CV)，CV来自当前观察图原始s_pmi的标准差/绝对均值，不用目标标签。该公式不保证某个具名图获得最高权重，实际值列于analysis/target_inputs.json。\n',
    'alpha_sem=.20*sigmoid(−2*max(z_freq,z_pmi)−.5)*透明掩码，并对无非自上下文的查询置零。sigmoid有限输入下通常非零，强支持只能受抑制，不能保证100%枢纽性能保护。语义alpha逐超边变化，最终组均值不保证为1；不再做最终重归一化，避免重新激活静默边。因两alpha之和≤.45且两分支归一化权重为正，实数计算下最终权重≥.55。\n',
    'alpha是混合比例，不是最终权重相对1的绝对偏离上限。exp(2*tanh(g/2))经组均值归一化后具有宽松上界exp(4)，本阶段最终权重的统一宽松上界约25.12，不能把alpha=.2解读为所有权重都在1±.2内。\n',
    mdtable(['观察图','原始s_pmi CV','自适应alpha_topo','掩码后alpha_sem均值','掩码后alpha_sem最大值'],[[r['key'],f"{r['s_pmi_cv']:.4f}",f"{r['alpha_adaptive']:.4f}",f"{r['semantic_alpha_masked_mean']:.4f}",f"{r['semantic_alpha_masked_max']:.4f}"] for r in frozen['targets']]),
    '\n原始排他支持比的CV与实体枢纽偏度不是同一个量；本次NELL_v1的CV并未高于NLIngram，不能声称该CV公式已经实现“对NELL_v1自动更强”。公式在源训练前固定，未依据目标观察分布改成有利于具名图谱的规则。\n',
    '混合采用重计算自定义反向，仅保存超边级权重，反传时重建逐入射权重，避免六层同时保存大入射张量。无梯度截断、无目标更新，显式小图算子和梯度通过独立核验。\n',
    '## 4. 优化预算与仅源选模\n',
    'gate_only冻结所有预训练参数；joint_low_lr仅解冻关系骨干且LR=2e−6，实体骨干冻结。门控LR=1e−3，AdamW decay=.01，裁剪1，500步，B=2，64严格负例；选模步含0/100/250/500。三源各512原始查询、双向全候选filtered MRR等权选模，沿用阶段四固定查询和五折删边数据流。第0步保留，若选中0步如实报告未获得非零门控支持。\n',
    '先对8种拓扑配置在seed1024筛选，每种自由度各保留源最优alpha，并补跑另两种子；主拓扑自由度只由seed1024源验证选定，U-Cos与乱序继承该固定配置。两个自由度均有三种子结果；没有把所有8种网格都跑成三种子完整交叉实验。B1-short、U-Cos、R-Cos-Shuffle均三种子。各轨迹同种子同数据流，各自以源宏MRR选步；未选中轨迹与检查点保留。\n',
    '独立于拓扑选择的B1-short三种子通过worker_b1.py并行执行，训练函数、代码哈希、步数、批量和选模网格完全相同，仅实时进度文件重定向以避免并发写入。主流水线按相同哈希复用完成记录，不重复训练或按结果加预算。固定随机数流并不承诺并行稀疏浮点归约逐位确定。\n',
    'T-4D与T-4D-alt分别代表两种自由度各自源选最优的候选；若所选alpha不同，其目标差值不能唯一归因为是否更新主干。同alpha下的自由度对照和图自适应方案均在上表源网格中保留，但该网格不是三种子的完整交叉目标消融。\n',
    mdtable(['拓扑配置','自由度','alpha','选中步','源宏MRR'],[[r['family'],r['freedom'],r['alpha'],r['selected_step'],f"{r['source_mrr']:.6f}"] for r in screens]),
    '\n## 5. 实体层HubCal\n',
    '对尾预测和转为逆关系的头预测都减lambda*log(1+观察图入度)。观察图含显式逆边，因此此处入度等于逆边增强后的出度，也等价于原有向图入出度总和；不是原始正向事实入度，也不额外使用查询关系条件度。该项是排序校准候选，不能先验称为无偏。\n',
    f"lambda仅在T-4D已选源检查点的三源×三种子宏MRR上选择，网格0/.001/.003/.01/.03/.1/.3，单一共享lambda={frozen['calibration']['lambda_hub']}；权重不再更新，目标无搜索。lambda=0是有效的源选结果，不能为得到正结论而去掉。完整网格见results/hub_calibration.json。\n",
    mdtable(['lambda_hub','三源三种子源宏MRR','相对lambda=0'],[[lam,f"{score:.6f}",f"{score-frozen['calibration']['macro_mrr'][0]:+.6f}"] for lam,score in zip(frozen['calibration']['grid'],frozen['calibration']['macro_mrr'])]),
    '\n此网格使用过选模源查询，细小差异受选择噪声影响；若选择0，只表明本轮源协议不支持启用该项，目标模型对比并未单独检验任意非零校准的普遍效用。\n',
    '## 6. 源验证与乱序检验\n',
    mdtable(['方法','种子','选中步','源宏MRR','相对自身第0步','语义参数','主干可训练参数'],[[r['family'],r['seed'],r['selected_step'],f"{r['macro_mrr']:.6f}",f"{r['delta_initial']:+.6f}",r['semantic_parameters'],r['backbone_trainable_parameters']] for r in seeds]),
    '\n三种子源宏MRR汇总（种子SD与查询Bootstrap不是同一种不确定性）：\n',
    mdtable(['方法','源宏MRR均值','种子SD','平均Δ初始值','非零选中数'],[[f,f"{np.mean([r['macro_mrr'] for r in seeds if r['family']==f]):.6f}",f"{np.std([r['macro_mrr'] for r in seeds if r['family']==f],ddof=1):.6f}",f"{np.mean([r['delta_initial'] for r in seeds if r['family']==f]):+.6f}",sum(r['selected_step']>0 for r in seeds if r['family']==f)] for f in formal_families(frozen)]),
    f"\n正式轨迹第0步源宏MRR范围为[{min(r['initial_macro_mrr'] for r in seeds):.6f}, {max(r['initial_macro_mrr'] for r in seeds):.6f}]。后端浮点归约与近似并列排名存在数值波动，初始门控为单位权重时的细小差异不作方法收益；完整初值保留在source_selection.csv。\n",
    '\nU-Cos−R-Cos-Shuffle逐源条件配对95%区间：\n',
    mdtable(['源图','ΔMRR','95%下界','95%上界'],[[r['source'],f"{r['delta']:+.6f}",f"{r['ci_low']:+.6f}",f"{r['ci_high']:+.6f}"] for r in source_ci]),
    '\n这些源查询参与过选模，区间未校正选择偏差；不能视为独立确认集。每个方法的源选步不同，乱序胜出或落后都只检验本次固定预算下的对应关系，不能由单次输赢证明或反驳所有记忆机制。\n',
    '## 7. 冻结P0总体和逐图比较\n',
    '所有源选择、HubCal标量、语义、固定查询掩码、代码和检查点哈希先写入frozen.json，再开始本阶段目标排名。B0从官方权重重新评估；旧B1/B2排名在冻结后只读复用并逐项核对查询顺序。三种子先算指标再平均，不是集成打分。\n',
    'B0只有一份官方固定权重，没有训练随机种子；SD列记0是固定基线的记账约定，不表示已从三次独立训练估计出零方差，也不替代查询抽样区间。\n',
    mdtable(['目标组','方法','宏MRR','训练种子SD','ΔB0','Δ旧B1','Δ旧B2','ΔB1-short'],[[r['group'],r['family'],f"{r['mrr']:.6f}",f"{r['seed_sd']:.6f}",*[f"{r['delta_'+b]:+.6f}" for b in ['B0','B1','B2','B1-short']]] for r in macro]),
    '\n逐图均值与训练种子SD：\n',
    mdtable(['图谱','方法','MRR','种子SD','ΔB0','Δ旧B1','Δ旧B2'],[[r['key'],r['family'],f"{r['mrr']:.6f}",f"{r['seed_sd']:.6f}",*[f"{r['mrr']-next(x['mrr'] for x in graphs if x['key']==r['key'] and x['family']==b):+.6f}" for b in ['B0','B1','B2']]] for r in graphs]),
    '\n主模型逐图配对95%区间（完整其他比较见paired_bootstrap.csv）：\n',
    mdtable(['图谱','模型','参照','ΔMRR [95% CI]'],[[r['key'],r['method'],r['baseline'],ci(r)] for r in intervals if r['query_group']=='all' and r['method'] in ['T-4D','T-4D+HubCal','U-Cos'] and r['baseline'] in ['B0','B1','B2','T-4D','R-Cos-Shuffle']]),
    '\n## 8. 固定查询组：枢纽与长尾是否同时受益\n',
    '直接复用阶段四在排名前固定的查询掩码并核对顺序哈希，避免阶段五看到效果后重划子组。high_degree_hub仍是关系端点度聚合代理；low_incident_motif_support为关系入射模体标准化支持均值，二者不是实际错误或置信度金标。组间可重叠。missing_description只为跨阶段可比保留旧掩码，本轮输入没有生成描述。\n']
    for querygroup in ['high_degree_hub','low_incident_motif_support','low_relation_frequency','opaque_label','unseen_relation','missing_description','external_reverse_clean','external_reverse_counterpart']:
        report.append('\n### '+querygroup+'\n')
        rows=[]
        for target in frozen['targets']:
            key=target['key']
            for f in ['T-4D','T-4D+HubCal','U-Cos']:
                r=lookup.get((key,querygroup,f,'B0'))
                if r is None:continue
                rows.append([key,f,r['n'],ci(r),ci(lookup[key,querygroup,f,'B1']),ci(lookup[key,querygroup,f,'B2'])])
        report.append(mdtable(['图谱','模型','原始查询数','ΔB0及95%CI','Δ旧B1及95%CI','Δ旧B2及95%CI'],rows))
    report+=['\n为隔离新机制增量，再直接比较U-Cos−T-4D以及HubCal−T-4D（同源选拓扑底座）：\n',
    mdtable(['图谱','查询组','U-Cos−T-4D','HubCal−T-4D'],[[t['key'],g,ci(lookup[t['key'],g,'U-Cos','T-4D']),ci(lookup[t['key'],g,'T-4D+HubCal','T-4D'])] for t in frozen['targets'] for g in ['high_degree_hub','low_incident_motif_support','low_relation_frequency']]),
    '\n未观察到显著下降不等于已证明保住性能；没有预注册非劣效界，因此本轮只报告点估计与区间，不作严格非劣效结论。n=1的Bootstrap区间退化，n=2极不稳定，均不支持总体稳健性主张。\n',
    '## 9. 算子、冻结参数和完整性核验\n',
    f"混合算子前向最大误差{op['forward_error']:.3g}，各输入最大梯度误差{max(op['gradient_errors']):.3g}，float64 gradcheck通过。预训练实体打分与T初始恒等最大误差{op['model_identity_max_score_error']:.3g}，U初始误差{op['semantic_model_identity_max_score_error']:.3g}；旋转内积最大误差{op['rotation_invariance_error']:.3g}。误差不包装为方法收益。语义完整FB训练批量预检峰值分配显存{mem['peak_allocated']/2**30:.3f}GiB。\n",
    '完整性与预训练参数逐项核验见results/verification.json。冻结校验通过只说明协议、文件和计算一致，不代表科学假设成立。\n',
    '## 10. 统计解释和交付索引\n',
    '逐图／逐组5000次百分位配对Bootstrap，将同一原始三元组头尾方向绑定；先对固定三个训练结果的逐查询RR均值作差，再重采样。区间仅反映给定训练结果下的查询抽样不确定性，训练种子SD单独列出；未作多重比较校正，图谱和三元组存在共享来源依赖。组宏平均是等权描述性汇总，不作独立知识库总体推断。\n',
    '- results/P0_metrics.csv：逐种子四指标与检查点哈希。\n- results/graph_seed_robustness.csv、macro_seed_robustness.csv：逐图／图组指标均值和种子SD。\n- results/paired_bootstrap.csv、query_stratified_bootstrap.csv：全图与固定子组配对区间。\n- results/query_stratified_metrics.csv：全部非空／空组指标。\n- results/source_selection.csv、source_validation.csv、topology_screen.csv：完整源选择与未选中筛选项。\n- results/source_shuffle_bootstrap.csv、hub_calibration.json：语义对照与单标量选择。\n- evaluation/：原始双向排名与查询；curves/：逐步损失、梯度、采样哈希与验证排名。\n- frozen.json、protocol.json、results/verification.json：冻结协议与最终核验。\n']
    (ROOT/'report.md').write_text('\n'.join(report),encoding='utf-8')
def formal_families(frozen):return list(frozen['formal'])
if __name__=='__main__':main()
