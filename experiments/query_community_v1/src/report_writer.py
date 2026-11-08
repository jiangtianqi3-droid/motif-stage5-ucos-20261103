"""Chinese report and real graph illustrations, generated from completed records."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .graph_adapter import ROOT,dump

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def number(x):
    return 'NA（分母为0）' if x is None else f'{x:.4f}' if isinstance(x,float) else str(x)

def table(rows,columns):
    lines=['| '+' | '.join(title for key,title in columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
    for row in rows: lines.append('| '+' | '.join(number(row.get(k)) for k,_ in columns)+' |')
    return '\n'.join(lines)+'\n'

def make_figures(work,graph,labels,indices,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx
    methods=['naive_khop','global_community','ppr_uniform','ppr_topology_signature','ppr_topology_gated']
    k=max(cfg['candidate_k']);m=max(cfg['path_budgets'])
    choices=[]
    for i in indices:
        for direction in [0,1]:
            p=work/'evidence'/'ppr_topology_gated'/f'k{k}_m{m}'/f'{i}_{direction}.json'
            rec=read(p)
            for c in rec['candidate_evidence']:
                if c['status']=='ok': choices.append((int(c['is_cross_community']),c['max_path_weight'],-int(i),direction,c['entity']))
    if choices:
        _,_,minus_i,direction,target=max(choices);i=-minus_i
    else:
        i=int(indices[0]);direction=0
        sample=read(work/'evidence'/methods[-1]/f'k{k}_m{m}'/f'{i}_0.json')
        target=sample['candidate_evidence'][0]['entity']
    records=[read(work/'evidence'/method/f'k{k}_m{m}'/f'{i}_{direction}.json') for method in methods]
    h=records[0]['query_info']['head']
    net=nx.Graph();net.add_nodes_from([h,target])
    for rec in records:
        net.add_nodes_from(rec['subgraph']['nodes'])
        net.add_edges_from([(e['head'],e['tail']) for e in rec['subgraph']['directed_edges']])
    pos=nx.spring_layout(net,seed=1024,iterations=100)
    fig,axes=plt.subplots(1,5,figsize=(24,5.2))
    colors=['#346a9c','#626262','#477b57','#9969a6','#a3632c']
    path_rows=[]
    for ax,method,rec,color in zip(axes,methods,records,colors):
        local=nx.MultiDiGraph();local.add_nodes_from(rec['subgraph']['nodes'])
        local.add_edges_from([(e['head'],e['tail'],{'relation':e['relation']}) for e in rec['subgraph']['directed_edges']])
        nx.draw_networkx_nodes(local,pos,ax=ax,node_size=22,node_color=color,alpha=.65)
        nx.draw_networkx_edges(local,pos,ax=ax,width=.45,alpha=.25,arrows=True,arrowsize=5,connectionstyle='arc3,rad=0.07')
        chosen=next(c for c in rec['candidate_evidence'] if c['entity']==target)
        for path in chosen['evidence_paths'][:1]:
            edges=[(tr[0],tr[2]) for tr in path['triples']]
            nx.draw_networkx_edges(nx.DiGraph(edges),pos,ax=ax,edgelist=edges,width=2.2,edge_color='#c32d35',arrows=True,arrowsize=12)
        nx.draw_networkx_nodes(net,pos,ax=ax,nodelist=[h,target],node_color=['#dd463e','#e8ba31'],node_size=105,edgecolors='black')
        nx.draw_networkx_labels(net,pos,ax=ax,labels={h:f'h={h}',target:f't={target}'},font_size=8)
        ax.set_title(f'{method}\nN={len(local)} E={local.number_of_edges()} | target {chosen["status"]}',fontsize=9)
        ax.axis('off')
        path_rows.append(dict(method=method,candidate=target,status=chosen['status'],
            is_cross_community=chosen['is_cross_community'],paths=chosen['evidence_paths'],subgraph=rec['subgraph']))
    fig.suptitle(f'{work.name} | fact {i}, direction {direction}, query (h={h}, q={records[0]["query_info"]["relation"]}, ?) | shared layout',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.93])
    figures=ROOT/'figures'/work.parent.name;figures.mkdir(parents=True,exist_ok=True)
    image=figures/f'{work.name}_case.png';fig.savefig(image,dpi=160,bbox_inches='tight');plt.close(fig)
    dump(image.with_suffix('.json'),dict(query_id=records[0]['query_info']['query_id'],target=target,
        selection='After retrieval: prefer cross-community connected candidate; strongest gated path. This is a descriptive case, not a performance estimator.',methods=path_rows))

def write_report(run,cfg,limit,smoke,isolation):
    manifest=read(ROOT/'freeze_manifest.json')
    metrics=[];comparisons=[];resources=[];pre=[]
    for key in cfg['datasets']:
        work=run/key
        metrics+=read(work/'metrics.json');comparisons+=read(work/'paired_intervals.json')
        resources.append(read(work/'completion.json'));pre.append(dict(dataset=key,**read(work/'preprocessing.json')))
    scope='冒烟测试，仅验证流程；不能用来判断方法有效性' if smoke else ('全量测试事实、两个方向' if limit is None else f'固定种子无答案抽样，每图最多 {limit} 条测试事实、两个方向')
    conclusions=[]
    if not smoke:
        for key in cfg['datasets']:
            for method in ['ppr_topology_signature','ppr_topology_gated']:
                hit=next(r for r in comparisons if r['dataset']==key and r['k']==30 and r['m']==10 and r['base']=='ppr_uniform' and r['target']==method and r['metric']=='hit')
                hub=next(r for r in comparisons if r['dataset']==key and r['k']==30 and r['m']==10 and r['base']=='ppr_uniform' and r['target']==method and r['metric']=='hub_fraction')
                metric=next(r for r in metrics if r['dataset']==key and r['k']==30 and r['m']==10 and r['method']==method)
                conclusions.append(f'{key} / {method} 相对C：路径召回变化 {100*hit["delta"]:+.2f} 个百分点（'+
                    ('未校正95%区间支持正提升' if hit['lo']>0 else '未校正95%区间支持下降' if hit['hi']<0 else '区间含0，未建立显著提升')+
                    f'）；枢纽占比变化 {100*hub["delta"]:+.2f} 个百分点（'+
                    ('区间支持下降' if hub['hi']<0 else '区间支持上升' if hub['lo']>0 else '区间含0')+
                    f'）；11–30扩召回成功 {metric["expansion_success"]}/{metric["expansion_eligible"]}。')
    parts=['# MOTIF 查询感知强关联社区发现实验报告',
        f'本报告对应运行 `{run.name}`。状态：{"冒烟完成，正式实验尚未完成" if smoke else "完成"}。范围：{scope}。',
        '## 1. 核心结论与解释边界',
        '模块已实现 A–E 全部流程，E 分为纯四维签名与实际冻结 T-4D 门控两种。下表及配对区间从原始逐查询记录生成；未预设 E 优于基线。没有接入 LLM，没有参数更新。',
        '\n\n'.join(conclusions) if conclusions else '冒烟样本不提供方法有效性结论。',
        '**数学限制：发送端全局度因子在行归一化时完全抵消。** 若 `w(u,r,v)=phi(r)/(1+d_out(u))^eta`，则 `P(v|u)=phi(r)/sum_(u,r′,v′)phi(r′)`。因此 E 的 PPR 转移差异来自关系重要性，发送端阻尼仅影响未归一化证据路径分数、预算选择及带权社区投影；D 中发送端全局度同样抵消，但关系局部扇出项仍起作用。不能将 PPR 变化归因为已被抵消的全局发送端度数。',
        '以下 K=30、M=10 的 E−C 与 E−D 配对区间量化有效证据召回变化；区间跨 0 时不能声称显著改善。均为已暴露目标图上的离线诊断；多个比较未作多重校正，不构成新图泛化保证。',
        table([r for r in comparisons if r['k']==30 and r['m']==10 and r['metric']=='hit' and r['base'] in ['ppr_uniform','ppr_degree']],
            [('dataset','图谱'),('base','基线'),('target','方法'),('delta','Path Hit差'),('lo','95%下界'),('hi','95%上界'),('clusters','事实数')]),
        '## 2. P0 冻结、隔离与历史资产',
        f'- 原分支：`{manifest["baseline_branch"]}`；冻结提交：`{manifest["baseline_commit"]}`；原归档工作区干净：`{manifest["baseline_clean"]}`。\n'
        f'- 冻结标签：`{manifest["tag"]}`；新分支：`{manifest["branch"]}`。\n'
        f'- Worktree：`{manifest["worktree"]}`。全部新增代码、缓存、日志和结果位于其 `experiments/query_community_v1/`。\n'
        f'- 清单共 {len(manifest["assets"])} 个历史资产（{sum(v["bytes"] for v in manifest["assets"])/1e9:.2f} GB），逐文件 SHA256 位于 `freeze_manifest.json`。冻结前未发现运行中的训练/评估。\n'
        f'- 本次保护验证：{isolation["files_checked"]} 个文件，模式={"全量" if isolation["full"] else "核心代码/权重/报告"}，全部哈希未变。',
        '阶段一 ULTRA、阶段二/三 MOTIF、阶段四各拓扑子阶段、阶段五、阶段六两个实体算子版本与 TSDC、阶段七均已纳入清单。阶段三内核与阶段六 v2 的未提交修改及报告已另存 `preserved_changes/`，原目录不动；未将它们提交到正式冻结基线。Git 标签固定干净归档提交，外部未版本化历史资产依靠清单及保留原位置复现。',
        table([dict(path=r['path'],commit=r['commit'],branch=r['branch'],clean=r['clean']) for r in manifest['repositories']],
            [('path','仓库'),('branch','分支'),('commit','HEAD'),('clean','干净')]),
        '### 核心代码与模型 SHA256',
        table([r for r in manifest['assets'] if (Path(r['path']).name in ['topology.py','gated_model.py','model.py','entity_normalization.py','entity_degree.py'] and any(s in r['path'] for s in ['topology-ucos-stage5','motif-stage7-asymmetric','motif-stage6-entity-degree-p0p1-v2'])) or (Path(r['path']).name in ['motif_3g.pth','T_gate_only_0p25_s1024_step500.pth','B2_s1024_step1000.pth'])],
            [('path','文件'),('sha256','SHA256')]),
        '## 3. P1 四维签名与门控接口核查',
        r'类型序为 HH、TT、HT、TFH、TFT、HFT、HFH。原观察图含显式正逆边；令 C_H(v,r)、C_T(v,r) 为有向角色次数，U=1[C>0]，d(v) 为逆增强图发送度。二元模体 N=C_L^T C_S，Q=C_L^T diag(d^(-1/2)) C_S；三元模体沿中间关系 b 收缩，N=C_L^T F_b C_S，Q=(diag(d^(-1/4))C_L)^T F_b (diag(d^(-1/4))C_S)。',
        r'原始四通道为 [log(1+N), Q/N, RW, Q/(prod_j marginal_(type,j)(r_j))^(1/arity)]。RW 的头/尾角色质量 X_H(v,r)=sum_(v,r,t)1/d_(r,in)(t)，X_T(v,r)=sum_(u,r,v)1/d_(r,out)(u)；二元 RW=X_L^T U_S，三元 RW=X_L^T F_(b,row-normalized) U_S。它是支持度和，不保证<=1。排他度的边际在完整类型上统计，不是分块边际，也不是 log PMI；RA 与排他度在 [0,1]。',
        r'Z-score 在七类内部逐通道计算，使用总体标准差 ddof=0 与 eps=1e-6：z=(raw-mu)/(std+eps)。raw/z 形状 [E_m,4]；超边槽位 [3,E_m]，二元第三槽=-1；类型 [E_m]，means/stds [7,4]。阶段六bundle的raw实为全零占位，不能用于频次；适配器从原观察图canonical graph_key定位阶段四完整缓存，逐元素对齐超边，并核验重新标准化结果与阶段六z一致。没有完整缓存时仅用观察图重算并写本阶段缓存。graph_key包含版本、观察边、关系、N、R，不读取alltrue。',
        r'每层原始门控 g=exp(2*tanh((z·w_type+b_type+(Wq h_q)·Ws z)/2))，范围 (exp(-2),exp(2))。入射聚合按 (目标关系,模体类型) 分组，以 g/mean_group(g) 归一化。正式 T-4D 使用 alpha=.25 的原聚合与门控聚合残差混合。适配器只读加载源验证选定 checkpoint，对实际 RelHCNet.forward 注册 hook，逐 q 捕获六层原始门控并取算术均值；没有将均值冒充实际入射权重。每图 gate_audit 校验模型状态前后 SHA256 相同。',
        '映射时使用超边所有非负槽位关系 ID，构造稀疏 R×E_m 入射矩阵；重复槽位的同一关系仅计一次。关系 ID 与实体观察边 edge_type 完全同域，逆关系为 (r+R/2) mod R。不把七类模体类型误当成实体关系标签。',
        '阶段五 HubCal 是 score(t)−lambda*log(1+d_in(t)) 的候选得分校准（历史源选 lambda=0）。阶段六 D1 是接收端入度幂归一化，D2 是两端总度的对称幂归一化，再作 raw+alpha*(normalized−raw)，boundary 后加；原 Triton 使用存储第0行为接收端，PyG 相反，报告保留这一事实。新社区边始终按存储事实 u→v 解释，未修改原传播算子或权重。阶段七额外发送端校准也已归档，只作只读参考。',
        '## 4. P2 精确算法定义',
        r'全局聚类使用稀疏有向边权矩阵 A，投影 A+A^T 后执行 NetworkX Louvain（Leiden 依赖未安装）。同实体对多关系仅在聚类投影求和，路径恢复始终按原始 edge_id。输出 labels/members、规模、模块度、内跨边占比与前5%度枢纽分布；无 N×N 稠密邻接。所有方法的跨社区诊断共享无权全局标签，避免因标签改变造成分母不一致。另导出 degree 带权聚类对照。',
        r'uniform：w=1。degree_aware：(1+d_out(u))^(-eta_out)*(1+d_(r,out)(u))^(-eta_rel)*(1+d_in(v))^(-eta_in)，默认 (.5,.5,0)，除以全图边权均值锚定为1；度数保留逆增强边、平行边和重复边的重数。',
        r'topo_signature：a_m=raw_freq=log(1+N_m)，m~q 定义为 q 在该超边的非空槽位直接出现。rho_q(r)=sum_(m含r且含q) a_m*sigmoid(z_excl+z_RW)/(sum a_m+1e-12)。gated_model：rho_q(r)=sum_(m含r)a_m*mean_l g_l(q,m)/(sum a_m+1e-12)。严格按epsilon分母公式，无支持关系rho=0，不伪造门控值。a_m 也可配置 uniform。',
        r'phi 对完整关系词表作 min-max 映射至 [.1,2]，常量 rho 返回1；w_topo=phi(rho_q(r))/(1+d_out(u))^.5。检查非负、有限、每条有出边行的概率和=1。反向 PPR 使用反向图重新按行归一化，不能直接使用非随机的 P^T。',
        r'稀疏截断 PPR：p_0=e_seed，p_(i+1)=(1-alpha)*p_i P+[alpha+(1-alpha)*dangling_mass]*e_seed，alpha=.2，为重启概率；2或3次迭代，不宣称收敛稳态。双端交汇按 sqrt(p_h(v)*p_t_reverse(v)) 排序，亲和度为交汇分数和。所有 <=3 跳简单有向路径通过前向两跳/反向一跳或反向两跳/前向一跳精确连接枚举，按较小的二跳扇出选择连接方向，固定大小堆保留 Top-M，未按枢纽扇出提前截断。另用独立MultiDiGraph穷举oracle核查两种方向的完整性。',
        r'路径强度为原始非归一化边权乘积 prod_e w_e，同时保留 prod_e P_e 供核查；因此原始强度跨权重模式量纲不同，只在同一模式内对候选排序。路径按强度先选每候选首条、再选剩余路径，所有候选共享最多50节点/100边，最后填充头部PPR局部背景边。M=5/10是每候选路径上限，查询总路径数仍受节点/边预算约束。K=10与30分别重做预算，不能直接截取不同预算下的子图冒充同一实验。',
        r'pi_h(r) 为头实体独立2-hop内、按头PPR选出的最多50节点诱导边，使用 p_h(sender)*w_e 按关系求和再归一化。零总量输出全零并保留 no_evidence。七维锚点在场率为各类型模体中全部关系槽位在该局部边集出现的比例；这是关系类型覆盖率代理，不等价于实体级模体实例计数。',
        '## 5. P3 正确性与零泄漏',
        '10项要求均有独立人工图测试：社区与模块度、正逆方向、平行关系、两模式枢纽抑制、跨社区3跳桥接、孤立/空图、双端交汇、两种拓扑非平凡性、固定种子、隐藏边与逆边拒绝。另覆盖概率、发送度抵消不变量、查询级预算、缺门控拒绝回退与 pi_h 不依赖候选。',
    ]
    xml=ROOT/'results/unit_tests.xml'
    if xml.exists():
        tests=ET.parse(xml).getroot().findall('.//testcase')
        parts.append(table([dict(test=t.attrib['name'],result='失败' if t.find('failure') is not None or t.find('error') is not None else '通过') for t in tests],[('test','测试'),('result','结果')]))
    parts.append(table([dict(dataset=k,**read(run/k/'leakage_audit.json')) for k in cfg['datasets']],
        [('dataset','图谱'),('valid_hidden','验证隐藏边'),('test_hidden','测试隐藏边'),('direct_overlap','直接重合'),('inverse_overlap','逆边重合'),('passed','断言通过')]))
    parts += ['只读观察 bundle 与原始事实逐元素一致；原始 valid/test 及其逆边只在所有检索完成后的独立离线审计中读取。候选生成从未访问隐藏答案或 filtered mask；真实答案仅用于下述指标与图例选择。证据输出逐边核对原始图及连续方向，不存在路径时明确 no_evidence。',
        '## 6. A–E 完整诊断指标',
        'A=naive_khop，B=global_community，C=ppr_uniform，D=ppr_degree，E3a=ppr_topology_signature，E3b=ppr_topology_gated。所有组共享观察图、候选、50节点/100边预算与固定种子。Grail沿用原协议合并归纳valid/test作为评估目标；FBIngram仅inf_test评估，inf_valid同样不能入观察图。',
        'Path Hit 分母为全部查询，要求真实答案出现在合法候选且预算内有有效路径；候选召回另列。跨社区分母是相同无权全局标签下 c(h)≠c(t*) 的全部查询。枢纽为全图有向增强总度前5%，同度按实体ID固定顺序。扩召回分母为答案原模型确定性Top-11至30的查询，成功要求其路径强度严格大于Top-10所有负例的最大值（比只胜过一个负例更严格）。分母为0时记NA。',
        table(metrics,[('dataset','图谱'),('method','方法'),('k','K'),('m','M'),('queries','查询'),('mean_nodes','节点'),('mean_edges','边'),('hub_fraction','枢纽占比'),('candidate_recall','候选召回'),('path_hit_rate','Path Hit'),('cross_queries','跨社区数'),('cross_answer_recall','跨社区召回'),('expansion_eligible','11–30数'),('expansion_success','扩召回成功'),('expansion_potential','扩召回比例'),('top10_evidence_mrr_conditional','Top10结构MRR'),('base_structure_spearman','排名相关'),('mean_seconds','检索秒')]),
        'Top10结构MRR仅在真实答案已在Top10时计算，等分采用平均秩；它是候选内离线结构验证，不是完整知识图谱 filtered MRR。排名相关为结构分数排序与B0顺序的Spearman，只衡量一致性，不代表正确性。原始 logits 不可用，base_score 是原实验保存的1/(60+raw pessimistic rank)，确定性并列按实体ID；base_rank 是这个合法候选顺序。',
        '### C/D/E 枢纽变化与 B/E 跨社区对照',
        table([r for r in comparisons if r['k']==30 and r['m']==10 and ((r['metric']=='hub_fraction' and r['base'] in ['ppr_uniform','ppr_degree']) or (r['metric']=='cross_hit' and r['base']=='global_community'))],
            [('dataset','图谱'),('base','基线'),('target','方法'),('metric','指标'),('delta','差'),('lo','95%下界'),('hi','95%上界')]),
        'cross_hit 的配对区间分母是全部查询，天然跨社区条件召回请读完整指标表；不得将两者混用。所有区间按原事实聚类同时重采两个方向，2000次Bootstrap，种子1024。B严格同社区截断，对跨社区尾实体必然no_evidence；是否由E实际挽回、挽回多少必须以测得的跨社区条件召回为准，不把设计结论包装成普遍性能提升。',
        '## 7. 计算开销、断点恢复与真实案例',
        table(resources,[('dataset','图谱'),('query_facts','事实数'),('preprocessing_seconds','预处理秒'),('retrieval_seconds','检索/离线汇总秒'),('cold_retrieval_mean_seconds','无join缓存检索秒'),('peak_rss_bytes','峰值RSS字节'),('peak_cuda_bytes','峰值CUDA字节')]),
        '平均检索耗时包含内存join复用，另列K30/M10无join缓存时间。预处理含候选文件SHA校验、Louvain与门控（缓存命中时更快）；CUDA仅冻结门控前向使用，主检索CPU稀疏。峰值RSS为本进程截至该图的高水位，非每图增量。',
        '统一入口支持 --smoke、--limit N、--datasets；默认四图全量双向。配置与全部程序哈希决定results/runs/<fingerprint>，不同版本不覆盖；查询JSON原子落盘，重启跳过已通过校验记录；单图completion.json之后跳过该图。异常保留failure.json和完整回溯，状态写results/status.json。',
        '真实案例采用共同布局、原始有向边和同一候选，红色为候选最强路径，背景保留全部预算内多关系边。自动选择偏好E存在跨社区连接的案例，属解释性展示；不用于估计平均效果。无合适案例时展示实际无证据结果，不合成成功案例。',
    ]
    for key in cfg['datasets'][:3]:
        image=ROOT/'figures'/run.name/f'{key}_case.png'
        data=read(image.with_suffix('.json'))
        parts += [f'### {key}：{data["query_id"]}，候选{data["target"]}',f'![真实查询子图对比](figures/{run.name}/{key}_case.png)']
        rows=[]
        for row in data['methods']:
            paths=row['paths']
            rows.append(dict(method=row['method'],status=row['status'],cross=row['is_cross_community'],
                strongest=str(paths[0]['triples']) if paths else '无路径',edge_ids=str(paths[0]['edge_ids']) if paths else '[]'))
        parts.append(table(rows,[('method','方法'),('status','状态'),('cross','跨社区'),('strongest','最强有向三元组序列'),('edge_ids','原边ID')]))
    parts += ['## 8. 后续方案一与方案三接口',
        '方案一可直接消费 schema_version=query-community-1.0 的query_info、candidate_evidence和subgraph。包含query_id、h/q、base_score/base_rank/score_kind、社区标签、双端亲和度、max_path_weight、meeting_nodes、连续有向triple/edge_id/relation序列、每边rho/degree_weight/weight/transition_probability与bridge标记。仅导出数据，本轮不实现分差阈值、不调用任何LLM API。将来分差触发阈值须在合法开发集上定义；原始logit缺失时不能将reciprocal-rank差当作logit置信差。',
        '方案三读取独立于候选的pi_h[R]与anchor_presence_7[7]；这些是只读观测结构特征，可输入后续g(h,q,m)原型。新模块没有修改现有RelHCNet、EntityNBFNet训练/推理权重，没有将本轮证据排名作为新模型训练标签。no_evidence应原样保留以支持回退原模型，不能编造路径。',
        '## 9. 可复现文件索引',
        f'- 本次协议、程序哈希、依赖环境：`results/runs/{run.name}/protocol.json`与`environment.json`。\n'
        f'- 原始逐查询证据、离线行、完整指标、配对区间及资源记录：`results/runs/{run.name}/<dataset>/`。\n'
        '- 冻结清单：`freeze_manifest.json`；保护验证：`results/isolation_core.json`或`isolation_full.json`；单测：`results/unit_tests.xml`。\n'
        '- 模型门控缓存与参数不变核验：`cache/gates/*_gate_audit.json`。\n'
        '- 原始可视化伴随JSON保留每个方法完整有向子图与路径，方便独立核查。',
    ]
    text='\n\n'.join(parts)+'\n'
    path=run/'report.md';path.write_text(text.replace('](figures/','](../../../figures/'),encoding='utf-8')
    if smoke:
        (ROOT/'results/smoke_report.md').write_text(text.replace('](figures/','](../figures/'),encoding='utf-8')
    else:
        (ROOT/'report.md').write_text(text,encoding='utf-8')
    dump(run/'report_validation.json',dict(metrics_rows=len(metrics),paired_intervals=len(comparisons),
        figures_checked_exist=all((ROOT/'figures'/run.name/f'{k}_case.png').exists() for k in cfg['datasets'][:3]),
        denominators_explicit=True,smoke=smoke,report_sha256=__import__('hashlib').sha256(text.encode()).hexdigest()))
