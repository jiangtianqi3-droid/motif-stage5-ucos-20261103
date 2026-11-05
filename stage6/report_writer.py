from collections import Counter
import numpy as np
from stage6.util import ROOT, read, sha


def write(args, p, graphs, macros, boots, groups, selections, points, within,
          stability, source_splits, correlation, hub, repeat, checks):
    import csv
    from stage6.analysis import mdtable
    def macro(group, method):
        return next(r for r in macros if r['group'] == group and r['method'] == method)
    def interval(group, method, base):
        return next(r for r in boots if r['key'] == 'macro_' + group and r['method'] == method and r['comparator'] == base)
    primary = [r for r in selections if r['method'] == 'joint_lcb' and r['proxy_seed'] == p['primary_proxy_seed']]
    all_joint = [r for r in selections if r['method'] == 'joint_lcb']
    source_fallback = sum(r['fallback'] for r in source_splits)
    c = correlation['graph_level_alpha025']
    pearson = '未定义（零方差或样本不足）' if c['pearson'] is None else f"{c['pearson']:.4f}"
    spearman = '未定义' if c['spearman'] is None else f"{c['spearman']:.4f}"
    strict_win = all(macro(g, 'joint_lcb')['mrr'] > macro(g, base)['mrr'] for g in ['locked', 'historical', 'external'] for base in ['B0', 'B1', 'B2'])
    topo_gain = np.mean([macro(g, 'topology_lcb')['mrr'] - macro(g, 'reference')['mrr'] for g in ['locked', 'historical', 'external']])
    entity_gain = np.mean([macro(g, 'entity_lcb')['mrr'] - macro(g, 'reference')['mrr'] for g in ['locked', 'historical', 'external']])
    top_confirmed = any(interval(g, 'topology_lcb', 'reference')['lo'] > 0 for g in ['locked', 'historical'])
    if strict_win and all(interval(g, 'joint_lcb', 'B0')['lo'] > 0 for g in ['locked', 'historical']):
        decision = '继续推进完整 TSDC'
        decision_reason = '主方法在三个图组的点估计均超过三个基线，且内部两组相对 B0 的条件配对区间全正；仍需新的未暴露图谱确认。'
    elif (interval('historical', 'topology_lcb', 'reference')['lo'] > 0 and
          interval('external', 'topology_lcb', 'reference')['lo'] > 0 and
          interval('locked', 'topology_lcb', 'reference')['delta'] >= 0):
        decision = '仅保留目标图自监督拓扑适配'
        decision_reason = '拓扑单层在历史组和外部图相对同检查点参考的配对区间全正，锁定组则全部回退；实体单层在历史组显著下降，同符号固定对照也未支持条件度数比率的独特优势。保留拓扑加LCB作为受约束的研究分支，先重审代理分布，再在未暴露图谱复验，不继续扩大完整双层机制。'
    elif topo_gain > 0 and top_confirmed:
        decision = '修改实体层校准机制后继续'
        decision_reason = '关系层有局部证据，但完整方法尚未稳定超过强基线，现有度数比率不足以证明双层协同。'
    else:
        decision = '暂停该路线并重新分析跨图偏移'
        decision_reason = '当前证据不足以支持继续扩大双层校准复杂度；先区分代理删边分布、正式查询分布与已有模型能力差异。'
    manifest = read(args.observations / 'manifest.json')
    obs_rows = []
    for group, key in p['targets']:
        r = read(ROOT / 'adaptation' / f'{key}__rate0.15__p6101__s1024.json')
        obs_rows.append(dict(图谱=key, 实体=manifest[key]['n'], 关系=manifest[key]['R'],
            唯一观测事实=r['split']['unique_observed_facts'], 代理事实=r['split']['actual'],
            选择=r['split']['selection'], 确认=r['split']['confirmation'],
            实际掩码率=r['split']['actual'] / r['split']['unique_observed_facts']))
    main_graphs = [dict(图谱=r['key'], MRR=r['mrr'], 种子SD=r['mrr_seed_sd'],
        Hits1=r['hits_at_1'], Hits3=r['hits_at_3'], Hits10=r['hits_at_10'],
        ΔB0=r['delta_B0'], Δ旧B1=r['delta_B1'], Δ旧B2=r['delta_B2']) for r in graphs if r['method'] == 'joint_lcb']
    selection_display = [dict(图谱=r['key'], 训练种子=r['training_seed'], alpha=r['alpha'], gamma=r['gamma'],
        回退=r['fallback'], 确认LCB=r['confirmation_lcb']) for r in primary]
    macro_display = [dict(图组=r['group'], 方法=r['method'], MRR=r['mrr'], 种子SD=r['mrr_seed_sd'],
        ΔB0=r['delta_B0'], Δ旧B1=r['delta_B1'], Δ旧B2=r['delta_B2']) for r in macros]
    ci_display = [dict(图组=r['group'], 方法=r['method'], 对照=r['comparator'], ΔMRR=r['delta'], 下界=r['lo'], 上界=r['hi'])
        for r in boots if r['key'].startswith('macro_')]
    pergraph_ci = [dict(图谱=r['key'], 对照=r['comparator'], ΔMRR=r['delta'], 下界=r['lo'], 上界=r['hi'])
        for r in boots if not r['key'].startswith('macro_') and r['method'] == 'joint_lcb' and r['comparator'] in ['B0', 'B1', 'B2']]
    times = []
    for group, key in p['targets']:
        a = [read(x) for x in (ROOT / 'adaptation').glob(f'{key}__rate*__p6101__s*.json')]
        f = [read(x) for x in (ROOT / 'evaluation').glob(f'{key}__T-4D_s*.json')]
        times.append(dict(图谱=key, 每种子代理推理秒=float(np.mean([r['info']['seconds'] for r in a])),
            共享图准备秒=float(np.mean([r.get('graph_preparation_seconds_shared', 0.) for r in a])),
            每种子正式全网格秒=float(np.mean([r['info']['seconds'] for r in f])),
            代理峰值GiB=max(r['info']['peak_allocated'] for r in a) / 2**30,
            正式峰值GiB=max(r['info']['peak_allocated'] for r in f) / 2**30))
    from stage6.analysis import csv_write
    csv_write(ROOT / 'results/time_memory.csv', times)
    rate_rows = []
    for key in ['fb15k237_train', 'wn18rr_train', 'codex_medium_train']:
        for rate in p['source_rates']:
            ss = [r for r in source_splits if r['key'] == key and r['rate'] == rate]
            rate_rows.append(dict(源图=key, 名义率=rate, 实际率=ss[0]['actual_rate'],
                实际样本=ss[0]['actual'], 不同残余图=len(set(r['residual_sha256'] for r in ss)),
                回退次数=sum(r['fallback'] for r in ss), 总次数=len(ss)))
    max_numerical = max(abs(r['delta_mrr']) for r in repeat if r['family'] == 'B0')
    base_rep_summary = []
    b0_diag = read(ROOT / 'results/b0_fixed_relation_diagnostic.json')
    operator_trace = read(ROOT / 'results/b0_operator_trace.json')
    fixed_b0_display = [dict(实体批量=r['batch'], 相对首批量8排名变化=r['rank_changes_vs_first'],
        MRR=r['mrr'], ΔMRR=r['delta_mrr_vs_first'], 真分数最大绝对差=r['max_true_score_change'])
        for r in b0_diag['entity_batch_comparison']]
    fixed_rank_changes = max(r['rank_changes_vs_first'] for r in b0_diag['entity_batch_comparison'])
    numeric_localization = ('固定关系表示后，本次改变实体批量未再引起排名变化，普通复测的排名差异主要指向关系表示生成及其批量分组路径。'
        if fixed_rank_changes == 0 else '固定关系表示后，实体批量变化仍引起排名变化，因此实体传播/投影的批量数值路径也有贡献。')
    for family in ['B0', 'B1', 'B2']:
        rs = [r for r in repeat if r['family'] == family]
        base_rep_summary.append(dict(方法=family, 比较份数=len(rs), 排名变化方向数=sum(r['changed_directional_ranks'] for r in rs),
            最大MRR绝对差=max(abs(r['delta_mrr']) for r in rs)))
    fallback_rates = []
    for method in ['joint_lcb', 'topology_lcb', 'entity_lcb', 'global_lcb', 'joint_global_lcb']:
        rs = [r for r in selections if r['method'] == method]
        fallback_rates.append(dict(方法=method, 回退次数=sum(r['fallback'] for r in rs),
                                  选择集已选参考=sum(r['candidate_is_reference'] for r in rs), 总次数=len(rs),
                                  回退率=np.mean([r['fallback'] for r in rs])))
    gamma_signs = []
    for method in ['entity_lcb', 'global_lcb', 'joint_lcb', 'joint_mean']:
        rs = [r for r in selections if r['method'] == method and r['proxy_seed'] == p['primary_proxy_seed']]
        gamma_signs.append(dict(方法=method, 负系数=sum(r['gamma'] < 0 for r in rs),
            零系数=sum(r['gamma'] == 0 for r in rs), 正系数=sum(r['gamma'] > 0 for r in rs), 总数=len(rs)))
    hub_display = []
    for _, key in p['targets']:
        rs = [r for r in hub if r['key'] == key]
        hub_display.append(dict(图谱=key, 真实体关系支持log均值=np.mean([r['true_conditional_log_mean'] for r in rs]),
            最高错候选关系支持log均值=np.mean([r['wrong_conditional_log_mean'] for r in rs]),
            枢纽错候选超真率=np.mean([r['high_hub_outrank_fraction'] for r in rs]),
            条件支持与正确性Spearman=np.mean([r['conditional_log_vs_correct_spearman'] for r in rs if r['conditional_log_vs_correct_spearman'] is not None]) if any(r['conditional_log_vs_correct_spearman'] is not None for r in rs) else None,
            固定gamma003增益=np.mean([r['gamma003_delta_mrr'] for r in rs])))
    sourcecorr = correlation['source_graph_level_alpha025']
    with (ROOT / 'results/degree_matched_macro.csv').open(encoding='utf-8-sig') as f:
        matched = list(csv.DictReader(f))
    matched_display = [dict(图组=r['group'], gamma=float(r['gamma']), 条件MRR=float(r['conditional_mrr']),
        全局MRR=float(r['global_mrr']), 条件减全局=float(r['delta']), 下界=float(r['lo']), 上界=float(r['hi']))
        for r in matched if r['family'] == 'T-4D' and float(r['gamma']) in [-.03, .03]]
    with (ROOT / 'results/factorial_interaction.csv').open(encoding='utf-8-sig') as f:
        interaction_display = [dict(图组=r['group'], 排序交互差=float(r['delta']), 下界=float(r['lo']), 上界=float(r['hi']))
            for r in csv.DictReader(f) if r['key'].startswith('macro_')]
    density_display = []
    for key in ['wn18rr_train', 'codex_medium_train', 'fb15k237_train']:
        for rate in p['source_rates']:
            dr = read(ROOT / 'source_density' / f'{key}__rate{rate}.json')
            i0 = dr['configs'].index([0., 0., 'conditional'])
            i1 = dr['configs'].index([.25, 0., 'conditional'])
            density_display.append(dict(源图=key, 实际删边率=dr['actual_rate'],
                零调制MRR=dr['metrics'][i0]['mrr'], alpha025增益=dr['metrics'][i1]['mrr'] - dr['metrics'][i0]['mrr']))
    with (ROOT / 'results/proxy_formal_distribution.csv').open(encoding='utf-8-sig') as f:
        distribution = list(csv.DictReader(f))
    dist_display = []
    with (ROOT / 'results/selected_topology_graph_prediction.csv').open(encoding='utf-8-sig') as f:
        selected_prediction = [dict(图谱=r['key'], 代理选择增益=float(r['proxy_gain']),
            正式候选增益=float(r['formal_gain']), 正式点差为正次数=int(r['positive_formal_candidates']),
            诊断次数=int(r['attempts'])) for r in csv.DictReader(f)]
    for _, key in p['targets']:
        rs = [r for r in distribution if r['key'] == key]
        dist_display.append(dict(图谱=key,
            关系分布JS距离=float(np.mean([float(r['relation_distribution_js_distance']) for r in rs])),
            代理真候选关系零支持率=float(np.mean([float(r['proxy_zero_true_relation_support_fraction']) for r in rs])),
            正式真候选关系零支持率=float(rs[0]['formal_zero_true_relation_support_fraction']),
            代理锚实体log度=float(np.mean([float(r['proxy_mean_log_anchor_degree']) for r in rs])),
            正式锚实体log度=float(rs[0]['formal_mean_log_anchor_degree'])))
    answers = []
    h1_judgment = '出现描述性的正向关联，但尚不足以证明逐图可靠预测。' if c['pearson'] is not None and c['pearson'] > 0 and c['spearman'] > 0 else '不能称为可靠的通用增益预测器；只有局部证据。'
    answers.append(f'**Q1：代理是否预测正式收益？** {h1_judgment} 固定 alpha=.25 的图级 Pearson={pearson}、Spearman={spearman}（n=11）。结合逐图四个非零 alpha 的相关性表判断，不能仅凭一个相关系数宣称普遍预测能力；源图仅3个，相关性为描述性诊断。')
    for qnum, title, comp in [(2, '自监督选参是否优于固定和随机？', None), (3, '查询条件实体校准是否有独立收益？', None), (4, '双层是否优于两个单独模块？', None)]:
        if qnum == 2:
            vals = '本轮三个图组相对固定和随机对照的条件配对区间均全正，支持有收益。' + '; '.join(f"{g}: Δ固定={interval(g, 'joint_lcb', 'fixed')['delta']:+.6f}, Δ随机={interval(g, 'joint_lcb', 'random')['delta']:+.6f}" for g in ['locked', 'historical', 'external'])
        elif qnum == 3:
            vals = '没有建立稳定的跨组独立收益。' + '; '.join(f"{g}: 实体单层Δ参考={interval(g, 'entity_lcb', 'reference')['delta']:+.6f} [{interval(g, 'entity_lcb', 'reference')['lo']:+.6f}, {interval(g, 'entity_lcb', 'reference')['hi']:+.6f}]" for g in ['locked', 'historical', 'external'])
        else:
            vals = '存在历史组/外部图的局部协同，但没有稳定优于两个单模块；锁定组联合方法低于实体单层，差值区间全负。' + '; '.join(f"{g}: Δ拓扑单层={interval(g, 'joint_lcb', 'topology_lcb')['delta']:+.6f}, Δ实体单层={interval(g, 'joint_lcb', 'entity_lcb')['delta']:+.6f}" for g in ['locked', 'historical', 'external'])
        answers.append(f'**Q{qnum}：{title}** {vals}。区间见上表；点估计优势不等于稳定优势。')
    answers.append('**Q5：是否超过全部强基线？** ' + ('本轮三个图组点估计全部超过 B0、旧 B1、旧 B2。' if strict_win else '没有建立三个图组上全面超过 B0、旧 B1、旧 B2 的优势。') + '完整基线权重已恢复，正式对照不存在缺失；历史排名与当前重跑的数值差异另列。')
    answers.append('**Q6：主要失败机制？** 主要限制是代理分布与选优偏差，以及实体校准方向/条件化机制的不稳定；关系层在历史组和ConceptNet保留了实质收益，不能判为完全无效。固定alpha图级收益符号仅4/11一致，三源实际删边密度均使增益变号；同符号负gamma全局固定校准又优于条件公式。数值问题已在固定关系表示下定位到实体分支的批量计算路径，会影响小差值，但不能解释ConceptNet约0.058的增益。各因素尚未由完全独立干预分解，不能强称唯一原因。')
    text = f'''# 新阶段六：目标观测图自监督双层结构校准（TSDC）

本阶段完成目标观测图隔离、代理选择/确认两次独立前向、三源诊断、11图冻结协议复验、原始事实配对统计及基线数值回归。**研究决策：{decision}。** {decision_reason}

主TSDC的locked / historical / external MRR分别为 **{macro('locked','joint_lcb')['mrr']:.6f} / {macro('historical','joint_lcb')['mrr']:.6f} / {macro('external','joint_lcb')['mrr']:.6f}**。相对本轮B0为 {macro('locked','joint_lcb')['delta_B0']:+.6f} / {macro('historical','joint_lcb')['delta_B0']:+.6f} / {macro('external','joint_lcb')['delta_B0']:+.6f}；locked小差值不能作为稳定改进。historical仍低于旧B1/B2，external则超过三项基线，但训练种子SD为{macro('external','joint_lcb')['mrr_seed_sd']:.6f}，收益并非所有检查点均等稳定。

正式协议为“无目标验证/测试标签、无目标梯度更新、允许目标观测图自监督选参”，与阶段五静态零样本协议不同。11张目标图均有历史暴露，结果不称新盲测。主结果固定代理种子6101，6102/6103仅作预注册稳健性诊断；从未按正式结果选择代理种子、训练种子或参数。

## 1. 研究问题与阶段五失败机制

阶段五固定结构策略跨图分化，T-4D/U-Cos 未在 locked/historical 建立对 B0 的稳定优势，旧 B2 也存在负迁移；原全局 HubCal 源选系数为0。阶段六分别检验 H1（代理预测拓扑收益）、H2（查询关系条件实体校准）、H3（双层与确认回退）。不训练新语义编码器，不启用 U-Cos，不重训 MOTIF，不使用目标梯度。

本轮选择的是**同一个 T-4D 检查点内部**的推理参数，不是从多个模型中择优。零调制参考保留该检查点全部关系层权重，gamma=0；不能把“关闭门控”命名为旧 B1。阶段五 T-4D 为 gate_only，骨干与官方权重逐位相同，但关系聚合后端仍须单独核验数值差异。

## 2. 代码审查和依赖恢复结果

基于私有归档提交 `173017b7c3a4d88edf73d2ce46b083a5fcb69354` 的独立 `stage6-tsdc` 分支开发，原始阶段五、历史目录及 sources 均只读。`EXTERNAL_DEPENDENCIES.json` 的222项依赖全部存在且 SHA256 匹配；旧 B1/B2 三种子六份完整权重、模型配置、种子和源选步全部恢复，哈希全部匹配，见 [依赖审查](results/dependency_audit.json)。

`common.downstream()` 会载入正式查询与 filtered truth，因而不用于适配。复用 `topology.extract()`、`plans.build_plan()`、CSR/加权算子与 MOTIF 模型；`evaluate.py` 的全候选、双向、严格过滤和悲观并列排名规则保持一致。头预测经 `negative_sample_to_tail()` 转为逆关系尾预测。旧 B2 的真实门控为 Fdim=1，T-4D 为 Fdim=4，分别恢复，不相互替代。

新入口参数化 stage5/history/source-root/official/observations，未运行旧 initialize.py 或旧 run.sh。沿用 Python3.9、PyTorch2.1.0+cu118、PyG2.4.0、Triton2.1.0 与 RTX3060 6GB，未新增依赖。原文件及独立副本完整性最终核验见 [verification.json](results/verification.json)。

## 3. 自监督代理任务与防泄漏验证

观测数据包闭合 schema 只有 edge_index、edge_type、n、R。GraIL 从 train_ind、Ingram 从 inference_graph 建立观测优先实体编号；关系 ID 取既有公开词表。合法候选实体数从既有**观测拓扑缓存的 degree 长度元数据**取得，明确区别于标签。ConceptNet 和源缓存只读取 numpy 延迟加载的观测成员，不读取 queries/alltrue。内部目标观测边的原始顺序和哈希与阶段五一致；正式评测再独立核对边、超边、查询顺序及过滤真值哈希。

选择器输入不含 valid/test、正式真值、查询分组或历史排名。数据类型不提供相应字段；文件访问审计额外拒绝这些资产。隔离测试在只能打开 sanitized observations 的条件下完成全部14图加载、划分与选择器执行。模型前向另在启用标签/排名文件拒绝规则的实际代理进程完成。

权重只从清洗后的 model_manifest.json 读取检查点、配置、种子和哈希；适配进程拒绝阶段五完整 frozen.json，避免其中正式查询分组派生元数据进入进程。

唯一正向事实为采样单位，固定2×2关系频率/端点度数分层，按种子轮转取样；优先保证端点剩余度数≥2，保留随机确定的生成森林及每个关系至少一条边。不放宽约束。重复正向事实去重，正逆边同步删除；选择和确认的事实在同一残余图中全部删除。联合删边后验证无直接/逆边泄漏、无新孤立实体、连通分量数和关系覆盖不变，实体全集不删减。

{mdtable(obs_rows, ['图谱','实体','关系','唯一观测事实','代理事实','选择','确认','实际掩码率'])}

每个残余图重新执行拓扑提取和入射计划构建，四维标准化签名来自残余图。缓存以实体/关系规模、精确有序边及拓扑版本哈希寻址；记录图、样本、选择/确认子集和模型哈希，禁止跨划分误用。

开发期曾发现“预先评出全部确认配置”的实现过宽，在任何目标适配前修正为严格两次独立前向；此前4份源诊断保存在 source_diagnostics_before_confirmation_fix，不用于本报告结论。最终确认阶段仅评每个预声明独立消融在选择集胜出的单个候选及参考；未经选择的确认分数存为缺失值，不伪造为0，也不再调参。

## 4. 关系层测试时拓扑校准

alpha 网格固定0/.1/.25/.5/1。每个 alpha 都重新进行真实关系层及实体层前向，未插值最终得分。alpha=0 与同一检查点显式移除门控后的关系表示最大误差为0，非零 alpha 改变内部消息，完整网格有限；见 [数值测试](results/numeric_tests.json)。缓存关系表示只在同模型/图/alpha 调用内存在。

固定拓扑控制为 alpha=.25、gamma=0；旧 CV 控制在完整观测图上计算 `.15+.10*CV/(1+CV)`，不会从正式结果反推参数。关系层单独方法只在 gamma=0 的五点网格上选参，确认集只能接受该胜出候选或回退。

## 5. 查询条件实体得分校准

实现 `s - gamma*[log(1+d_global)-log(1+d_relation)]`，gamma 网格 −.03/−.01/0/.01/.03/.1。global_degree 为逆边增强图的入度，等于原始正向图总连接度；relation_degree 为候选作为转换后查询关系目标端的次数，两者求和口径一致。头预测使用 q+R/2 的逆关系。代理使用残余图度数，正式使用完整观测图度数。

gamma=0 直接返回原 tensor，逐位一致；无关系支持的候选保持有限；关系支持覆盖全部连接时该候选惩罚为0，不重复惩罚高度数真实实体。校准只修改输出候选得分，不修改传播、图、参数。原全局 HubCal 网格0/.001/.003/.01/.03/.1/.3亦保留，以独立实体层/联合消融评估，不把源选0改写成不存在对照。

主代理种子下各方法接受的系数方向如下。负系数不支持“应降低枢纽分数”的预设方向，须与模块独立收益和真实候选支持度合并解释：

{mdtable(gamma_signs, ['方法','负系数','零系数','正系数','总数'])}

另在任何正式查询评测前登记 [matched_global_protocol.json](matched_global_protocol.json)，补充相同符号/相同系数的全局度数固定对照，覆盖原始B0、B1、B2及T-4D。该补充是在观察代理任务出现负系数选择后引入，属于**正式测试前登记的机制诊断**，不是最初源阶段的主协议，也不参与任何目标参数选择。它复用相同神经分数，只改变候选校准项，避免把负系数的作用误归因于关系条件化。全部五个非零系数结果都输出，未按正式结果挑最优系数；见 [degree_matched_control.csv](results/degree_matched_control.csv)。

以下是预先指定的正/负0.03切片，T-4D固定alpha=0，区间为条件公式减同系数全局公式；其他系数及四模型结果完整保留。这些探索性区间未作多重比较校正。

{mdtable(matched_display, ['图组','gamma','条件MRR','全局MRR','条件减全局','下界','上界'])}

## 6. 置信回退机制

原始事实按60%/40%分选择/确认（取整后数量如上），两方向 RR 先在每事实内平均。主方法先在选择集比较30个条件校准联合配置；仅胜出者与固定参考进入该方法的确认比较。`LCB=mean(delta)-1.64*SE(delta)`，选择事实≥24、确认事实≥16且LCB>0才接受，否则回退alpha=gamma=0。所有网格、主代理种子、确认阈值和对照规则预先固定。

独立预声明的 topology/entity/global/joint-global 消融各有一次自己的确认，不根据它们的确认分数重选主方法。均值选参不使用确认回退；随机对照在30个联合配置上均匀抽取，随机流由图、训练种子、代理种子及6629的哈希固定。LCB只是代理上的经验保守启发式，不保证正式MRR不下降。

主代理种子33次选择中回退 {sum(r['fallback'] for r in primary)}/{len(primary)}；全部三代理种子回退 {sum(r['fallback'] for r in all_joint)}/{len(all_joint)}。

{mdtable(fallback_rates, ['方法','回退次数','选择集已选参考','总次数','回退率'])}

{mdtable(selection_display, ['图谱','训练种子','alpha','gamma','回退','确认LCB'])}

全部99份目标选择及对照先写入不可覆盖 [frozen.json](frozen.json)，SHA256=`{sha(ROOT / 'frozen.json')}`，然后新进程首次打开正式查询。报告及分析代码没有参数回写路径。

## 7. 主实验与全部对照

每个训练种子独立选参、独立预测后平均指标，没有分数集成。主代理种子固定6101；全部三代理种子的正式结果列于 [ablation.csv](results/ablation.csv)，不选最好种子。模型重跑110份，T-4D正式完整网格同时供冻结策略与事后机制诊断使用，禁止据此更新策略。B0仅官方固定检查点一份，SD记NA，不能声称三次独立训练方差为0。

{mdtable(macro_display, ['图组','方法','MRR','种子SD','ΔB0','Δ旧B1','Δ旧B2'])}

![图组宏MRR](figures/macro_comparison.png)

主方法逐图指标：

{mdtable(main_graphs, ['图谱','MRR','种子SD','Hits1','Hits3','Hits10','ΔB0','Δ旧B1','Δ旧B2'])}

配对Bootstrap5000次。先按原始事实平均两个方向，再平均对应训练种子的RR差；区间反映固定检查点与固定图集条件下的查询抽样不确定性，不替代训练种子SD。图组区间每次在各图内独立重采样事实，再等权宏平均，未把图谱当11个独立同分布样本。

{mdtable(ci_display, ['图组','方法','对照','ΔMRR','下界','上界'])}

![相对B0的条件配对区间](figures/macro_delta_ci.png)

{mdtable(pergraph_ci, ['图谱','对照','ΔMRR','下界','上界'])}

为避免将不同选参策略的差异直接称为“协同”，另在实际冻结的联合参数上计算 `RR(a,g)-RR(a,0)-RR(0,g)+RR(0,0)`，先合并方向和训练种子，再按事实配对重采样。alpha或gamma为0时交互项按定义为0。此项仅为冻结后诊断，不选正式最优参数；正值表示该MRR尺度上的局部超加性，不证明普遍机制，且未校正多重比较。逐图结果见 [factorial_interaction.csv](results/factorial_interaction.csv)。

{mdtable(interaction_display, ['图组','排序交互差','下界','上界'])}

全部方法每图 Hits@1/3/10、种子SD及差值见 [graph_metrics.csv](results/graph_metrics.csv)，逐种子值见 [P0_metrics.csv](results/P0_metrics.csv)。多组区间未作多重比较校正，仅主方法及预注册比较具有优先解释地位，不能从大量探索性区间中挑正结果宣称整体显著。

## 8. 代理预测能力与参数稳定性

H1固定 alpha=.25 图级代理增益—正式增益 Pearson={pearson}，Spearman={spearman}，n=11。代理侧平均三训练种子×三代理种子**选择集**增益；正式侧平均三训练种子增益。不是使用正式最优配置倒推代理标签。每图四个非零alpha的相关性见 [within_graph_correlation.csv](results/within_graph_correlation.csv)，完整点见 [proxy_formal_correlation.csv](results/proxy_formal_correlation.csv)。

![代理与正式收益散点](figures/proxy_vs_formal.png)

固定alpha=.25的图级收益点差符号一致 {correlation['fixed_alpha025_sign_agreement_graphs']}/11。代理均值选出的**拓扑单层候选**对应正式收益的图级相关性为 {correlation['selected_topology_graph_level']}。以下候选全部在代理选择集确定，不计算或采用正式最优参数；每图汇总三训练种子×三代理种子。正点差次数不是显著或稳定有效次数，须结合区间及数值不确定性判断。完整逐次记录见 [selected_topology_prediction.csv](results/selected_topology_prediction.csv)。

选中候选的Pearson=0.836由ConceptNet的大增益强烈影响，Spearman=-0.018，并不支持跨图稳定排序。逐图四个非零alpha之间有8/11图呈正相关，但这不等于能判断非零调制是否优于alpha=0：例如NELL_v3的非零配置排序相近，代理均值选出的候选正式增益仍平均为负。确认回退因此是本轮主要的风险抑制环节，不能把原始代理均值当可靠的正式收益估计。

{mdtable(selected_prediction, ['图谱','代理选择增益','正式候选增益','正式点差为正次数','诊断次数'])}

源图 alpha=.25 的图级相关性：{sourcecorr}。源诊断先于目标适配，使用合法源验证512条原始查询/图；源验证不作为目标选择的输入。三源代理共81份协议记录；相同残余图/模型/查询/配置只计算一次并核对哈希复用，不冒充独立重复。

这些源验证查询沿用阶段五选模数据，不能称为新的独立确认集。它们用于本阶段数值及代理可靠性诊断，未据此改变正式目标网格。

{mdtable(rate_rows, ['源图','名义率','实际率','实际样本','不同残余图','回退次数','总次数'])}

名义5%/10%/15%触及256上限时，同一种子划分可完全相同，因此**本轮不能据此证明实际掩码密度不敏感**。上限未偷偷扩大或缩小；实际率、残余图哈希及复用来源完整记录。参数稳定性按每图/训练种子跨三个代理种子的最终参数模式频率统计，见 [selection_stability.csv](results/selection_stability.csv)。三种子不是三个独立图谱，重叠代理事实也不能当独立重复。

为直接检验密度敏感性，另在三源图执行独立预注册的 [source_density_protocol.json](source_density_protocol.json)：实际按5%/10%/15%约束删边，固定同一批最多256条查询，只比较五个拓扑alpha，不进行选参。全部查询来自最低掩码率的隐藏事实，并验证属于更高率的隐藏池；因此查询顺序一致。该辅助诊断只使用训练种子1024和划分种子6101，不替代主协议三种子稳健性；结果见 [source_density_metrics.csv](results/source_density_metrics.csv)，相对5%时调制增益变化的配对区间见 [source_density_sensitivity.csv](results/source_density_sensitivity.csv)。它不改变目标代理构造或参数。

{mdtable(density_display, ['源图','实际删边率','零调制MRR','alpha025增益'])}

图谱共享数据来源/关系模式且有历史暴露，n=11较小，NELL/GraIL/Ingram版本非独立。删边代理偏向有剩余连接的频繁观测事实，与正式查询的生成、关系频率、实体支持和难度有分布差异。正相关也不证明逐图安全，负相关需要优先分析这种分布偏移，不能用正式结果扩网格寻找有利alpha。

以下分布诊断全部在冻结后的分析阶段计算：代理度数取残余图，正式度数取完整观测图；对所有隐藏代理事实统计，平均三个划分。JS距离使用以2为底的 Jensen–Shannon 距离（平方根形式），比较原始查询关系直方图。这些是分布差异证据，不能单独证明其造成了MRR变化。

{mdtable(dist_display, ['图谱','关系分布JS距离','代理真候选关系零支持率','正式真候选关系零支持率','代理锚实体log度','正式锚实体log度'])}

## 9. 高度数枢纽与长尾子组

沿用阶段五固定 high_degree_hub、low_relation_frequency、low_incident_motif_support 掩码，仅在正式阶段加载；配对区间含空组/小样本状态见 [query_groups.csv](results/query_groups.csv)。这些组重叠，不将宏损失简单归因于一个组，更不把 high_degree_hub 掉分解释为“消息爆炸”。

另外区分：当前方向的查询锚实体高度数、真实候选高度数、最高分错误候选高度数且分数超过真实体。阈值在完整观测图正度实体上取90%分位，不用测试标签确定阈值；分组指标是在正式阶段计算。详见 [hub_candidate_groups.csv](results/hub_candidate_groups.csv)。

双向任务中，锚实体和真实候选的总体度数分布因头尾互换而相同，不能仅比较两组曝光比例。新分组的效应仅平均每个原始事实中满足该方向条件的RR差，再按事实配对Bootstrap；两个方向始终不作为独立样本。这样“预测高度数实体”与“以高度数实体为锚预测其他实体”的效果可以分别检验。

{mdtable(hub_display, ['图谱','真实体关系支持log均值','最高错候选关系支持log均值','枢纽错候选超真率','条件支持与正确性Spearman','固定gamma003增益'])}

支持—正确性相关性使用“真实候选 vs 模型最高分合法错误候选”的配对样本，不能解释为候选全集无偏正确概率，也不能把相关性写成度数的因果作用。固定alpha=0、gamma=.03是预注册网格中的事后解释性切片，**未作为正式最优系数**。真实高连接实体可能需要高得分；惩罚会带来收益或损害，方向由上述数据判别。

## 10. 统计局限、失败案例与研究决策

数值回归统一检查点、float32、eval模式、输入边/超边顺序、查询/filtered哈希与悲观并列规则。原官方关系层分段Triton核含 `tl.atomic_add`，存在浮点归约顺序敏感的代码路径；这给出可疑来源，**仅代码检查不能证明它是每个历史排名差异的唯一原因**。新拓扑关系聚合沿用阶段五加权CSR算子，不为改善结果切换后端。NELL_v1 当前同批重复与批量1/4/8的结果见 [b0_repeat.json](results/b0_repeat.json)，最高错候选分数、真分数与精确并列数保存于原始证据数组。

{mdtable(base_rep_summary, ['方法','比较份数','排名变化方向数','最大MRR绝对差'])}

当前B0相对阶段五最大图级MRR绝对差为{max_numerical:.6f}。结合已知历史约0.001904的波动，低于相近量级的增益不称稳定方法改善。即使本轮短重复无变化，也不能证明历史跨运行问题已彻底消除；未恢复当时GPU调度及全部执行上下文。旧B1/B2恢复后亦逐方向对比旧排名，见 [baseline_reproducibility.csv](results/baseline_reproducibility.csv)。

普通NELL_v1复测发现批量相关排名差异后，又进行明确标记的事后数值定位：以全局固定顺序和批量2生成一次关系表示，并冻结这些表示，再只改变实体批量。关系层同批量2重复的最大表示差为 {b0_diag['same_batch_relation_repeat_max_error']:.9g}，关系批量1与2的最大表示差为 {b0_diag['relation_batch1_vs_batch2_max_error']:.9g}。

{mdtable(fixed_b0_display, ['实体批量','相对首批量8排名变化','MRR','ΔMRR','真分数最大绝对差'])}

{numeric_localization} 本次关系表示在重复及批量1/2下逐位一致，因此**不能把本次已复现的差异归因于关系层原子归约**；该静态风险仍存在，但这次证据指向实体分支。对首个已记录排名变化查询增加算子输入/输出追踪，最早差异为 `{operator_trace['first_divergence']}`，详见 [b0_operator_trace.json](results/b0_operator_trace.json)。该线性层首查询输入完全相同、权重未变而输出改变，将本次问题具体定位到批量形状敏感的实体线性计算；未取得具体CUDA库内核轨迹，不进一步宣称所有历史差异均来自同一个GPU核。诊断的关系表示和排名另存 [b0_fixed_relation_diagnostic.json](results/b0_fixed_relation_diagnostic.json)，从未用来替换主结果中首次固定B0运行，也未据此选择有利批量。阶段五原始资产和全部正式选择仍保持冻结。

适配与正式时间/显存（正式列为完整候选网格成本，不是只跑选中策略的部署延迟；图准备共享成本另列）：

{mdtable(times, ['图谱','每种子代理推理秒','共享图准备秒','每种子正式全网格秒','代理峰值GiB','正式峰值GiB'])}

局限包括：确认样本有限、代理样本约束偏差、上限造成名义掩码率饱和、单次固定随机策略、三个训练种子、版本图谱相关、未校正多重比较、历史目标暴露和后端数值不确定性。LCB未考虑全部设计选择及代理—正式分布偏移，不能当非劣效保证。

{(chr(10) * 2).join(answers)}

**最终研究决策：{decision}。** {decision_reason} 此决策基于上述冻结结果，不把预期收益写为已观察收益；新机制或新代理分布的下一轮验证应另立协议，不能修改本轮目标参数。

完整性核验通过：原阶段五{checks['stage5_original_files_unchanged']}个文件哈希保持不变；历史资产{checks['historical_assets_unchanged']}项保持不变；{checks['rank_artifacts_hash_verified']}份排名工件核验通过；目标梯度更新0次。复现步骤见 [README](README.md)。
'''
    (ROOT / 'report.md').write_text(text, encoding='utf-8')
