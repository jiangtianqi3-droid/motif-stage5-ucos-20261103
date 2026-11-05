"""Chinese report from verified results plus an explicit evidence-based decision."""
import csv
from collections import Counter
import numpy as np
from stage7.util import ROOT,read,sha,log


NAMES={'reference':'同检查点α0','fixed025':'固定α0.25','stage6_mean':'阶段六采样·均值','stage6_lcb':'阶段六采样·LCB',
 'single_mean':'A/10%·均值','single_lcb':'A/10%·LCB','equal_mean_unconfirmed':'12环境均值无确认',
 'equal_mean':'12环境等权均值+确认','sample_weighted':'样本量权重+确认','longtail_weighted':'长尾权重+确认',
 'worst_mean':'最坏均值+确认','robust_lcb':'12环境鲁棒LCB','four_env_lcb_10':'10%四环境LCB',
 'three_density_A_lcb':'A三密度LCB','four_env_mean_10':'10%四环境均值+确认','random':'随机α','B0':'B0','B1':'旧B1','B2':'旧B2'}


def table(name):
    with (ROOT/'results'/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def number(v):
    return 'NA' if v is None or v=='' else f'{float(v):.6f}'


def md(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def run(args):
    p=read(ROOT/'protocol.json');frozen=read(ROOT/'frozen.json');verify=read(ROOT/'results/verification.json')
    analysis=read(ROOT/'results/analysis_status.json')
    assert analysis['status']=='complete' and not verify['errors']
    decision=read(ROOT/'research_decision.json')
    gm=table('P0_metrics.csv');macro=table('macro_metrics.csv');ci=table('paired_bootstrap.csv')
    mci=table('macro_paired_bootstrap.csv');choices=table('selection_decisions.csv');corr=table('correlations.csv')
    masks=table('mask_rates.csv');density=table('density_sensitivity.csv');distribution=table('proxy_formal_distribution.csv')
    numeric=read(ROOT/'results/numeric_probe.json');gpu=read(ROOT/'results/gpu_checks.json');tie=read(ROOT/'results/tie_checks.json')
    exposure=read(ROOT/'results/historical_exposure_search.json');new=read(ROOT/'observations/manifest.json')
    overlap=read(ROOT/'results/pretraining_observation_overlap.json');preflight=read(ROOT/'results/new_graph_preflight.json')
    keyorder=[k for _,k in p['targets'] if any(r['key']==k for r in gm)]
    texts=[]
    texts.append('# 阶段七：分布鲁棒的目标观测图自监督拓扑适配\n')
    texts.append('本报告仅陈述实际运行结果。所有排名来自既有检查点的真实前向，不进行梯度更新、实体度数校准、语义融合或分数插值。')
    texts.append(f"完成状态：{verify['status']}；正式评测{analysis['graphs']}图/{analysis['formal_runs']}个模型运行；封存{frozen['selection_count']}组目标选择；核对{verify['preserved_files']}份阶段五/六原始文件，内容哈希均保持不变。")
    texts.append(f"研究决策：**{decision['decision']}**。{decision['summary']}")
    texts.append('## 1. 研究问题与阶段六审计\n')
    texts.append('检验H1单分布代理不足、H2多分布覆盖、H3鲁棒选择更稳定、H4结构分布与掩码密度影响迁移。阶段六11张图均是历史暴露目标，不能视为本阶段的新盲测。阶段六固定α0.25的图级Pearson约−0.071、Spearman约−0.027，不能据ConceptNet单点的巨大收益认定一般预测能力。旧双层协同和实体校准没有稳定证据，本阶段停止扩展这两项。')
    texts.append('阶段六实际掩码率差异有两个独立原因：256条上限使ConceptNet为256/99997≈0.256%，NELL-v1则为124/833≈14.886%；生成森林、保留关系覆盖及逐步剩余度数≥2的约束进一步限制稀疏图。原2×2桶轮转对桶近似等权，会改变自然事实分布；森林保护及端点约束又偏向可删除的环和较高连接度上下文。正式查询本身的关系分布、词表与图规模，及删除后拓扑变化，是另外三类来源，不能全归因于掩码率。')
    texts.append('已逐表审计阶段六要求的代理关联、选择预测、密度、稳定性、消融、交互项与完整性结果；见 [审计记录](results/stage6_audit.json)。阶段六原报告、冻结选择及结果没有重写。')
    texts.append('## 2. 接口复用、检查点与新增数据\n')
    texts.append('观测加载复用封闭的ObservedGraph；关系超图和四维签名复用topology.extract，CSR复用plans.build_plan；模型加载复用已核验的阶段六适配器。正式查询仅由独立进程在全局封存后读取，并对历史图核对阶段五查询、观测图、超图及过滤真值哈希。没有调用common.downstream把正式标签交给选择器，也没有运行initialize.py。')
    texts.append('B0为官方motif_3g；T-4D为阶段五gate_only检查点；旧B1/B2为已恢复的真实历史权重与实现。T-4D的α0是同一检查点关闭额外拓扑残差，不能叫作旧B1。每一训练种子独立校准，不在不同检查点之间进行测试时模型选择。完整路径、源选择记录与SHA见只读 [模型清单](../stage6/model_manifest.json)。')
    texts.append(f"历史暴露检索检查了{exposure['scanned_files']}份研究文件；仅有{len(exposure['matches'])}处新候选名称命中，均须结合证据解释，不能把官方数据目录中的名称视为实际旧实验。早期已做结构审计的Metafam、ILPC Small及若干WikiTopics明确不标为全新目标。")
    rows=[]
    for key,rec in new.items():
        ov=next((r for r in overlap['rows'] if r['key']==key),{})
        rows.append([key,rec.get('family'),rec.get('n'),rec.get('R'),rec.get('observed_facts'),
                     preflight.get(key,{}).get('status','不可取得'),ov.get('syntactic_fact_overlap','NA')])
    texts.append(md(['新增候选','来源','实体','含逆关系数','唯一观测事实','资源预检','观测事实与预训练语法重合'],rows))
    texts.append('HM-1k、WikiTopics-MT4 health来自官方归纳基准目标图，但分别与WordNet、Wikidata/CoDEx来源相关；官方归纳/OOKB类别本身也不等于源图与推理图实体完全不相交。Aristo和UMLS是新的外部知识来源，采用原公开转导划分上的跨图零样本评测，不能改称严格实体完全不相交的标准归纳基准。因此不宣称取得了三个完全独立、从未暴露的标准归纳来源。WikiTopics观测事实存在预训练直接重合；其他图的字符串零重合也不能证明语义独立。正式查询的事后质量与重合审查见 [逐图质量](results/new_graph_quality.csv)。')
    quality=table('new_graph_quality.csv')
    texts.append(md(['新图','测试事实','重复','直接观测重合','同谓词反向观测','预训练语法重合'],[[r['key'],r['test_facts'],r['duplicate_test_facts'],r['direct_observation_matches'],r['reverse_same_predicate_observation_matches'],r['syntactic_pretraining_test_fact_matches']] for r in quality]))
    texts.append('同谓词反向事实(t,r,h)是另一条合法观测事实，不等同于本轮必须同步删除的显式逆关系边(t,r+R/2,h)。上述反向结构和预训练重合按原公开划分保留，只作封存后的数据质量诊断，不删除正式查询、不重选参数，也不把它们混称为本轮选择器读取测试标签。')
    bo=read(ROOT/'results/benchmark_train_inference_overlap.json')['rows']
    texts.append('发布包源图/同主题转导邻图的额外来源审查如下；这些图从未用于阶段七训练或目标选择，且这项审查不读取正式查询。必须区分原MT4训练混合图和运行时同主题转导邻图。')
    texts.append(md(['目标','来源图版本','源实体','推理实体','实体交集','关系交集','事实交集'],[[r['key'],r['source_variant'],r['benchmark_train_entities'],r['inference_entities'],r['shared_entities'],r['shared_relations'],r['shared_facts']] for r in bo]))
    texts.append('数据参考：[INDIGO官方数据](https://github.com/shuwen-liu-ox/INDIGO)、[MOTIF官方数据接口](https://github.com/HxyScotthuang/MOTIF)、[UMLS公开数据说明](https://pykeen.readthedocs.io/en/stable/_modules/pykeen/datasets/umls.html)、[Aristo-v4原分发](https://zenodo.org/records/5942560)。下载哈希、固定提交、来源和版本限制均随观测清单保存。')
    texts.append('## 3. 多分布代理、密度与隔离\n')
    texts.append('以唯一原始正向事实为单位，保留随机生成森林、各观测关系至少一条事实及逐步剩余端点度数2，绝不偷偷放宽。同步删除显式逆边和被掩码事实的全部重复拷贝，保留未掩码事实的原有重复数；实体和关系全集保持不变。按5%、10%、15%形成嵌套物理掩码，不再以256限制删除数量。查询预算min(192,max(64,ceil(2√E)))仅限制评估查询，不能限制物理掩码；不足记录实际数量。')
    texts.append('目标适配前发现并修正了沿用阶段六时的残余图全局去重：HM有221条、ConceptNet有3条重复前向记录，不能在未掩码时额外删除。修正仅在阶段七生效；三个源图无重复且逐残余图输入哈希验证完全相同。相关新图缓存已重建并封存验证，见 [重复拷贝核验](results/proxy_multiplicity_verification.json)。原阶段六采样基线保留其原去重行为，因此该基线与新方法在相关图上的差异不能全归因于多环境；新单环境和新多环境对照共享修正后的构图。')
    texts.append('四环境共用物理残余图与共同5%隐藏查询池：A受约束自然频率，B关系均衡，C观测正事实两端度数3×3分层，D正逆关系合并模体入射支持三分层。这样控制图变化，尽量只比较查询分布。A不是不受约束的原始关系精确配额；B也无法凭空恢复被森林或稀有关系约束保护的事实。全部密度使用相同查询池，所有选择与确认事实均隐藏；全局60%/40%事实池互斥，防止一个环境的确认事实进入另一个环境选择。各环境查询数量匹配。')
    texts.append('每个残余图重新计算超边、模体、四维签名、CSR及观测结构统计。全图签名仅用于定义采样分层，绝不作为残余模型的签名。缓存以真实残余边、类型、n/R及算子版本标识，查询、模型权重和配置另有哈希。约束导致不同名义密度对应同一真实掩码时，明确保留该别名，不伪造密度变化。')
    texts.append('另对每个图和代理种子重放森林保护步骤，核对被保护事实数量及其与隐藏集合完全互斥；逐组报告关系覆盖、关系分布JS距离、端点度数、关系频率和低度数端点比例，量化约束带来的分布限制。该审查只读观测图，不改变冻结采样，见 [约束偏差表](results/observation_constraint_bias.csv) 和 [重放核验](results/observation_constraint_bias_verification.json)。')
    texts.append(md(['图谱','5%真实率','10%真实率','15%真实率','每环境选择/确认（种子6101）'],[[key]+[
        number(next(r['actual_rate'] for r in masks if r['key']==key and r['proxy_seed']=='6101' and r['density']==rate)) for rate in ['0.05','0.1','0.15']]+[
        '/'.join(next(r[k] for r in masks if r['key']==key and r['proxy_seed']=='6101') for k in ['selection_per_environment','confirmation_per_environment'])] for key in keyorder]))
    texts.append(f"选择器在文件访问拦截下运行，禁止正式查询、历史排名、下载包、旧结果和原始数据目录。代理过滤只用合法完整观测事实，包含被隐藏代理正例；没有正式valid/test真值。{verify['unit_tests']}项CPU回归、实际图泄漏断言和GPU机制检查结果见 [完整性核验](results/verification.json) 及 [测试日志](results/tests.log)。")
    texts.append('## 4. 拓扑选择与独立确认\n')
    texts.append('α固定为{0,0.1,0.25,0.5,1}，逐配置执行真实关系层与实体层前向。主要方法在选择集最大化12个“环境×密度”的最小LCB，κ=1.64；平局取较小α。仅各预声明方法的一个最终候选和α0进入确认；先封存候选计划，再计算确认，不重试。任一相关确认环境LCB≤0或选择/确认不足24/16事实，则回退α0。独立消融方法可复用各自已预声明候选的真实前向，不能把确认集当新搜索网格。')
    texts.append(r'对每个原始事实，$RR_i(\alpha)=\frac12(1/r_{i,tail}+1/r_{i,head})$，$\Delta_{ki}=RR_i(\alpha)-RR_i(0)$；$LCB_k=\overline\Delta_k-1.64\,s(\Delta_k)/\sqrt{n_k}$，样本标准差使用自由度$n_k-1$。主方法先最大化选择集的$\min_k LCB_k$，再要求同一候选在确认集全部相关环境的LCB均大于0。加权均值确认使用环境LCB的加权和，即保守地使用$\sum_k w_k SE_k$，不将重叠环境当成独立样本来缩小误差。')
    texts.append('同时保留阶段六原采样的均值/LCB、A在10%下的均值/LCB、固定α0.25、α0、随机α、等权/样本权重/长尾权重（B权重2）均值、最坏均值、仅四环境10%和仅A三密度等控制。等样本量下样本量权重理应接近等权，这是匹配设计的结果。LCB只是代理上的经验风险控制，不能保证正式MRR不下降。')
    lookup={(r['key'],r['training_seed'],r['proxy_seed'],r['method']):r for r in choices}
    eq=[r for r in choices if r['method']=='equal_mean']
    weight_changes=sum(float(r['final_alpha'])!=float(lookup[r['key'],r['training_seed'],r['proxy_seed'],'sample_weighted']['final_alpha']) for r in eq)
    texts.append(f"匹配样本数控制中，样本量加权相对等权的最终α在{len(eq)}组中有{weight_changes}组不同；不能把两者近似相同的结果解释为两份独立验证，也不能据此推断在不等样本量设计下仍无权重影响。")
    texts.append(f"规则冻结UTC：{p['created_utc']}；全部选择封存UTC：{frozen['created_utc']}。最终冻结SHA256：`{sha(ROOT/'frozen.json')}`。正式输入的打开标记均晚于这一全局封存。")
    texts.append('封存后首次正式启动因历史清单的Windows反斜杠路径在WSL中不可解析而中止，尚未打开正式查询。随后使用独立path_compat启动器，仅在SHA文件读取边界规范化路径分隔符，先核验全部历史文件，再进入原正式程序；所有冻结代码及选择记录保持原字节。兼容修复记录见 [路径核验](results/path_compat_formal.json)，不涉及参数、模型、评估公式或结果筛选。')
    texts.append('## 5. 数值稳定性与统一基线回归\n')
    native=[r for r in numeric['rows'] if r['path']=='native_fp32' and not r['tf32']]
    fixed=[r for r in numeric['rows'] if r['path']=='querywise_fp32' and not r['tf32']]
    texts.append(f"历史诊断中，原路径相对batch1的最大分数差为{max(r['score_max_error'] for r in native):.9g}；固定逐查询GEMM形状后，所测四模型、batch1/2/4/8的最大差为{max(r['score_max_error'] for r in fixed):.9g}。源图也逐一验证了批量一致性、α0与同检查点去门控输出一致、非零α改变关系表示及阶段五标准双向排名接口的一致性。")
    texts.append('最终各方法统一采用FP32、关闭TF32、逐查询实体线性投影和固定全局关系编号对；没有按MRR选择批量。保留悲观>=并列排名，不做分数舍入或容差并列。TF32、FP64/BF16局部投影及自动混合精度只是独立诊断，其支持情况和误差如实保存在 [GPU检查](results/gpu_checks.json)；不能混同为全模型FP64。')
    texts.append(md(['路径','batch','对原batch1排名变化','最大分数差','错误候选精确并列','MRR'],[[r['path'],r['batch'],r['rank_changes_to_native_batch1'],f"{r['max_score_error']:.8g}",r['pessimistic_filtered_wrong_exact_ties'],number(r['mrr'])] for r in tie['rows']]))
    texts.append('正式B0、真实B1/B2及T-4D均在这条共同数值路径上重跑。与阶段六旧数值路径的差异列于 [完整基线回归](results/numeric_baseline_regression.csv)，旧阶段六冻结α在新路径上的重放列于 [选择重放](results/stage6_frozen_selection_replay.csv)。这些数值变化不能计入拓扑适配收益。固定样本批量一致不等于所有设备、原子归约和输入都已获严格确定性证明。')
    aliases=table('aliased_graph_repeatability.csv')
    texts.append(f"利用名义密度因约束而产生的相同实际残余图，额外检查{len(aliases)}组相同图、相同查询及α的重复排名，方向排名变化总数为{sum(int(r['directional_rank_changes']) for r in aliases)}。这些是相关输入上的排名重复性检查，不是独立泛化样本；此处没有保存分数，不能由排名相同推断逐分数完全一致。见 [相同输入复验](results/aliased_graph_repeatability.csv)。")
    ak=read(ROOT/'results/alpha0_relation_kernel_audit.json')
    texts.append(f"NELL-v1的α0/B0额外事后诊断检查了{ak['weights']['common_tensors']}个共同权重/缓冲张量，不相同的有{len(ak['weights']['different_common_tensors'])}个；在固定关系对下，首个关系层线性输出最大误差为{ak['first_relation_layer_linear_max_error']:.9g}，最终关系表示最大误差为{ak['relation_representation_max_error']:.9g}。另把T-4D关系前向临时替换为官方关系前向，保持同一权重、实体前向、精度与排名定义，只作路径隔离诊断：")
    texts.append(md(['诊断路径','MRR','对B0排名变化','对B0最大分数误差','对本族冻结α0排名变化'],[[r['path'],number(r['mrr']),r['rank_changes_to_B0'],f"{r['score_max_error_to_B0']:.9g}",r['rank_changes_to_frozen_family_alpha0']] for r in ak['rows']]))
    texts.append('按关系层顺序的线性输出最大误差：'+', '.join(f'{v:.9g}' for v in ak['relation_linear_max_errors_by_layer'])+'。')
    texts.append('此检查在全部正式运行之后执行，没有替换任何冻结排名或重新选择α。α0保留阶段五CSR关系归约实现，官方B0使用其原关系实现；逐查询实体GEMM修复不能自动保证两条关系层路径逐位一致。路径诊断及其复现命令见 [α0数值审查](results/alpha0_relation_kernel_audit.json) 和README。')
    texts.append('## 6. 源图机制与预测可靠性\n')
    source=table('source_validation_methods.csv')
    texts.append('先完成三个源图×三个代理划分，共9个机制组合；此部分使用训练种子1024，另以原合法源验证前512事实作标准双向评测。源图训练种子不确定性不由这9个重复冒充，正式目标实验另运行三个既有训练种子。源代理密度、环境、α的完整指标见 [源代理表](results/source_proxy_metrics.csv)，代理—源验证对应见 [源预测表](results/source_proxy_validation_predictions.csv)。')
    texts.append('这些源验证数据曾用于旧阶段检查点选择，属于机制和设计诊断，不是新的独立泛化证据；其固定α配对区间见 [源验证统计](results/source_validation_bootstrap.csv)。')
    texts.append(md(['源图','α0验证MRR','固定0.25','阶段六LCB','A/10% LCB','鲁棒LCB','随机α均值'],[[key]+[
        number(np.mean([float(r['mrr']) for r in source if r['key']==key and r['method']==method])) for method in ['reference','fixed025','stage6_lcb','single_lcb','robust_lcb','random']]
        for key in ['wn18rr_train','codex_medium_train','fb15k237_train']]))
    texts.append('## 7. 正式主实验、强基线与消融\n')
    mainmethods=['B0','B1','B2','reference','fixed025','stage6_lcb','single_lcb','equal_mean','robust_lcb']
    texts.append('以下主结果使用事先指定代理种子6101，MRR对三个既有训练种子取均值；B0只有一份官方检查点。另两代理种子用于稳健性诊断，不据正式结果挑选最好划分。所有方法的Hits和训练种子明细见 [完整逐图表](results/P0_metrics.csv)、[逐种子表](results/per_seed_metrics.csv)。')
    texts.append('这三个种子对应既有源端门控或关系继续训练，不是三个独立预训练MOTIF骨干。阶段七没有重新训练任何检查点。')
    texts.append('同检查点α0参考与官方B0分别列出；两者不能因都关闭额外调制而直接视为完全相同的模型/算子路径。拓扑模块的配对增益一律相对同检查点α0计算，对强基线的差值另列。')
    texts.append(md(['图谱']+[NAMES[m] for m in mainmethods],[[key]+[number(next(r['mrr'] for r in gm if r['key']==key and r['method']==m)) for m in mainmethods] for key in keyorder]))
    texts.append('鲁棒LCB的逐图指标与相对真实强基线的点差：')
    texts.append(md(['图谱','MRR±种子SD','H@1','H@3','H@10','ΔB0','Δ旧B1','Δ旧B2'],[[r['key'],number(r['mrr'])+' ± '+number(r['mrr_sd']),number(r['hits_at_1']),number(r['hits_at_3']),number(r['hits_at_10']),number(r['delta_B0']),number(r['delta_B1']),number(r['delta_B2'])] for r in gm if r['method']=='robust_lcb']))
    groups=['locked','historical','external','new_prospective']
    texts.append('完整控制的宏MRR（原锁定/原历史/原外部均为历史暴露图；新增混合组另分官方归纳/OOKB和外部零样本）：')
    texts.append(md(['方法']+groups,[[NAMES.get(method,method)]+[number(next((r['mrr'] for r in macro if r['group']==g and r['method']==method),None)) for g in groups] for method in p['methods']+['B0','B1','B2']]))
    texts.append(md(['组','相对比较','ΔMRR','配对95%区间'],[[r['group'],r['reference'],number(r['delta']),f"[{number(r['lo'])}, {number(r['hi'])}]"] for r in mci if r['reference'] in ['reference','stage6_lcb','B0','B1','B2']]))
    texts.append('区间以原始事实为配对单位：先合并头尾RR，再平均固定训练种子，5000次重采样。宏区间在图内分层重采样后等权平均，不混合候选得分，不把同图的两个方向或多个种子当独立事实。区间条件于本批图、检查点和主代理划分的冻结选择；适配种子差异另列稳定性结果，不纳入该区间。它不代表对新知识来源总体的置信保证。主要逐图对比为鲁棒LCB对阶段六LCB，并报告Holm校正；其他对照和相关性为探索分析，完整区间见 [配对统计](results/paired_bootstrap.csv)。')
    qi=table('query_identity_audit.csv')
    texts.append(f"正式查询身份审查覆盖{len(qi)}图，完全重复三元组总计{sum(int(r['duplicate_rows']) for r in qi)}条；同图所有模型的查询顺序哈希一致。因此本轮逐查询行的重采样没有把完全重复事实误当成独立单位，见 [查询身份审计](results/query_identity_audit.csv)。")
    texts.append('![鲁棒相对单环境](figures/robust_vs_single_forest.png)')
    texts.append('## 8. 代理预测、分布偏移、密度与选择稳定性\n')
    texts.append(md(['范围','估计器','密度','图数','Pearson','Spearman'],[[r['scope'],r['estimator'],r['density'],r['n'],number(r['pearson']),number(r['spearman'])] for r in corr if r['estimator'] in ['stage6_legacy','equal_mean','worst_mean','worst_lcb']]))
    texts.append('固定α0.25的图级对应与已选择候选的对应是不同问题；后者存在选择条件，不能据高Pearson而忽略Spearman或ConceptNet支配。全部环境/密度及去掉ConceptNet的结果均保留；常数预测器相关性未定义，绝不当成预测成功。见 [固定参数关联](results/correlations.csv)、[已选候选关联](results/selected_correlations.csv)。图数有限、多个版本共享知识来源，图级点不是严格IID。')
    texts.append('Spearman在图数≤8时枚举全部标签置换给出双侧精确概率；例如3图完全同序的p为2/6，而非渐近近似产生的0。较大样本保留渐近概率，均属探索性。Pearson的参数化概率也不修复小图数、来源依赖或多重比较问题，不据新增3图的高相关宣布泛化成立。此处修正的是统计计算，不改相关系数、冻结选择、排名或主要MRR比较。')
    inf=table('correlation_influence.csv');influence_rows=[]
    for est,den in [('stage6_legacy','capped15%'),('A','0.1'),('equal_mean','all'),('worst_mean','all'),('worst_lcb','all')]:
        data=[r for r in inf if r['estimator']==est and r['density']==den and r['pearson']]
        if not data:continue
        most=max(data,key=lambda r:abs(float(r['pearson'])-float(r['full_pearson'])))
        influence_rows.append([est,number(most['full_pearson']),
            f"[{min(float(r['pearson']) for r in data):.3f}, {max(float(r['pearson']) for r in data):.3f}]",
            f"[{min(float(r['spearman']) for r in data):.3f}, {max(float(r['spearman']) for r in data):.3f}]",most['omitted_key']])
    texts.append('为避免只围绕ConceptNet选择性诊断，对每个估计器逐一剔除全部14个图级点，报告留一相关性范围；UMLS等新增高收益点同样接受检查。这是事后影响度诊断，不用于重选密度或方法，见 [完整留一结果](results/correlation_influence.csv)。')
    texts.append(md(['估计器','全图Pearson','留一Pearson范围','留一Spearman范围','Pearson变化最大时剔除图'],influence_rows))
    texts.append('选择α0会同时把代理和正式增益定义为0，可能机械抬高所选候选的相关性；因此另列非零候选图数，不把大量零点加一个非零图形成的高相关当成H1/H2证据。优于随机若主要来自退回强参考，也不能等同于拓扑适配产生了收益。')
    texts.append('![代理与正式对应](figures/proxy_formal_scatter.png)\n\n![关系分布差异](figures/relation_distribution_shift.png)\n\n![密度与环境收益](figures/density_environment_gain.png)')
    texts.append('五幅图均同时提供直接生成的PDF/SVG矢量版本；完整独立图注与统计口径见 [科研图注](figures/captions.md)。PNG仅作报告预览。视觉与格式审查见 [图形审计](figures/design_audit.md) 和 [矢量/字号核验](figures/verification.json)。')
    significant=sum(r.get('holm_p') and float(r['holm_p'])<.05 for r in density)
    texts.append(f"同一事实跨密度的α0.25相对收益差，已按训练种子平均后配对检验；{len(density)}组比较中Holm<0.05为{significant}组。实际密度、拓扑漂移、环境重叠及参数偏好均单独输出，不能把名义密度当真实密度，也不依据这些正式关联反向选择最佳密度。")
    ea=table('environment_parameter_agreement.csv');da=table('density_parameter_agreement.csv')
    true_density=[r for r in da if int(r['distinct_residual_graphs'])>1]
    texts.append(f"环境偏好的描述性核对：{len(ea)}组图×训练种子×代理种子×密度中，四环境均值最优α不完全相同的有{sum(int(r['distinct_preferences'])>1 for r in ea)}组，平均两两一致率{number(np.mean([float(r['pairwise_agreement']) for r in ea]))}。至少存在两个不同残余图的{len(true_density)}组密度比较中，α0.25收益严格变号的有{sum(r['alpha025_sign_changes']=='True' for r in true_density)}组。这些是相关重复下的描述计数，不是独立图数或显著性；见 [环境一致性](results/environment_parameter_agreement.csv) 和 [密度一致性](results/density_parameter_agreement.csv)。")
    texts.append(md(['选择器','回退比例','最终α0比例'],[[NAMES.get(method,method),number(np.mean([r['fallback']=='True' for r in choices if r['method']==method])),number(np.mean([float(r['final_alpha'])==0 for r in choices if r['method']==method]))] for method in ['stage6_lcb','single_lcb','equal_mean','four_env_lcb_10','three_density_A_lcb','robust_lcb']]))
    texts.append('![参数稳定性](figures/selection_stability.png)')
    ps_macro=table('macro_proxy_seed_metrics.csv')
    texts.append('三个代理划分的正式宏MRR分别如下；每个值先对三个训练种子平均，再对相应图组等权平均。它们不是三份独立知识来源实验，不能据此挑选最好代理种子。逐图结果见 [代理种子性能](results/proxy_seed_metrics.csv)。')
    texts.append(md(['选择器','图组','6101（主）','6102','6103'],[[NAMES[method],group]+[
        number(next((r['mrr'] for r in ps_macro if r['group']==group and r['method']==method and r['proxy_seed']==str(ps)),None))
        for ps in [6101,6102,6103]] for method in ['stage6_lcb','single_lcb','equal_mean','robust_lcb']
        for group in ['locked','historical','external','new_prospective']]))
    texts.append('环境间共用部分事实且跨密度查询完全相同，因此12环境并非12份独立证据。最坏环境阻断、最佳非零α的阻断环境数和删一环境诊断见 [环境支配](results/environment_dominance.csv)。高回退率只能说明本代理证据不足，不能称为已证明在目标测试上安全。')
    overlaps=table('environment_overlap.csv')
    degenerate=[key for key in keyorder if all(float(r['selection_jaccard'])==1 for r in overlaps if r['key']==key and r['proxy_seed']=='6101')]
    texts.append(f"主代理种子下，{len(degenerate)}图的四环境选择查询集合完全相同：{', '.join(degenerate) or '无'}。这是小型共同隐藏池耗尽造成的环境退化；这些图不能提供多分布选择的独立检验。分层实际覆盖和空分层见 [环境覆盖](results/environment_strata_coverage.csv)，没有用任意ID打散并列度数来制造分层差异。")
    texts.append('## 9. 子组、失败案例、资源与统计局限\n')
    texts.append('沿用阶段五固定high_degree_hub、low_relation_frequency、low_incident_motif_support掩码，只在最终分析读取；见 [原分组结果](results/archived_query_groups.csv)。高度数查询组与错误枢纽候选的得分机制不是同一事实，本阶段不据分组掉分推断消息爆炸，也不重新引入实体罚分。')
    texts.append('另列 [测试网格事后上界](results/posthoc_grid_ceiling.csv)，用于区分“网格存在收益但代理没选到”和“现有关系层配置本身没有足够上界”。它使用了正式标签，是带多次比较乐观偏差的诊断，绝不作为部署方法、重新选参或泛化结果。')
    texts.append(decision['limitations'])
    texts.append('资源预检在本机WSL约7.4GiB内存和2GiB交换空间下，设置7GiB地址空间上限，优先精确块8、必要时块2/1重建全部新图残余拓扑；只据观测与资源结果决定是否可执行，不据正式MRR剔除图。预检失败不等同于模型数学上不兼容或实测GPU显存不足。逐尝试错误与峰值见 [资源预检](results/new_graph_preflight.json)；推理时间/显存见 [运行成本](results/time_memory.csv)。显存不足时仅按冻结规则缩小批量，不删除查询。')
    texts.append('运行成本口径：正式T-4D记录包含五个α的真实前向，B0/B1/B2各为一个配置；适配记录包含全部预声明选择器及确认候选并集，图重建另列共享准备时间。因此这些是本轮研究的实际成本，不能直接把T-4D五点网格耗时当作部署单个冻结α的延迟，也不能把多个方法共享的推理重复计为每个方法的独占成本。')
    texts.append('新图的官方归纳/OOKB结果与外部零样本结果分别统计，UMLS规模小且密集，不能替代大规模未知来源验证。观察图采样不能覆盖无法删除的森林事实、极低频关系或正式任务独有的结构位置。多环境扩展覆盖不等于消除缺失机制偏移；密度与签名变化也不只是采样方差。三训练种子和三代理划分仍不足以估计知识来源总体风险，Holm调整也不修复图谱间依赖。')
    texts.append('事实共享实体、关系和逻辑规则，事实Bootstrap也是条件近似；本阶段未将其宣称为节点块或关系块重采样的替代。有限代理环境的max-min选参不是对任意分布外任务的风险保证。')
    texts.append('关系频率、端点度数与关系模体支持相互相关，环境B/C/D不是对三个独立因子的完全正交干预。代理—正式分布表分别记录关系频率JS、端点度数分桶JS、模体支持分桶JS及其均值与覆盖率；分桶界限只来自观测图。密度对照固定了查询身份，但同时改变连接度、模体和传播上下文，因此不能把所有变化解释成单一结构因子的因果效应。')
    texts.append('预注册查询预算使单环境最多约115条选择、77条确认事实，小图还可能更少；对千分级收益的检验效能有限。源验证中的正向点差仍可能被代理LCB拒绝。因此回退不能证明真实收益不存在，本轮也不能据此普遍否定全部自监督代理设计。预算和统计效能是明确限制，不能用较多相关环境或重复种子冒充更大的独立样本量。')
    pm=table('proxy_metrics.csv')
    offsets=[1.64*float(r['gain_se']) for r in pm if r['alpha']=='0.25' and r['gain_se']]
    texts.append(f"作为效能诊断，固定α0.25的有效选择单元中，1.64×经验SE的中位数为{number(np.median(offsets))}，四分之三分位为{number(np.quantile(offsets,.75))}。这是本批代理增益方差下LCB扣除的经验偏移，不是给定统计功效的最小可检测效应，也不代表正式收益的误差界。")
    texts.append('## 10. 明确回答与研究决策\n')
    texts.append(md(['假设','本轮证据结论'],[[name,value] for name,value in decision['hypotheses'].items()]))
    for i in range(1,9):texts.append(f"**Q{i}：** {decision['answers']['Q'+str(i)]}")
    texts.append(f"**最终决策：{decision['decision']}。** {decision['rationale']}")
    texts.append('复现入口与依赖见 [README](README.md)；规则见 [protocol](protocol.json)；不可覆盖的逐图选择见 [frozen](frozen.json)。此结论不修改阶段五/六，不用正式结果重新调整α网格、环境权重、密度或回退阈值。')
    (ROOT/'report.md').write_text('\n\n'.join(texts)+'\n',encoding='utf8')
    log('report_written',path=str(ROOT/'report.md'),sha256=sha(ROOT/'report.md'))
