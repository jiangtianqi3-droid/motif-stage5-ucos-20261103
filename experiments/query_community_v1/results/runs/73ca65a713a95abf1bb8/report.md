# MOTIF 查询感知强关联社区发现实验报告

本报告对应运行 `73ca65a713a95abf1bb8`。状态：冒烟完成，正式实验尚未完成。范围：冒烟测试，仅验证流程；不能用来判断方法有效性。

## 1. 核心结论与解释边界

模块已实现 A–E 全部流程，E 分为纯四维签名与实际冻结 T-4D 门控两种。下表及配对区间从原始逐查询记录生成；未预设 E 优于基线。没有接入 LLM，没有参数更新。

冒烟样本不提供方法有效性结论。

**数学限制：发送端全局度因子在行归一化时完全抵消。** 若 `w(u,r,v)=phi(r)/(1+d_out(u))^eta`，则 `P(v|u)=phi(r)/sum_(u,r′,v′)phi(r′)`。因此 E 的 PPR 转移差异来自关系重要性，发送端阻尼仅影响未归一化证据路径分数、预算选择及带权社区投影；D 中发送端全局度同样抵消，但关系局部扇出项仍起作用。不能将 PPR 变化归因为已被抵消的全局发送端度数。

以下 K=30、M=10 的 E−C 与 E−D 配对区间量化有效证据召回变化；区间跨 0 时不能声称显著改善。均为已暴露目标图上的离线诊断；多个比较未作多重校正，不构成新图泛化保证。

| 图谱 | 基线 | 方法 | Path Hit差 | 95%下界 | 95%上界 | 事实数 |
| --- | --- | --- | --- | --- | --- | --- |
| FB15k237Inductive_v1 | ppr_uniform | ppr_topology_signature | 0.0000 | 0.0000 | 0.0000 | 1 |
| FB15k237Inductive_v1 | ppr_uniform | ppr_topology_gated | 0.0000 | 0.0000 | 0.0000 | 1 |
| FB15k237Inductive_v1 | ppr_degree | ppr_topology_signature | 0.0000 | 0.0000 | 0.0000 | 1 |
| FB15k237Inductive_v1 | ppr_degree | ppr_topology_gated | 0.0000 | 0.0000 | 0.0000 | 1 |
| WN18RRInductive_v1 | ppr_uniform | ppr_topology_signature | 0.0000 | 0.0000 | 0.0000 | 1 |
| WN18RRInductive_v1 | ppr_uniform | ppr_topology_gated | 0.0000 | 0.0000 | 0.0000 | 1 |
| WN18RRInductive_v1 | ppr_degree | ppr_topology_signature | 0.0000 | 0.0000 | 0.0000 | 1 |
| WN18RRInductive_v1 | ppr_degree | ppr_topology_gated | 0.0000 | 0.0000 | 0.0000 | 1 |
| NELLInductive_v1 | ppr_uniform | ppr_topology_signature | 0.0000 | 0.0000 | 0.0000 | 1 |
| NELLInductive_v1 | ppr_uniform | ppr_topology_gated | 0.0000 | 0.0000 | 0.0000 | 1 |
| NELLInductive_v1 | ppr_degree | ppr_topology_signature | 0.0000 | 0.0000 | 0.0000 | 1 |
| NELLInductive_v1 | ppr_degree | ppr_topology_gated | 0.0000 | 0.0000 | 0.0000 | 1 |
| FBIngram_25 | ppr_uniform | ppr_topology_signature | 0.0000 | 0.0000 | 0.0000 | 1 |
| FBIngram_25 | ppr_uniform | ppr_topology_gated | 0.0000 | 0.0000 | 0.0000 | 1 |
| FBIngram_25 | ppr_degree | ppr_topology_signature | 0.0000 | 0.0000 | 0.0000 | 1 |
| FBIngram_25 | ppr_degree | ppr_topology_gated | 0.0000 | 0.0000 | 0.0000 | 1 |


## 2. P0 冻结、隔离与历史资产

- 原分支：`stage6-tsdc`；冻结提交：`5f1a91dd8a34c8d890735187caa8e7f5904fa0e9`；原归档工作区干净：`True`。
- 冻结标签：`baseline-4d-gate-frozen-20261108`；新分支：`experiment/query-community-v1`。
- Worktree：`D:\motif-phase4-work\query-community-v1-20261108`。全部新增代码、缓存、日志和结果位于其 `experiments/query_community_v1/`。
- 清单共 19538 个历史资产（41.61 GB），逐文件 SHA256 位于 `freeze_manifest.json`。冻结前未发现运行中的训练/评估。
- 本次保护验证：2017 个文件，模式=核心代码/权重/报告，全部哈希未变。

阶段一 ULTRA、阶段二/三 MOTIF、阶段四各拓扑子阶段、阶段五、阶段六两个实体算子版本与 TSDC、阶段七均已纳入清单。阶段三内核与阶段六 v2 的未提交修改及报告已另存 `preserved_changes/`，原目录不动；未将它们提交到正式冻结基线。Git 标签固定干净归档提交，外部未版本化历史资产依靠清单及保留原位置复现。

| 仓库 | 分支 | HEAD | 干净 |
| --- | --- | --- | --- |
| D:\ultra-phase1-work\ULTRA | main | 427966ad8ed60420eef034063d44f3153addff90 | True |
| D:\motif-phase2-work\MOTIF | main | 30b2ff3c64892a14f1715b3e929855fc1abd2c81 | True |
| D:\motif-phase3-work\MOTIF | main | 30b2ff3c64892a14f1715b3e929855fc1abd2c81 | False |
| D:\motif-phase4-work\github-archives\motif-stage5-ucos-20261103 | main | 173017b7c3a4d88edf73d2ce46b083a5fcb69354 | True |
| D:\motif-phase4-work\motif-stage6-entity-degree-p0p1 | stage6/entity-degree-p0p1 | deae5a3f55d13c6cbea6143f94dae4e0529d0c41 | True |
| D:\motif-phase4-work\motif-stage6-entity-degree-p0p1-v2 | stage6/entity-degree-v2 | 2f46975533503f6a5efde6de8114a996d6b8e680 | False |
| D:\motif-phase4-work\tsdc-stage6-20261103 | stage6-tsdc | 5f1a91dd8a34c8d890735187caa8e7f5904fa0e9 | True |


### 核心代码与模型 SHA256

| 文件 | SHA256 |
| --- | --- |


## 3. P1 四维签名与门控接口核查

类型序为 HH、TT、HT、TFH、TFT、HFT、HFH。原观察图含显式正逆边；令 C_H(v,r)、C_T(v,r) 为有向角色次数，U=1[C>0]，d(v) 为逆增强图发送度。二元模体 N=C_L^T C_S，Q=C_L^T diag(d^(-1/2)) C_S；三元模体沿中间关系 b 收缩，N=C_L^T F_b C_S，Q=(diag(d^(-1/4))C_L)^T F_b (diag(d^(-1/4))C_S)。

原始四通道为 [log(1+N), Q/N, RW, Q/(prod_j marginal_(type,j)(r_j))^(1/arity)]。RW 的头/尾角色质量 X_H(v,r)=sum_(v,r,t)1/d_(r,in)(t)，X_T(v,r)=sum_(u,r,v)1/d_(r,out)(u)；二元 RW=X_L^T U_S，三元 RW=X_L^T F_(b,row-normalized) U_S。它是支持度和，不保证<=1。排他度的边际在完整类型上统计，不是分块边际，也不是 log PMI；RA 与排他度在 [0,1]。

Z-score 在七类内部逐通道计算，使用总体标准差 ddof=0 与 eps=1e-6：z=(raw-mu)/(std+eps)。raw/z 形状 [E_m,4]；超边槽位 [3,E_m]，二元第三槽=-1；类型 [E_m]，means/stds [7,4]。阶段六bundle的raw实为全零占位，不能用于频次；适配器从原观察图canonical graph_key定位阶段四完整缓存，逐元素对齐超边，并核验重新标准化结果与阶段六z一致。没有完整缓存时仅用观察图重算并写本阶段缓存。graph_key包含版本、观察边、关系、N、R，不读取alltrue。

每层原始门控 g=exp(2*tanh((z·w_type+b_type+(Wq h_q)·Ws z)/2))，范围 (exp(-2),exp(2))。入射聚合按 (目标关系,模体类型) 分组，以 g/mean_group(g) 归一化。正式 T-4D 使用 alpha=.25 的原聚合与门控聚合残差混合。适配器只读加载源验证选定 checkpoint，对实际 RelHCNet.forward 注册 hook，逐 q 捕获六层原始门控并取算术均值；没有将均值冒充实际入射权重。每图 gate_audit 校验模型状态前后 SHA256 相同。

映射时使用超边所有非负槽位关系 ID，构造稀疏 R×E_m 入射矩阵；重复槽位的同一关系仅计一次。关系 ID 与实体观察边 edge_type 完全同域，逆关系为 (r+R/2) mod R。不把七类模体类型误当成实体关系标签。

阶段五 HubCal 是 score(t)−lambda*log(1+d_in(t)) 的候选得分校准（历史源选 lambda=0）。阶段六 D1 是接收端入度幂归一化，D2 是两端总度的对称幂归一化，再作 raw+alpha*(normalized−raw)，boundary 后加；原 Triton 使用存储第0行为接收端，PyG 相反，报告保留这一事实。新社区边始终按存储事实 u→v 解释，未修改原传播算子或权重。阶段七额外发送端校准也已归档，只作只读参考。

## 4. P2 精确算法定义

全局聚类使用稀疏有向边权矩阵 A，投影 A+A^T 后执行 NetworkX Louvain（Leiden 依赖未安装）。同实体对多关系仅在聚类投影求和，路径恢复始终按原始 edge_id。输出 labels/members、规模、模块度、内跨边占比与前5%度枢纽分布；无 N×N 稠密邻接。所有方法的跨社区诊断共享无权全局标签，避免因标签改变造成分母不一致。另导出 degree 带权聚类对照。

uniform：w=1。degree_aware：(1+d_out(u))^(-eta_out)*(1+d_(r,out)(u))^(-eta_rel)*(1+d_in(v))^(-eta_in)，默认 (.5,.5,0)，除以全图边权均值锚定为1；度数保留逆增强边、平行边和重复边的重数。

topo_signature：a_m=raw_freq=log(1+N_m)，m~q 定义为 q 在该超边的非空槽位直接出现。rho_q(r)=sum_(m含r且含q) a_m*sigmoid(z_excl+z_RW)/(sum a_m+1e-12)。gated_model：rho_q(r)=sum_(m含r)a_m*mean_l g_l(q,m)/(sum a_m+1e-12)。严格按epsilon分母公式，无支持关系rho=0，不伪造门控值。a_m 也可配置 uniform。

phi 对完整关系词表作 min-max 映射至 [.1,2]，常量 rho 返回1；w_topo=phi(rho_q(r))/(1+d_out(u))^.5。检查非负、有限、每条有出边行的概率和=1。反向 PPR 使用反向图重新按行归一化，不能直接使用非随机的 P^T。

稀疏截断 PPR：p_0=e_seed，p_(i+1)=(1-alpha)*p_i P+[alpha+(1-alpha)*dangling_mass]*e_seed，alpha=.2，为重启概率；2或3次迭代，不宣称收敛稳态。双端交汇按 sqrt(p_h(v)*p_t_reverse(v)) 排序，亲和度为交汇分数和。所有 <=3 跳简单有向路径通过前向两跳/反向一跳精确连接枚举，固定大小堆保留 Top-M，未按枢纽扇出提前截断。

路径强度为原始非归一化边权乘积 prod_e w_e，同时保留 prod_e P_e 供核查；因此原始强度跨权重模式量纲不同，只在同一模式内对候选排序。路径按强度先选每候选首条、再选剩余路径，所有候选共享最多50节点/100边，最后填充头部PPR局部背景边。M=5/10是每候选路径上限，查询总路径数仍受节点/边预算约束。K=10与30分别重做预算，不能直接截取不同预算下的子图冒充同一实验。

pi_h(r) 为头实体独立2-hop内、按头PPR选出的最多50节点诱导边，使用 p_h(sender)*w_e 按关系求和再归一化。零总量输出全零并保留 no_evidence。七维锚点在场率为各类型模体中全部关系槽位在该局部边集出现的比例；这是关系类型覆盖率代理，不等价于实体级模体实例计数。

## 5. P3 正确性与零泄漏

10项要求均有独立人工图测试：社区与模块度、正逆方向、平行关系、两模式枢纽抑制、跨社区3跳桥接、孤立/空图、双端交汇、两种拓扑非平凡性、固定种子、隐藏边与逆边拒绝。另覆盖概率、发送度抵消不变量、查询级预算、缺门控拒绝回退与 pi_h 不依赖候选。

| 测试 | 结果 |
| --- | --- |
| test_01_community_modularity | 通过 |
| test_02_inverse_directed_recovery | 通过 |
| test_03_parallel_relations | 通过 |
| test_04_hub_suppression | 通过 |
| test_05_cross_community_three_hop_bridge | 通过 |
| test_06_isolate_no_evidence | 通过 |
| test_07_dual_seed_intersection | 通过 |
| test_08_nontrivial_topology_both_modes | 通过 |
| test_09_seed_reproducibility | 通过 |
| test_10_zero_leakage_target_and_inverse | 通过 |
| test_probability_and_finite[uniform] | 通过 |
| test_probability_and_finite[degree_aware] | 通过 |
| test_probability_and_finite[topology_aware] | 通过 |
| test_empty_graph_safe | 通过 |
| test_sender_factor_cancels_from_row_transition | 通过 |
| test_weighted_community_supported | 通过 |
| test_no_gate_fallback | 通过 |
| test_pi_is_head_only | 通过 |
| test_budgets_are_query_wide | 通过 |
| test_real_signature_has_counts_not_raw_placeholders | 通过 |
| test_all_methods_export_json[naive_khop] | 通过 |
| test_all_methods_export_json[global_community] | 通过 |
| test_all_methods_export_json[ppr_uniform] | 通过 |
| test_all_methods_export_json[ppr_degree] | 通过 |
| test_all_methods_export_json[ppr_topology_signature] | 通过 |
| test_all_methods_export_json[ppr_topology_gated] | 通过 |
| test_paths_match_independent_multigraph_oracle | 通过 |


| 图谱 | 验证隐藏边 | 测试隐藏边 | 直接重合 | 逆边重合 | 断言通过 |
| --- | --- | --- | --- | --- | --- |
| FB15k237Inductive_v1 | 206 | 205 | 0 | 0 | True |
| WN18RRInductive_v1 | 185 | 188 | 0 | 0 | True |
| NELLInductive_v1 | 101 | 100 | 0 | 0 | True |
| FBIngram_25 | 5716 | 5716 | 0 | 0 | True |


只读观察 bundle 与原始事实逐元素一致；原始 valid/test 及其逆边只在所有检索完成后的独立离线审计中读取。候选生成从未访问隐藏答案或 filtered mask；真实答案仅用于下述指标与图例选择。证据输出逐边核对原始图及连续方向，不存在路径时明确 no_evidence。

## 6. A–E 完整诊断指标

A=naive_khop，B=global_community，C=ppr_uniform，D=ppr_degree，E3a=ppr_topology_signature，E3b=ppr_topology_gated。所有组共享观察图、候选、50节点/100边预算与固定种子。Grail沿用原协议合并归纳valid/test作为评估目标；FBIngram仅inf_test评估，inf_valid同样不能入观察图。

Path Hit 分母为全部查询，要求真实答案出现在合法候选且预算内有有效路径；候选召回另列。跨社区分母是相同无权全局标签下 c(h)≠c(t*) 的全部查询。枢纽为全图有向增强总度前5%，同度按实体ID固定顺序。扩召回分母为答案原模型确定性Top-11至30的查询，成功要求其路径强度严格大于Top-10所有负例的最大值（比只胜过一个负例更严格）。分母为0时记NA。

| 图谱 | 方法 | K | M | 查询 | 节点 | 边 | 枢纽占比 | 候选召回 | Path Hit | 跨社区数 | 跨社区召回 | 11–30数 | 扩召回成功 | 扩召回比例 | Top10结构MRR | 排名相关 | 检索秒 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FB15k237Inductive_v1 | naive_khop | 10 | 5 | 2 | 46.0000 | 100.0000 | 0.3050 | 1.0000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2381 | 0.3782 | 0.0087 |
| FB15k237Inductive_v1 | naive_khop | 10 | 10 | 2 | 46.0000 | 100.0000 | 0.3050 | 1.0000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2381 | 0.3782 | 0.0093 |
| FB15k237Inductive_v1 | naive_khop | 30 | 5 | 2 | 45.5000 | 100.0000 | 0.2860 | 1.0000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2381 | 0.3782 | 0.0111 |
| FB15k237Inductive_v1 | naive_khop | 30 | 10 | 2 | 45.5000 | 100.0000 | 0.2860 | 1.0000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2381 | 0.3782 | 0.0466 |
| FB15k237Inductive_v1 | global_community | 10 | 5 | 2 | 26.5000 | 100.0000 | 0.1536 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2679 | 0.5312 | 0.0085 |
| FB15k237Inductive_v1 | global_community | 10 | 10 | 2 | 26.5000 | 100.0000 | 0.1536 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2679 | 0.5312 | 0.0087 |
| FB15k237Inductive_v1 | global_community | 30 | 5 | 2 | 31.0000 | 100.0000 | 0.1450 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2679 | 0.5312 | 0.0110 |
| FB15k237Inductive_v1 | global_community | 30 | 10 | 2 | 31.0000 | 100.0000 | 0.1450 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2679 | 0.5312 | 0.0454 |
| FB15k237Inductive_v1 | ppr_uniform | 10 | 5 | 2 | 46.0000 | 100.0000 | 0.2033 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2000 | 0.1161 | 0.0090 |
| FB15k237Inductive_v1 | ppr_uniform | 10 | 10 | 2 | 45.5000 | 100.0000 | 0.2054 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2000 | 0.1161 | 0.0096 |
| FB15k237Inductive_v1 | ppr_uniform | 30 | 5 | 2 | 50.0000 | 100.0000 | 0.1800 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2000 | 0.1161 | 0.0135 |
| FB15k237Inductive_v1 | ppr_uniform | 30 | 10 | 2 | 50.0000 | 100.0000 | 0.1800 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2000 | 0.1161 | 0.0486 |
| FB15k237Inductive_v1 | ppr_degree | 10 | 5 | 2 | 46.0000 | 100.0000 | 0.2014 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.3667 | 0.7225 | 0.0084 |
| FB15k237Inductive_v1 | ppr_degree | 10 | 10 | 2 | 45.0000 | 100.0000 | 0.2050 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.3667 | 0.7225 | 0.0100 |
| FB15k237Inductive_v1 | ppr_degree | 30 | 5 | 2 | 50.0000 | 100.0000 | 0.2000 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.3667 | 0.7225 | 0.0127 |
| FB15k237Inductive_v1 | ppr_degree | 30 | 10 | 2 | 50.0000 | 100.0000 | 0.2000 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.3667 | 0.7225 | 0.0508 |
| FB15k237Inductive_v1 | ppr_topology_signature | 10 | 5 | 2 | 46.5000 | 100.0000 | 0.1998 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.5833 | 0.7585 | 0.0092 |
| FB15k237Inductive_v1 | ppr_topology_signature | 10 | 10 | 2 | 45.5000 | 100.0000 | 0.2032 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.5833 | 0.7585 | 0.0091 |
| FB15k237Inductive_v1 | ppr_topology_signature | 30 | 5 | 2 | 50.0000 | 100.0000 | 0.1700 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.5833 | 0.7585 | 0.0122 |
| FB15k237Inductive_v1 | ppr_topology_signature | 30 | 10 | 2 | 50.0000 | 100.0000 | 0.1800 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.5833 | 0.7585 | 0.0494 |
| FB15k237Inductive_v1 | ppr_topology_gated | 10 | 5 | 2 | 47.5000 | 100.0000 | 0.2189 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.6667 | 0.8328 | 0.0092 |
| FB15k237Inductive_v1 | ppr_topology_gated | 10 | 10 | 2 | 46.0000 | 100.0000 | 0.2133 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.6667 | 0.8328 | 0.0098 |
| FB15k237Inductive_v1 | ppr_topology_gated | 30 | 5 | 2 | 50.0000 | 100.0000 | 0.1900 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.6667 | 0.8328 | 0.0125 |
| FB15k237Inductive_v1 | ppr_topology_gated | 30 | 10 | 2 | 50.0000 | 100.0000 | 0.1900 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.6667 | 0.8328 | 0.0511 |
| WN18RRInductive_v1 | naive_khop | 10 | 5 | 2 | 16.0000 | 64.0000 | 0.1270 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.3333 | 0.5222 | 0.0035 |
| WN18RRInductive_v1 | naive_khop | 10 | 10 | 2 | 16.0000 | 64.0000 | 0.1270 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.3333 | 0.5222 | 0.0040 |
| WN18RRInductive_v1 | naive_khop | 30 | 5 | 2 | 16.0000 | 64.0000 | 0.1270 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.3333 | 0.5222 | 0.0044 |
| WN18RRInductive_v1 | naive_khop | 30 | 10 | 2 | 16.0000 | 64.0000 | 0.1270 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.3333 | 0.5222 | 0.0422 |
| WN18RRInductive_v1 | global_community | 10 | 5 | 2 | 16.5000 | 64.5000 | 0.1222 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2361 | 0.3284 | 0.0036 |
| WN18RRInductive_v1 | global_community | 10 | 10 | 2 | 16.5000 | 65.0000 | 0.1222 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2361 | 0.3284 | 0.0039 |
| WN18RRInductive_v1 | global_community | 30 | 5 | 2 | 21.0000 | 70.5000 | 0.1224 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2361 | 0.3284 | 0.0050 |
| WN18RRInductive_v1 | global_community | 30 | 10 | 2 | 21.0000 | 71.0000 | 0.1224 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2361 | 0.3284 | 0.0394 |
| WN18RRInductive_v1 | ppr_uniform | 10 | 5 | 2 | 19.5000 | 69.5000 | 0.1043 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2111 | 0.1886 | 0.0036 |
| WN18RRInductive_v1 | ppr_uniform | 10 | 10 | 2 | 19.5000 | 70.0000 | 0.1043 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2111 | 0.1886 | 0.0042 |
| WN18RRInductive_v1 | ppr_uniform | 30 | 5 | 2 | 24.5000 | 76.0000 | 0.1071 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2111 | 0.1886 | 0.0050 |
| WN18RRInductive_v1 | ppr_uniform | 30 | 10 | 2 | 24.5000 | 76.5000 | 0.1071 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.2111 | 0.1886 | 0.0416 |
| WN18RRInductive_v1 | ppr_degree | 10 | 5 | 2 | 19.5000 | 68.5000 | 0.1043 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.3333 | 0.2736 | 0.0032 |
| WN18RRInductive_v1 | ppr_degree | 10 | 10 | 2 | 19.5000 | 70.0000 | 0.1043 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.3333 | 0.2736 | 0.0037 |
| WN18RRInductive_v1 | ppr_degree | 30 | 5 | 2 | 24.5000 | 75.0000 | 0.1071 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.3333 | 0.2736 | 0.0056 |
| WN18RRInductive_v1 | ppr_degree | 30 | 10 | 2 | 24.5000 | 76.5000 | 0.1071 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.3333 | 0.2736 | 0.0440 |
| WN18RRInductive_v1 | ppr_topology_signature | 10 | 5 | 2 | 19.5000 | 69.5000 | 0.1043 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.5833 | 0.5607 | 0.0036 |
| WN18RRInductive_v1 | ppr_topology_signature | 10 | 10 | 2 | 19.5000 | 70.0000 | 0.1043 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.5833 | 0.5607 | 0.0037 |
| WN18RRInductive_v1 | ppr_topology_signature | 30 | 5 | 2 | 24.5000 | 76.0000 | 0.1071 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.5833 | 0.5607 | 0.0054 |
| WN18RRInductive_v1 | ppr_topology_signature | 30 | 10 | 2 | 24.5000 | 76.5000 | 0.1071 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.5833 | 0.5607 | 0.0418 |
| WN18RRInductive_v1 | ppr_topology_gated | 10 | 5 | 2 | 19.5000 | 69.5000 | 0.1043 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.5833 | 0.5180 | 0.0035 |
| WN18RRInductive_v1 | ppr_topology_gated | 10 | 10 | 2 | 19.5000 | 70.0000 | 0.1043 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.5833 | 0.5180 | 0.0042 |
| WN18RRInductive_v1 | ppr_topology_gated | 30 | 5 | 2 | 24.5000 | 76.0000 | 0.1071 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.5833 | 0.5180 | 0.0050 |
| WN18RRInductive_v1 | ppr_topology_gated | 30 | 10 | 2 | 24.5000 | 76.5000 | 0.1071 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.5833 | 0.5180 | 0.0432 |
| NELLInductive_v1 | naive_khop | 10 | 5 | 2 | 49.0000 | 100.0000 | 0.2449 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.6667 | 0.2532 | 0.0070 |
| NELLInductive_v1 | naive_khop | 10 | 10 | 2 | 48.5000 | 100.0000 | 0.2474 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.6667 | 0.2532 | 0.0071 |
| NELLInductive_v1 | naive_khop | 30 | 5 | 2 | 49.0000 | 100.0000 | 0.2449 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.6667 | 0.2532 | 0.0134 |
| NELLInductive_v1 | naive_khop | 30 | 10 | 2 | 48.0000 | 100.0000 | 0.2500 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.6667 | 0.2532 | 0.0680 |
| NELLInductive_v1 | global_community | 10 | 5 | 2 | 25.5000 | 51.0000 | 0.3520 | 0.5000 | 0.0000 | 2 | 0.0000 | 0 | 0 | NA（分母为0） | 0.1667 | 0.4062 | 0.0058 |
| NELLInductive_v1 | global_community | 10 | 10 | 2 | 25.5000 | 51.0000 | 0.3520 | 0.5000 | 0.0000 | 2 | 0.0000 | 0 | 0 | NA（分母为0） | 0.1667 | 0.4062 | 0.0054 |
| NELLInductive_v1 | global_community | 30 | 5 | 2 | 28.5000 | 53.5000 | 0.1714 | 0.5000 | 0.0000 | 2 | 0.0000 | 0 | 0 | NA（分母为0） | 0.1667 | 0.4062 | 0.0103 |
| NELLInductive_v1 | global_community | 30 | 10 | 2 | 28.5000 | 53.5000 | 0.1714 | 0.5000 | 0.0000 | 2 | 0.0000 | 0 | 0 | NA（分母为0） | 0.1667 | 0.4062 | 0.0540 |
| NELLInductive_v1 | ppr_uniform | 10 | 5 | 2 | 50.0000 | 100.0000 | 0.1700 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.2000 | 0.1741 | 0.0076 |
| NELLInductive_v1 | ppr_uniform | 10 | 10 | 2 | 50.0000 | 100.0000 | 0.1800 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.2000 | 0.1741 | 0.0077 |
| NELLInductive_v1 | ppr_uniform | 30 | 5 | 2 | 50.0000 | 100.0000 | 0.1500 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.2000 | 0.1741 | 0.0127 |
| NELLInductive_v1 | ppr_uniform | 30 | 10 | 2 | 47.5000 | 100.0000 | 0.1578 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.2000 | 0.1741 | 0.1279 |
| NELLInductive_v1 | ppr_degree | 10 | 5 | 2 | 44.5000 | 100.0000 | 0.1897 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.6667 | 0.7717 | 0.0068 |
| NELLInductive_v1 | ppr_degree | 10 | 10 | 2 | 44.0000 | 100.0000 | 0.1821 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.6667 | 0.7717 | 0.0076 |
| NELLInductive_v1 | ppr_degree | 30 | 5 | 2 | 40.5000 | 100.0000 | 0.1568 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.6667 | 0.7717 | 0.0136 |
| NELLInductive_v1 | ppr_degree | 30 | 10 | 2 | 40.5000 | 100.0000 | 0.1568 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.6667 | 0.7717 | 0.1312 |
| NELLInductive_v1 | ppr_topology_signature | 10 | 5 | 2 | 50.0000 | 100.0000 | 0.1600 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 1.0000 | 0.6155 | 0.0070 |
| NELLInductive_v1 | ppr_topology_signature | 10 | 10 | 2 | 50.0000 | 100.0000 | 0.1600 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 1.0000 | 0.6155 | 0.0076 |
| NELLInductive_v1 | ppr_topology_signature | 30 | 5 | 2 | 40.5000 | 100.0000 | 0.1868 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 1.0000 | 0.6155 | 0.0128 |
| NELLInductive_v1 | ppr_topology_signature | 30 | 10 | 2 | 38.5000 | 100.0000 | 0.1729 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 1.0000 | 0.6155 | 0.1282 |
| NELLInductive_v1 | ppr_topology_gated | 10 | 5 | 2 | 50.0000 | 100.0000 | 0.1700 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.5000 | 0.6031 | 0.0070 |
| NELLInductive_v1 | ppr_topology_gated | 10 | 10 | 2 | 50.0000 | 100.0000 | 0.1700 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.5000 | 0.6031 | 0.0076 |
| NELLInductive_v1 | ppr_topology_gated | 30 | 5 | 2 | 40.5000 | 100.0000 | 0.1868 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.5000 | 0.6031 | 0.0127 |
| NELLInductive_v1 | ppr_topology_gated | 30 | 10 | 2 | 39.0000 | 100.0000 | 0.1819 | 0.5000 | 0.5000 | 2 | 0.5000 | 0 | 0 | NA（分母为0） | 0.5000 | 0.6031 | 0.1285 |
| FBIngram_25 | naive_khop | 10 | 5 | 2 | 42.5000 | 100.0000 | 0.3535 | 0.5000 | 0.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.1333 | 0.3873 | 0.0477 |
| FBIngram_25 | naive_khop | 10 | 10 | 2 | 42.0000 | 100.0000 | 0.3463 | 0.5000 | 0.0000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.1333 | 0.3873 | 0.0524 |
| FBIngram_25 | naive_khop | 30 | 5 | 2 | 41.5000 | 100.0000 | 0.3352 | 1.0000 | 0.0000 | 0 | NA（分母为0） | 1 | 0 | 0.0000 | 0.1333 | 0.3873 | 0.0764 |
| FBIngram_25 | naive_khop | 30 | 10 | 2 | 39.5000 | 100.0000 | 0.3389 | 1.0000 | 0.0000 | 0 | NA（分母为0） | 1 | 0 | 0.0000 | 0.1333 | 0.3873 | 0.1396 |
| FBIngram_25 | global_community | 10 | 5 | 2 | 37.5000 | 100.0000 | 0.6747 | 0.5000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.1818 | NA（分母为0） | 0.0490 |
| FBIngram_25 | global_community | 10 | 10 | 2 | 30.0000 | 100.0000 | 0.7333 | 0.5000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.1818 | NA（分母为0） | 0.0513 |
| FBIngram_25 | global_community | 30 | 5 | 2 | 39.5000 | 100.0000 | 0.6685 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 1 | 0 | 0.0000 | 0.1818 | NA（分母为0） | 0.0788 |
| FBIngram_25 | global_community | 30 | 10 | 2 | 35.0000 | 100.0000 | 0.6958 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 1 | 0 | 0.0000 | 0.1818 | NA（分母为0） | 0.1659 |
| FBIngram_25 | ppr_uniform | 10 | 5 | 2 | 39.5000 | 100.0000 | 0.6150 | 0.5000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.1818 | NA（分母为0） | 0.0485 |
| FBIngram_25 | ppr_uniform | 10 | 10 | 2 | 32.5000 | 100.0000 | 0.6461 | 0.5000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.1818 | NA（分母为0） | 0.0505 |
| FBIngram_25 | ppr_uniform | 30 | 5 | 2 | 50.0000 | 100.0000 | 0.5700 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 1 | 0 | 0.0000 | 0.1818 | NA（分母为0） | 0.0857 |
| FBIngram_25 | ppr_uniform | 30 | 10 | 2 | 50.0000 | 100.0000 | 0.5800 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 1 | 0 | 0.0000 | 0.1818 | NA（分母为0） | 0.1786 |
| FBIngram_25 | ppr_degree | 10 | 5 | 2 | 42.0000 | 100.0000 | 0.5966 | 0.5000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.1111 | 0.3975 | 0.0520 |
| FBIngram_25 | ppr_degree | 10 | 10 | 2 | 27.0000 | 100.0000 | 0.3519 | 0.5000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.1111 | 0.3975 | 0.0538 |
| FBIngram_25 | ppr_degree | 30 | 5 | 2 | 47.0000 | 100.0000 | 0.4759 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 1 | 1 | 1.0000 | 0.1111 | 0.3975 | 0.0807 |
| FBIngram_25 | ppr_degree | 30 | 10 | 2 | 47.0000 | 100.0000 | 0.4759 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 1 | 1 | 1.0000 | 0.1111 | 0.3975 | 0.2717 |
| FBIngram_25 | ppr_topology_signature | 10 | 5 | 2 | 42.5000 | 100.0000 | 0.5995 | 0.5000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.1111 | 0.2186 | 0.0533 |
| FBIngram_25 | ppr_topology_signature | 10 | 10 | 2 | 33.5000 | 100.0000 | 0.4929 | 0.5000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.1111 | 0.2186 | 0.0524 |
| FBIngram_25 | ppr_topology_signature | 30 | 5 | 2 | 46.5000 | 100.0000 | 0.4618 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 1 | 0 | 0.0000 | 0.1111 | 0.2186 | 0.0815 |
| FBIngram_25 | ppr_topology_signature | 30 | 10 | 2 | 45.5000 | 100.0000 | 0.4385 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 1 | 0 | 0.0000 | 0.1111 | 0.2186 | 0.1742 |
| FBIngram_25 | ppr_topology_gated | 10 | 5 | 2 | 41.5000 | 100.0000 | 0.6141 | 0.5000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.1111 | 0.2186 | 0.0492 |
| FBIngram_25 | ppr_topology_gated | 10 | 10 | 2 | 31.5000 | 100.0000 | 0.5071 | 0.5000 | 0.5000 | 0 | NA（分母为0） | 0 | 0 | NA（分母为0） | 0.1111 | 0.2186 | 0.0522 |
| FBIngram_25 | ppr_topology_gated | 30 | 5 | 2 | 45.0000 | 100.0000 | 0.4889 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 1 | 0 | 0.0000 | 0.1111 | 0.2186 | 0.0807 |
| FBIngram_25 | ppr_topology_gated | 30 | 10 | 2 | 45.0000 | 100.0000 | 0.4886 | 1.0000 | 1.0000 | 0 | NA（分母为0） | 1 | 0 | 0.0000 | 0.1111 | 0.2186 | 0.1712 |


Top10结构MRR仅在真实答案已在Top10时计算，等分采用平均秩；它是候选内离线结构验证，不是完整知识图谱 filtered MRR。排名相关为结构分数排序与B0顺序的Spearman，只衡量一致性，不代表正确性。原始 logits 不可用，base_score 是原实验保存的1/(60+raw pessimistic rank)，确定性并列按实体ID；base_rank 是这个合法候选顺序。

### C/D/E 枢纽变化与 B/E 跨社区对照

| 图谱 | 基线 | 方法 | 指标 | 差 | 95%下界 | 95%上界 |
| --- | --- | --- | --- | --- | --- | --- |
| FB15k237Inductive_v1 | global_community | ppr_topology_signature | cross_hit | 0.0000 | 0.0000 | 0.0000 |
| FB15k237Inductive_v1 | global_community | ppr_topology_gated | cross_hit | 0.0000 | 0.0000 | 0.0000 |
| FB15k237Inductive_v1 | ppr_uniform | ppr_topology_signature | hub_fraction | 0.0000 | 0.0000 | 0.0000 |
| FB15k237Inductive_v1 | ppr_uniform | ppr_topology_gated | hub_fraction | 0.0100 | 0.0100 | 0.0100 |
| FB15k237Inductive_v1 | ppr_degree | ppr_topology_signature | hub_fraction | -0.0200 | -0.0200 | -0.0200 |
| FB15k237Inductive_v1 | ppr_degree | ppr_topology_gated | hub_fraction | -0.0100 | -0.0100 | -0.0100 |
| WN18RRInductive_v1 | global_community | ppr_topology_signature | cross_hit | 0.0000 | 0.0000 | 0.0000 |
| WN18RRInductive_v1 | global_community | ppr_topology_gated | cross_hit | 0.0000 | 0.0000 | 0.0000 |
| WN18RRInductive_v1 | ppr_uniform | ppr_topology_signature | hub_fraction | 0.0000 | 0.0000 | 0.0000 |
| WN18RRInductive_v1 | ppr_uniform | ppr_topology_gated | hub_fraction | 0.0000 | 0.0000 | 0.0000 |
| WN18RRInductive_v1 | ppr_degree | ppr_topology_signature | hub_fraction | 0.0000 | 0.0000 | 0.0000 |
| WN18RRInductive_v1 | ppr_degree | ppr_topology_gated | hub_fraction | 0.0000 | 0.0000 | 0.0000 |
| NELLInductive_v1 | global_community | ppr_topology_signature | cross_hit | 0.5000 | 0.5000 | 0.5000 |
| NELLInductive_v1 | global_community | ppr_topology_gated | cross_hit | 0.5000 | 0.5000 | 0.5000 |
| NELLInductive_v1 | ppr_uniform | ppr_topology_signature | hub_fraction | 0.0151 | 0.0151 | 0.0151 |
| NELLInductive_v1 | ppr_uniform | ppr_topology_gated | hub_fraction | 0.0241 | 0.0241 | 0.0241 |
| NELLInductive_v1 | ppr_degree | ppr_topology_signature | hub_fraction | 0.0161 | 0.0161 | 0.0161 |
| NELLInductive_v1 | ppr_degree | ppr_topology_gated | hub_fraction | 0.0251 | 0.0251 | 0.0251 |
| FBIngram_25 | global_community | ppr_topology_signature | cross_hit | 0.0000 | 0.0000 | 0.0000 |
| FBIngram_25 | global_community | ppr_topology_gated | cross_hit | 0.0000 | 0.0000 | 0.0000 |
| FBIngram_25 | ppr_uniform | ppr_topology_signature | hub_fraction | -0.1415 | -0.1415 | -0.1415 |
| FBIngram_25 | ppr_uniform | ppr_topology_gated | hub_fraction | -0.0914 | -0.0914 | -0.0914 |
| FBIngram_25 | ppr_degree | ppr_topology_signature | hub_fraction | -0.0374 | -0.0374 | -0.0374 |
| FBIngram_25 | ppr_degree | ppr_topology_gated | hub_fraction | 0.0127 | 0.0127 | 0.0127 |


cross_hit 的配对区间分母是全部查询，天然跨社区条件召回请读完整指标表；不得将两者混用。所有区间按原事实聚类同时重采两个方向，2000次Bootstrap，种子1024。B严格同社区截断，对跨社区尾实体必然no_evidence；是否由E实际挽回、挽回多少必须以测得的跨社区条件召回为准，不把设计结论包装成普遍性能提升。

## 7. 计算开销、断点恢复与真实案例

| 图谱 | 事实数 | 预处理秒 | 检索/离线汇总秒 | 无join缓存检索秒 | 峰值RSS字节 | 峰值CUDA字节 |
| --- | --- | --- | --- | --- | --- | --- |
| FB15k237Inductive_v1 | 1 | 15.4535 | 8.3850 | 0.0487 | 775651328 | 12608000 |
| WN18RRInductive_v1 | 1 | 10.4815 | 5.6863 | 0.0420 | 830509056 | 10976256 |
| NELLInductive_v1 | 1 | 9.3330 | 6.2100 | 0.1063 | 830509056 | 11252736 |
| FBIngram_25 | 1 | 12.6999 | 9.5810 | 0.1835 | 1144020992 | 110027776 |


平均检索耗时包含内存join复用，另列K30/M10无join缓存时间。预处理含候选文件SHA校验、Louvain与门控（缓存命中时更快）；CUDA仅冻结门控前向使用，主检索CPU稀疏。峰值RSS为本进程截至该图的高水位，非每图增量。

统一入口支持 --smoke、--limit N、--datasets；默认四图全量双向。配置与全部程序哈希决定results/runs/<fingerprint>，不同版本不覆盖；查询JSON原子落盘，重启跳过已通过校验记录；单图completion.json之后跳过该图。异常保留failure.json和完整回溯，状态写results/status.json。

真实案例采用共同布局、原始有向边和同一候选，红色为候选最强路径，背景保留全部预算内多关系边。自动选择偏好E存在跨社区连接的案例，属解释性展示；不用于估计平均效果。无合适案例时展示实际无证据结果，不合成成功案例。

### FB15k237Inductive_v1：FB15k237Inductive_v1:229:1，候选304

![真实查询子图对比](../../../figures/73ca65a713a95abf1bb8/FB15k237Inductive_v1_case.png)

| 方法 | 状态 | 跨社区 | 最强有向三元组序列 | 原边ID |
| --- | --- | --- | --- | --- |
| naive_khop | ok | True | [[367, 54, 548], [548, 234, 304]] | [502, 3380] |
| global_community | no_evidence | True | 无路径 | [] |
| ppr_uniform | ok | True | [[367, 54, 548], [548, 234, 304]] | [502, 3380] |
| ppr_topology_signature | ok | True | [[367, 54, 548], [548, 234, 304]] | [502, 3380] |
| ppr_topology_gated | ok | True | [[367, 54, 548], [548, 234, 304]] | [502, 3380] |


### WN18RRInductive_v1：WN18RRInductive_v1:208:1，候选698

![真实查询子图对比](../../../figures/73ca65a713a95abf1bb8/WN18RRInductive_v1_case.png)

| 方法 | 状态 | 跨社区 | 最强有向三元组序列 | 原边ID |
| --- | --- | --- | --- | --- |
| naive_khop | no_evidence | True | 无路径 | [] |
| global_community | no_evidence | True | 无路径 | [] |
| ppr_uniform | ok | True | [[208, 9, 144], [144, 1, 387], [387, 0, 698]] | [2912, 1516, 1473] |
| ppr_topology_signature | ok | True | [[208, 9, 144], [144, 1, 387], [387, 0, 698]] | [2912, 1516, 1473] |
| ppr_topology_gated | ok | True | [[208, 9, 144], [144, 10, 387], [387, 0, 698]] | [2912, 1880, 1473] |


### NELLInductive_v1：NELLInductive_v1:112:0，候选1

![真实查询子图对比](../../../figures/73ca65a713a95abf1bb8/NELLInductive_v1_case.png)

| 方法 | 状态 | 跨社区 | 最强有向三元组序列 | 原边ID |
| --- | --- | --- | --- | --- |
| naive_khop | ok | True | [[88, 2, 1]] | [108] |
| global_community | no_evidence | True | 无路径 | [] |
| ppr_uniform | ok | True | [[88, 2, 1]] | [108] |
| ppr_topology_signature | ok | True | [[88, 17, 1]] | [1583] |
| ppr_topology_gated | ok | True | [[88, 17, 1]] | [1583] |


## 8. 后续方案一与方案三接口

方案一可直接消费 schema_version=query-community-1.0 的query_info、candidate_evidence和subgraph。包含query_id、h/q、base_score/base_rank/score_kind、社区标签、双端亲和度、max_path_weight、meeting_nodes、连续有向triple/edge_id/relation序列、每边rho/degree_weight/weight/transition_probability与bridge标记。仅导出数据，本轮不实现分差阈值、不调用任何LLM API。将来分差触发阈值须在合法开发集上定义；原始logit缺失时不能将reciprocal-rank差当作logit置信差。

方案三读取独立于候选的pi_h[R]与anchor_presence_7[7]；这些是只读观测结构特征，可输入后续g(h,q,m)原型。新模块没有修改现有RelHCNet、EntityNBFNet训练/推理权重，没有将本轮证据排名作为新模型训练标签。no_evidence应原样保留以支持回退原模型，不能编造路径。

## 9. 可复现文件索引

- 本次协议、程序哈希、依赖环境：`results/runs/73ca65a713a95abf1bb8/protocol.json`与`environment.json`。
- 原始逐查询证据、离线行、完整指标、配对区间及资源记录：`results/runs/73ca65a713a95abf1bb8/<dataset>/`。
- 冻结清单：`freeze_manifest.json`；保护验证：`results/isolation_core.json`或`isolation_full.json`；单测：`results/unit_tests.xml`。
- 模型门控缓存与参数不变核验：`cache/gates/*_gate_audit.json`。
- 原始可视化伴随JSON保留每个方法完整有向子图与路径，方便独立核查。
