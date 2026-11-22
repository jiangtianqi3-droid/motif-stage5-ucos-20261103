# V2.1 Stage 1 资产审计

检查时研究仓库位于 D:/motif-phase4-work/motif-pcst-api-poc-20261117-codex/motif-research。原工作区干净，从 P2 提交 37a28dff3569214fa28308ae408606e9bb160a0c 创建 experiment/p2a-evidence-audit-v21。

P1 已完整运行，运行 ID 为 b2273b01f095035cf925fd78ab456213ecf706d21e90c130024c41fc458997b4，源码提交 180f77feccfea1f7dfc26a86cb2162073444df55。P2 已完成，运行 ID 为 d2381c0713f890d957476797daf4c62285ddea9d2b66fd27b1c1ee38435a8fd2。未发现这两个任务的活跃工作进程。

## 已定位资产

- 冻结 B2 模型在 D:/motif-phase4-work/topology-semantic-stage4-20261102/checkpoints/B2_s1024_step1000.pth。预期 SHA256 为 1e85a7e85b57afa14b5b79813bab85c0fd075323b16ac05a5948f200c2aed790。启动前核对文件、严格加载 state_dict 并核对完整参数哈希。
- D* 沿用阶段七 calibration.py 的 D-Sender，gamma=0.75，alpha=0.20，anchor=true。common.make_data、engine.model 和 runtime/motif 由历史 run_identity.json 校验。
- FBIngram_25 完整观察数组与协议数据在 D:/motif-phase4-work/motif-stage6-entity-degree-p0p1-v2/data/FBIngram_25.npz。独立观察图和离线标签快照在 experiments/query_community_v1/artifacts/inputs/FBIngram_25/，哈希通过正式清单核对。
- R0–R5 的实际事实 ID、候选、路径、原分数和查询行在 P1 outputs 对应的 results/<run_id>/runs/expanded/FBIngram_25/k12_b32/，共 500 条事实的两个方向。
- B0.rrf.npy 和 B2_s1024+Dstar.rrf.npy 已由前次 PoC 按原批次重算，完整矩阵与历史哈希一致。它们是全实体 1/(60+悲观原始名次)。B0.npz 与 B2_s1024+Dstar.npz 包含 ranks、triples、degree、alpha，前者只存目标过滤名次，无法单独导出候选。
- 候选生成、实体/关系原始编号、头尾转换、GraIL 和 Ingram 过滤集合均已恢复。本次只使用 FBIngram_25 的原测试事实和历史过滤集合。

## 本次必须重新计算

全图关系向量、全图 D* 的 a/b/z 张量及原始均值锚点需从冻结模型在真实全图预计算，保存在新实验目录。充分性和必要性的实体传播结果此前未运行，本次真实计算。模型完整加载、逐次固定统计、全实体空间、完整观察图逐查询名次、重复前向确定性均为冒烟验收条件。

## 技术定义核对

历史 P1 按最多 32 条有向观察记录选择证据。本次保持这些选中记录，审计路径时仅使用原选中方向。用于冻结模型保真度时按规范正向三元组去重，再恢复完整观察图中的正向、逆向和重复消息记录。报告同时列唯一事实数及消息边数，随机对照匹配唯一事实数。

历史 Triton 在 edge_index[1] 发出消息，在 edge_index[0] 接收消息。calibration.factors 以 row 端计数，完整观察图具有逐条逆向配对，两端度数相等。保存原 inc、sender_inc、a、b、z 与 float64 原始全局均值，按历史 float32 锚点参与传播。所有子图复用这批张量，保留全实体空间。

新增实体级高阶模体实例定义未在 P1 暴露，本阶段不实现新的模体接地算法。该字段记录 NA；已有有向简单路径实例与关系模板分别精确计数。本次不增加训练或超参数选择。
