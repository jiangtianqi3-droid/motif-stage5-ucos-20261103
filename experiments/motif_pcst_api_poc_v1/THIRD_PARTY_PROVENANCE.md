# 第三方代码来源

- G-Retriever: https://github.com/XiaoxinHe/G-Retriever ，commit 315b0ff8a206536067602fb97e77c10f4d646d5d，MIT，Copyright (c) 2024 Xiaoxin He。完整许可证见 G_RETRIEVER_LICENSE.txt。
- `src/pcst_adapter.py` 的边奖励减成本、超过成本时虚拟节点转换及映射还原，参考并改写 `src/dataset/utils/retrieval.py` 中 `retrieval_via_pcst`。本实现保留知识图谱事实 ID、方向和多关系；暂不包含原实现的文本余弦奖励。
- KRLM: https://github.com/lazyloafer/KRLM ，commit 984f7f5e890271a949c8f7b1105ea653f29eb765。仓库根目录未发现许可证；本实验未复制其代码。使用前需确认许可。
- 研究仓库: https://github.com/jiangtianqi3-droid/motif-stage5-ucos-20261103 ，主线 c0324b432b3651a3bd2d81338836475c72ebbf8b，实验基线 f61ab20f445523cadfe0308613becc83cac8b53f。直接导入该基线的 query_community_v1 中 ObservedGraph、CommunityEdgeWeighter 和 local_ppr。重算直接调用本地阶段七固定的 common.make_data、engine.model/RelationCache 和 calibration.install，文件哈希核验结果保存在 results/recovery/recovery.json。

G-Retriever 使用节点文本与查询的余弦相似度，给 top-k 节点赋递减整数奖励；边按不同余弦值分组分配递减奖励，对并列边分摊，并调整边成本。求解器参数 root=-1、clusters=1、pruning=gw、verbosity=0。普通边由求解器边索引还原，虚拟节点由节点索引还原原边；最后将节点表和 src/edge_attr/dst 边表序列化为 CSV。原实现无根且无向优化，不能直接等同于有向多关系图。

KRLM 的 main.py 实例化 KnowledgeEncoder；KGFM/models.py 明确实例化 RelHCNet 与 EntityNBFNet，配置文件使用 MOTIF、HCNet 和 IndNBFNet 命名，关系超图表征传给实体模型，实体模型输出结构表示和得分。它已经包含两阶段 MOTIF 类结构组件；当前并未复现其语言模型知识注意力，Top-K 与实体约束路径仍需继续审计。
