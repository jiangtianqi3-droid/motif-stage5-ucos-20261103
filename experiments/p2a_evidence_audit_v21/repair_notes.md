# 2026-11-23 关系缓存批次修复

旧正式运行 581dae1db50ccae320b5ce5163e1c2fe3ea494647cc16a2b64a93583c252c8f8 的全图检查在 100 条查询中发现 8 条未过滤排序代理不一致，涉及事实行 248、1681、2607、3172 的两个方向。每条有 2–8 个实体相邻名次变化，最大变化 2 名。目标过滤名次仍与历史一致。

修复前 V2.1 把所需关系按 ID 排序并统一两两编码。阶段七 RelationCache 则按原始 16 条事实批次依次访问尾、头方向，对首次缺失关系排序后两两编码，剩余一项时使用单关系批次。直接对比 262 个关系向量发现 58 个存在约 4e-7 至 1.1e-6 的差异，这 58 个关系在历史上均以单关系批次首次编码。上述 8 条不一致查询的关系均在这些差异关系中。

修复后的 historical_relation_groups 从原始方向化查询投影重建历史首次缺失分组。预计算阶段按该顺序和批次形状编码，再冻结全部反事实所用的全图关系表示。没有更改权重、精度、测试样本、D* 或悲观并列规则，也没有引入容差放宽。

新增回归测试覆盖原批次首次缺失分组、单关系余项、只请求部分关系时保留历史同伴关系。正式前新增全部 100 条查询的逐实体完整排序代理一致性检查；对应日志为新运行目录的 baseline_verification.json，关系分组记录为 relation_cache_schedule.json。

修复前诊断保留于 outputs/repair_diagnostic/cached_score_comparison.json、historical_relation_comparison.json 和 historical_relation_embeddings.npz。原失败目录和十条冒烟检查点保留，新代码使用独立运行 ID，防止混合修复前后的关系表示。

复核脚本为 scripts/diagnose_baseline.py 和 scripts/diagnose_relations.py。二者只读原模型与缓存，其新诊断输出位于当前实验 outputs 下。

## 修复验收

2026-11-23 已运行 scripts/verify_baseline_repair.py，全部原始 100 条查询的完整未过滤排序代理和真实目标过滤名次均精确一致，包括原失败的 8 条方向查询。样本清单哈希与旧运行相同。验收数据保存在 outputs/repair_diagnostic/repair_acceptance.json。修复后重新执行的 10 条查询冒烟已完成，12 项测试通过。
