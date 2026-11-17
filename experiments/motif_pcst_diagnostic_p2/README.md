# P2 拓扑证据到预测的影响诊断

P2 复用 P1 冻结检查点、增广观察图、完整未过滤候选、PCST 证据、路径评分和过滤评测。先独立重放 R2–R5，比较边、路径特征、分数和候选顺序。再在源域匹配遮蔽候选、锁定多路径与模体消融的评分及融合。最后比较四张既有归纳式基准上的原始 B0、冻结 B2+D*、历史三种子 M5 和 P2。没有 LLM 或主干训练路径。

- WSL 快速审计：`bash scripts/run.sh -m p2.run --mode smoke`。
- Windows 后台：`scripts/launch_background.ps1`。整个任务使用实验锁，按运行哈希、阶段和图谱恢复。
- 状态：`python scripts/status.py`。指标目录：`results/`。中文报告：`report_structural_p2.md`。
- P1 的 G-Retriever MIT 适配器、查询社区 PPR、历史过滤规则均按原实现复用。P2 的候选独立路径来自当前选中的真实事实集合。
