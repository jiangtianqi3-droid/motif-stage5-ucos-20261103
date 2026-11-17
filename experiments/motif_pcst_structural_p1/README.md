# MOTIF PCST P1

结构实验，配置禁用 LLM、API、KRLM 及主干训练。以 B2+D* 种子 1024 的未过滤排序为主干，比较 R0–R5。用冻结的 Stage-7 观察图和 POC 恢复资产，M5 同行对照来自历史 universal 11g 三种子实验。

WSL 快速验证：`bash scripts/run.sh -m p1.run --mode quick`。Windows 后台运行：`scripts/launch_background.ps1`。复跑根据配置、源码和历史输入哈希定位同一运行目录，并复用已经原子落盘的查询。结果在 `results/<run_id>/runs/<phase>/<dataset>/`，包括预测、检索、固定容量结构记忆和诊断。状态为 `results/status.json`，中文报告为 `report_structural_p1.md`。

复用 PoC 的 G-Retriever MIT 适配器、邻居检索、排序约束和 Stage-7 核验逻辑，复用查询社区的稀疏 PPR 与观察图数据类。P1 消融、路径拟合、证据评分和记忆导出位于 `p1/`。历史分支和输出不参与写入。设备与源协议由冻结资产核验。
