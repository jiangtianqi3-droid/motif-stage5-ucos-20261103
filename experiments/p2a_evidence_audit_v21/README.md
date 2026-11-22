# P2-A Stage 1 · V2.1

本阶段仅审计 P1 检索证据，并把选中观察事实送入冻结 B2+D* 的第二阶段实体传播。关系表示与全图 D* 的缩放、统计、均值锚点固定。最终维持全实体过滤排名。

Windows 统一入口：`python scripts/run_all.py`。只跑前 10 条冒烟：`python scripts/run_all.py --smoke-only`。独立后台：`scripts/launch_background.ps1`。查看状态：`python scripts/status.py`。

所有运行按配置、源码、依赖和冻结资产标识定位 `outputs/<run_id>/`。逐查询原子检查点确保断点续跑不混合输入。运行脚本自动执行测试、冒烟、100 查询、统计、七幅 PNG/SVG 图及 `report_p2a.md`。主要 CSV 同步到 `outputs/propagation`、`outputs/aggregation`、`outputs/sufficiency` 和 `outputs/necessity`。

路径枚举与效用直接复用 P1，固定聚合温度 0.08。实体级高阶模体实例缺少历史定义，字段保留 NA。模型保真度按规范原事实预算恢复完整逆向消息记录。随机删除仅在同一个 Gq 里分层匹配，按配置记录每次放宽和重叠。

复用 G-Retriever PCST 的来源、MIT 声明及改造见相邻 `motif_pcst_api_poc_v1/THIRD_PARTY_PROVENANCE.md`。原始源码和所有历史权重保持只读。
