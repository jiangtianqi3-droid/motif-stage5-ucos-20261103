# MOTIF-Topology-Retrieval-API-PoC-v1

实验在两个已暴露的归纳图谱上比较候选固定后的 PPR、普通 PCST、模体感知 PCST、随机证据和无证据提示。冻结主干为阶段七 B2+D* 的种子 1024。B0 单独恢复，用于历史候选协议核对。每个查询固定未过滤 Top-30，仅允许前 12 个实体改变顺序。各组的候选别名排列完全一致。

## 本地运行

Windows 上的路径通过 WSL Ubuntu-24.04 映射。以该目录的 WSL 路径替换 `<experiment>`。

```bash
bash <experiment>/scripts/run.sh -m src.recovery
bash <experiment>/scripts/run.sh -m unittest discover -s tests -v
bash <experiment>/scripts/run.sh -m src.pipeline --stage retrieval
bash <experiment>/scripts/run.sh -m src.pipeline --stage smoke
```

配置模板为 `config/api.example.env`。放在实验根目录的 `.env.local` 由 Git 忽略。PowerShell 后台入口为 `scripts/launch_background.ps1`。运行相同入口即可从缓存恢复。默认 `pilot` 要求同一模型、同一采样的冒烟记录已通过。`-Stage all` 在统一进程中顺序执行各阶段。状态和日志写入 `results/status.json` 和 `results/logs/`。以 `python scripts/status.py` 查看。

API 账本记录预检查、每次尝试和成功响应。有效响应和弃权均可从本地 SQLite 复用。网络断开或超时造成交付不明时，预算仍保留，当前缓存键停在需人工核对的状态。重新启动不会自动为该请求付费重发。请求、输入和输出限额分别为 300、1,000,000 和 60,000。保守融合只接受有效排序且引用至少一条选中候选路径上的事实。直接重排和保守融合共享 API 响应。

## 数据与结果

`src.recovery` 核验阶段七源码、检查点、观察图、历史批次及过滤协议。它从冻结观察图和已投影的 (锚点，关系) 重算全候选倒数排名，核对完整文件哈希及原始过滤名次。固定抽样清单位于 `results/sample_manifest.json`。无 API 诊断每图 100 条方向查询。冒烟每图两条事实，正式实验每图十二条不同事实，共 48 个方向查询。抽样基于行号和固定种子，独立于答案和检索结果。

PCST 使用原始事实索引维护方向和平行边。关系模体由已冻结的 MOTIF 关系超图签名映射。分阶段候选覆盖先保留可达候选路径，再用 PCST 扩展并执行容量限制。头尾查询统一采用显式逆关系。名次指标沿用原过滤集合和悲观并列规则，同时保存候选的完整写回顺序。

完整矩阵、缓存、账本和逐查询结果留在本地。指标入口为 `results/retrieval_metrics.csv`、`results/api_metrics.csv` 和 `report_poc_v1.md`。来源见 [THIRD_PARTY_PROVENANCE.md](THIRD_PARTY_PROVENANCE.md)。
