# 路线 A 代码和结果归档

本轮提交保留原始运行 `f20ab88ff89d82d8506c` 的代码、中文报告、完整数值结果和少量代表案例。代码中被运行协议记录的 13 项源码及启动脚本与冻结指纹一致，文件字节和结果来源见 [提交清单](artifacts/f20ab88ff89d82d8506c/publication_manifest.json)。

归档目录为 `artifacts/f20ab88ff89d82d8506c/`：

|内容|位置|
|---|---|
|中文报告|本目录 `report_route_a_rerank.md`，归档目录另存原始报告|
|完整 M0 与 R1–R5 指标|`main_metrics.csv`；匿名图不运行 R5|
|2,000 次事实配对 Bootstrap|`paired_bootstrap.csv` 及每组 `summary.json`|
|Top 11–30 扩召回分解|`expansion_diagnostics.csv` 及每组 `summary.json`|
|每图、每套基线的原始最终排名|`<dataset>/<family>/ranks.npz`|
|每个查询的事实索引、方向和各方法排名|`<dataset>/<family>/per_query_ranks.jsonl`|
|验证网格、选中参数和 LLM 校准|每组 `selected_structure.json`、具名图的 `selected_llm.json`|
|模型、算子、数据划分和内存核验|每组 `model_audit.json`、`operator_audit.json`、`split_leakage_audit.json`、`completion.json`|
|报告中列出的代表案例|具名图每组 `cases/` 中按文件名字典序选取的前四例|
|冒烟、API 回放、进度窗口检查|`validation/`|

`ranks.npz` 中每个方法的数组按官方测试事实索引依次存储两个方向，方向 0 为预测尾实体，方向 1 为预测头实体；逐查询 JSONL 提供对应关系。可用 `../completion_review/verify_results.py` 在本机原始结果目录独立复算指标，已完成的核验记录为 `final_delivery_verification.json`。

此次精简提交不包含完整逐查询路径树、全实体 logit 缓存、LLM 响应缓存、模型权重、运行日志和包含凭证的用户提示词文档。它们继续保留在本机原始目录，未移动或删除。归档中的代表案例是已经完成的离线分析材料，不参与选参或构造候选。

复跑依赖原有只读图谱数据、拓扑签名、历史模块和冻结模型，位置及环境要求沿用代码说明。调用 LLM 前需自行设置环境变量，仓库不保存明文 API Key。独立进度窗口保留源码和本机编译的 Windows 程序，构建入口为 `../progress_window/BuildProgress.ps1`。
