# 路线 A 统一拓扑边权与候选重排

独立目录，历史 `d48ae432a2533cdaaec0` 及阶段一至七数据、源码与模型均作为只读依赖。本目录不使用任何 handoff 状态。当前项目保存源码；运行缓存、日志和结果放在 `D:/motif-phase4-work/route-a-rerank-20261109`，避免占用空间不足的C盘，可用 `ROUTE_A_OUTPUT` 覆盖输出位置。

运行环境复用 WSL Ubuntu-24.04 的 `/mnt/d/ultra-phase1-env/env`。所有新增缓存、编译产物、日志和结果写入上述独立D盘输出目录。

```bash
bash run.sh -m pytest tests -s -q
bash run.sh scripts/run_route_a_rerank.py --smoke
bash scripts/launch_background.sh
# 完全相同命令使用内容指纹目录断点续跑。
```

支持 `--skip-llm`、`--datasets`、`--families`。冒烟每张图、每套基线、每个 split 检索五条查询，不形成正式有效性结论。正式实验测试全部事实的双向查询；验证集无标签均匀抽样最多256事实，LLM残差校准最多64触发查询。

用户已确认答案隔离协议：生成候选只使用观察图（测试时额外可过滤合法valid正例），测试正例仅用于离线全实体Filtered计分。这种Top-30候选池不是先按测试真值过滤后取Top-30；报告单列差异。官方valid/test拆开，历史合并口径不冒称同一测试集。

LLM配置读取 `LLM_API_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL_NAME`、`LLM_MAX_CONCURRENCY`、`LLM_TIMEOUT_SEC`。未给Key环境变量时，从用户指定的原始提示词文档读入环境，仅在进程内保留，绝不把Key写入文件或日志。默认deepseek-flash、最多4并发、30秒超时、初次失败后最多3次指数退避重试。

输出目录内的 `results/rerank_status.json` 是实时状态。每图/模型使用独立进程，按操作系统记录的峰值RSS检查1.8GB限制，超限停止并保留断点；证据每次只读取单查询，PPR矩阵按单关系/方法释放。证据、响应和逐查询排名均落盘；拒答或格式失败精确保留结构结果。发生网络持续失败会停止并保留断点，不冒称R5完成。

结果按代码/协议指纹写入 `results/runs/<fingerprint>/`。全量完成自动产生 `report_route_a_rerank.md`、完整指标、2000次事实配对bootstrap、扩召回与LLM诊断、真实案例和历史文件哈希核验。全量后台启动后不持续轮询。
