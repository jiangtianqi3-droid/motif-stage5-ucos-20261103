# 路线 A 最新代码与实验报告

本轮运行 `f20ab88ff89d82d8506c` 已于北京时间 2026-11-09 18:21 正常完成，覆盖四张图谱、B0 与 T-4D 两套冻结基线。四图共有 6,209 条官方测试事实，每套基线评估 12,418 个双向查询。

- [完整中文报告](rerank_v2/report_route_a_rerank.md)
- [全量主指标表](rerank_v2/artifacts/f20ab88ff89d82d8506c/main_metrics.csv)
- [配对 Bootstrap 区间](rerank_v2/artifacts/f20ab88ff89d82d8506c/paired_bootstrap.csv)
- [扩召回诊断](rerank_v2/artifacts/f20ab88ff89d82d8506c/expansion_diagnostics.csv)
- [最终排名与完整性核验](rerank_v2/artifacts/f20ab88ff89d82d8506c/final_delivery_verification.json)
- [实验代码与运行说明](rerank_v2/README.md)
- [提交范围与结果位置](rerank_v2/ARCHIVE.md)
- [Windows 独立进度窗口](progress_window/README.md)

统一边权与结构重排的效果具有图谱和基线依赖，尚未建立普遍优势；FBIngram 的 MRR 小幅提高但 Hits@10 下降。LLM 的新增收益主要出现在 WN18RR 的 T-4D 配置。报告保留全部负结果，不把算子修复等同于指标保证提升。

本轮采用用户确认的答案隔离候选协议：生成候选不读测试答案，其他测试正例仅用于离线 Filtered 计分。Grail 的官方 valid/test 已拆分，不能直接与上一阶段合并 valid/test 的主表数值混同。
