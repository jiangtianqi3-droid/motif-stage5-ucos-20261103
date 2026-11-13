# 实验隔离与过拟合风险修复

本目录归档了 2026-11-12 完成的路线 A 修复复验。归档包含 [完整报告](artifacts/bf4548a3d51fc603c977/report.md)、[主指标](artifacts/bf4548a3d51fc603c977/main_metrics.csv)、[主要比较](artifacts/bf4548a3d51fc603c977/primary_comparisons.csv)、[三种子结果](artifacts/bf4548a3d51fc603c977/seed_robustness.csv) 和 [核验记录](artifacts/bf4548a3d51fc603c977/final_verification.json)。16 个评估组均保留逐查询排名、验证划分、选参及模型核验记录。

实验代码按实际执行版本归档。运行入口使用本机目录 `D:/motif-phase4-work/route-a-repair-20261112` 和既有 WSL 环境，复跑依赖原始图谱、历史缓存和检查点。仓库内可直接查阅归档指标及排名。进度窗口位于 `progress_window/ExperimentProgress.exe`，使用实验目录中的实时状态；本版本以允许状态文件原子替换的方式读取进度。

本目录是独立修复版本。原路线 A、历史阶段的源码、模型和结果保持原位。项目 C 盘无可用空间，因此新增代码、训练、缓存及结果均保存在此 D 盘目录。

## 已实现

- 后训练负采样只读取源图 train 真值；源验证计分只读取 train+valid。旧缓存中的 alltrue 字段不再被加载。三个种子 1024/2026/3407 沿用历史固定架构、500 步与相同正例日程，重做源验证选检查点。
- 结构选参与 LLM 校准按共享公开查询连接的整组事实隔离。两个方向、同一查询的不同答案不跨角色。每角色最多 512 条事实，所有划分和抽样固定种子、落盘核验。
- 结构网格从 61 组降至三档残差；其他参数固定。采用查询等权 MRR、配对 one-SE 简单模型规则、200 次选择稳定性重采样，不以测试集决定参数。
- LLM 按公开查询去重后均匀随机抽取，最多 64 个查询；不足 32 个去重触发查询或提示时回退为零；零残差跳过测试 API 调用。同一查询的全部目标共用一份回答，评分按查询等权。
- 主要比较明确为 R4−R1、R4−R3、R5−R4；新增事实配对符号翻转检验、统一 Holm 校正、查询/关系分组敏感性分析、三种子描述性统计，同时保留 MRR 和 Hits@10。
- 新增 ReverseThenM0 反向关系捷径对照，以及开发已见查询/未见查询、有/无观察反向事实的分层结果。
- 复用旧 B0 分数及完全相同有序候选池的确定性结构证据；候选列表不一致则重新检索。旧文件只读。新证据压缩保存，CPU 检索最多四个独立进程，检查每进程内存限制。

## 运行

环境为已有 WSL Ubuntu-24.04 和 `/mnt/d/ultra-phase1-env/env`。不要启动两个相同任务。

**推荐：双击本目录的 `Run-Experiment.cmd`。** 它会在隐藏窗口后台启动整个工作流，顺序完成代码测试、训练、训练隔离核验、GPU 小样本检查、全量评估、最终结果核验和报告生成，不需要持续看守。

任何一步失败都会停止后续步骤，原因和日志位置写入 `workflow_status.json`。再次运行同一入口会按指纹检查和断点恢复；正在运行时不会启动第二份任务。可运行 `Get-ExperimentStatus.ps1` 按需查看一次状态，无持续轮询。

`report.md` 是实验输出报告，只有 `repair_completion.json` 和 `workflow_status.json` 同时标记 `complete` 才代表全部核验完成。最终核验失败时，即使报告文件已生成，也不能当作已完成结果。

完整命令行入口为：

```bash
bash /mnt/d/motif-phase4-work/route-a-repair-20261112/run_all.sh
# 仅检查代码和缓存执行链路，不启动训练或正式实验：
bash /mnt/d/motif-phase4-work/route-a-repair-20261112/run_all.sh --check-only
```

以下为单阶段入口，仅供排查问题：

```bash
bash /mnt/d/motif-phase4-work/route-a-repair-20261112/run_training.sh
bash /mnt/d/motif-phase4-work/route-a-repair-20261112/code/run.sh scripts/run_repair.py
```

第二条命令要求三种子训练完成。支持 `--smoke`、`--skip-llm`、`--datasets`、`--families`；不同代码、依赖和协议使用不同运行指纹。恢复原命令会跳过完整组并恢复缺失文件，不覆盖历史实验。

代码位于 `code/`，训练隔离实现位于 `training/strict_data.py` 和 `training/common.py`。运行状态见 `training/training_progress.json`、`results/rerank_status.json`、`results/active_run.json`；完整报告在全量运行成功后生成 `report.md`。

API 凭证仍按原流程从进程环境或用户已有文档读取，不写入代码、协议或报告。此处的 API 调用仅用于既有图谱候选精排。

## 验证

`code/tests/test_repair.py` 覆盖留出标签扰动、标签存储防串用、连通查询隔离、文件顺序独立抽样、重复查询不增加选参权重、小校准集回退、稀疏增益精确检验及多重比较校正。另保留原 7 项测试。`scripts/check_cached_pipeline.py` 使用原 B0 分数验证完整 CPU 选参、重排与计分链路，不把该小样本检查视作方法有效性结果。

结构选参的 Filtered 计分也只允许读取结构选参子集标签，不能读取 LLM 校准子集标签；该信息流有单独的回归测试和运行审计文件。

## 无法靠代码消除的限制

本轮数据仍有历史测试暴露，三张具名测试图仍小，且来源家族不独立。已有模型/图上的重新选参不构成首次盲测。新增独立数据和锁定评估仍是后续必要实证工作；本修复不承诺统计显著提升，也不自动扩张论文结论。源验证过滤口径改变后，新旧源验证 MRR 不可直接相减。
