# 新阶段五（修正版）

独立实验目录。旧阶段和项目sources只读。**全部完成：21条训练轨迹、10,500次更新、275份目标结果，最终完整性核验通过。**

完整报告：[report.md](report.md)。主要数据为`results/macro_seed_robustness.csv`、`results/paired_bootstrap.csv`、`results/query_stratified_bootstrap.csv`及`results/summary.json`。源端真实余弦对应优于乱序，但目标6/11图乱序点估计更好；主模型在locked/historical均未超过B0。HubCal源选lambda=0。报告开头给出汇总，附录保留新旧B0数值核验。

- 协议：`protocol.json`；源训练：`pipeline.py` → `train.py`。
- 名称编码：Windows `python encode_names.py`，冻结BGE384，仅名称。
- 算子和显存检查：WSL `bash run.sh tests.py`、`bash run.sh preflight.py`。
- 运行：WSL `bash run.sh pipeline.py`，完成所有源训练后自动校准、冻结、评估、汇总和核验。
- Windows只读查看：`python status.py`，日志为`logs/pipeline.log`和`pipeline.stderr.log`。
- 独立B1-short三种子由`worker_b1.py`并行执行，算法与预算完全相同，仅进度文件独立；主流水线会按哈希校验后复用其完成记录。执行记录见`analysis/execution_parallelism.json`，日志为`logs/worker_b1.log`。
- 500步轨迹共21条：8条源网格筛选、两种自由度各增加2个种子、B1-short/U-Cos/R-Cos-Shuffle各3种子。
- 目标11图，新评估209项，另复用旧B1/B2的66份，合计275份逐模型排名。
- 源选步0/100/250/500，不使用目标验证集或目标测试效果作任何选择。
- 两种语义标量：`z_type((mean+max)/2)`和`z_type(mean_adjacent_cos)`；每层2参数，无PCA或语义投影。
- sigmoid降低语义干扰不等于枢纽严格无损；不透明掩码仅保证固定权重下语义通路为零；乱序可能因统计波动获胜，均据实报告。

`frozen.json`生成后禁止更改协议、语义或模型。出现错误写入`results/failure.json`并保留日志；仅最终核验通过后写入`results/completion.json`。
