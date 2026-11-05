# 阶段六 TSDC 复现

本目录独立于阶段五全部资产；协议允许目标观测图自监督选参，不使用目标 valid/test 标签或梯度。目标图均为历史暴露图上的冻结协议复验。最终结论以 [report.md](report.md) 和 [verification.json](results/verification.json) 为准。

## 资产和依赖

- 上游归档提交：`173017b7c3a4d88edf73d2ce46b083a5fcb69354`；独立分支 `stage6-tsdc`。
- 依赖清单及旧 B1/B2 完整恢复：[dependency_audit.json](results/dependency_audit.json)。旧 B1/B2 不是 T-4D/B1-short 的替身。
- 沿用既有 WSL Python3.9/PyTorch2.1.0+cu118/PyG2.4.0/Triton2.1.0/SciPy/NumPy 环境，不更新包。
- ML 环境没有 matplotlib；图形脚本单独使用本机**已有的** Windows Python/matplotlib，`--plot-python` 可替换。没有向原环境安装新依赖。
- `run.sh` 从自身位置确定项目根目录，不使用原阶段五 run.sh。模型/历史/源图/官方检查点路径均由参数提供；CUDA、编译器及临时目录可由 `TSDC_ENV`、`TSDC_CUDA_HEADERS`、`TSDC_CC`、`TSDC_CXX`、`TSDC_TMPDIR` 覆盖。

## 从原始归档开始

先在独立副本中工作，不运行阶段五 initialize.py。下面的路径记录本次实际运行环境，可整体替换：

```bash
TASK=/mnt/d/motif-phase4-work/tsdc-stage6-20261103
ARGS=(
  --stage5 /mnt/d/motif-phase4-work/github-archives/motif-stage5-ucos-20261103
  --history /mnt/d/motif-phase4-work/topology-semantic-stage4-20261102
  --source-root /mnt/d/motif-phase4-work/topology-stage2-20261102
  --official /mnt/d/motif-phase3-work/MOTIF/ckpts/motif_3g.pth
)
# 一次性依赖哈希审查（Windows Python 也可以运行）
bash "$TASK/stage6/run.sh" -m stage6.audit \
  --stage5 "${ARGS[1]}" --history "${ARGS[3]}"
# 只读清洗观测成员。不调用含标签的 common.downstream。
bash "$TASK/stage6/run.sh" -m stage6.prepare_observed \
  --stage5 "${ARGS[1]}" --history "${ARGS[3]}" --source-root "${ARGS[5]}"
# 首次运行记录既有ML环境；已有冻结记录时保留原快照。
if [ ! -f "$TASK/stage6/results/environment.json" ]; then
  bash "$TASK/stage6/run.sh" -m stage6.environment
fi
# 测试 → 源代理 → 实际密度诊断与源验证 → 目标适配 → 全局冻结 → 正式评测 → 分析报告
bash "$TASK/stage6/workflow.sh" "${ARGS[@]}"
```

第一次登记的 protocol.json 与最终 frozen.json 使用独占创建，不能覆盖。中断后可重运行工作流，已完成文件按协议/图/模型/查询/排名哈希核验后复用，不重选更好的随机种子。改变协议应建立新实验目录，保留本轮结果。

## 分步执行

```bash
bash "$TASK/stage6/run.sh" -m stage6.run tests "${ARGS[@]}"
bash "$TASK/stage6/run.sh" -m stage6.run source "${ARGS[@]}"
bash "$TASK/stage6/run.sh" -m stage6.run source-validation "${ARGS[@]}"
bash "$TASK/stage6/run.sh" -m stage6.run adapt "${ARGS[@]}"
bash "$TASK/stage6/run.sh" -m stage6.run freeze "${ARGS[@]}"
# 必须另起进程。适配进程启用拒绝正式标签/排名的访问审计。
bash "$TASK/stage6/run.sh" -m stage6.run formal "${ARGS[@]}"
bash "$TASK/stage6/run.sh" -m stage6.run report "${ARGS[@]}" \
  --plot-python /mnt/e/Python/Python39/python.exe
```

也可使用 `stage6.run all` 执行全部步骤；每一步由独立子进程执行，保证适配进程的文件隔离不会延续到正式评测。workflow.sh 额外逐阶段保存日志。

## 数据隔离与原始排名

`observations/*.npz` 只有 edge_index/edge_type/n/R，manifest 带来源与哈希；观察事实与编号元数据分别记录。原 external observed.npz 及源缓存含其他成员，清洗脚本只按名字读取四个合法成员。代理进程无法打开原含标签文件、valid/test 文本、正式 evaluation 文件和固定查询掩码。

`adaptation/`、`source_diagnostics/` 保存划分、选择集全网格、每个独立消融的单个胜出候选确认分数、参数和回退。未经确认评估的候选在 NPZ 中为 NaN、JSON 中为 null；不是缺失实验，更不能拿这些值参与重新选参。确认事实及逆边在选择和确认前均已删去。`source_diagnostics_before_confirmation_fix/` 是目标适配前发现并修正的开发版本诊断，单独保留，不参与结论。

`evaluation/*__T-4D*.npz` 存完整正式网格排名；正式网格只能用于事后检验代理预测能力，不能回写选择配置。`triples` 保持阶段五 `(h,t,r)` 顺序，`queries` 为适配器内部 `(h,r,t)`，`ranks` 最后一维顺序为尾/头。模型调用遵循阶段五 `(h,t,r)`。

证据数组列为：alpha、原始查询索引、方向(0尾/1头)、真候选ID、最高分错误候选ID、真全局log度、真关系log度、错误全局log度、错误关系log度、真分数、错误分数、与真分数精确并列的候选数。错误候选经合法 filtered mask 排除其他真值。

## 统计与交付

- [proxy_metrics.csv](results/proxy_metrics.csv)：全候选选择集与获准确认集；未评确认指标保持空值。
- [selection_decisions.csv](results/selection_decisions.csv)：99次目标适配、独立消融、回退、哈希、时间和显存。
- [P0_metrics.csv](results/P0_metrics.csv)：固定主代理种子的逐图/训练种子正式结果。
- [ablation.csv](results/ablation.csv)：三个代理种子的全部正式对照，不按正式表现择优。
- [paired_bootstrap.csv](results/paired_bootstrap.csv)：5000次原始事实配对区间；双方向和训练种子先配对平均，不独立抽方向。
- [query_groups.csv](results/query_groups.csv)、[hub_diagnostics.csv](results/hub_diagnostics.csv)：阶段五固定查询组及锚实体/真候选/错误枢纽分开诊断。
- [baseline_reproducibility.csv](results/baseline_reproducibility.csv)：恢复的真实 B0/B1/B2 与历史排名逐方向对照。
- [degree_matched_control.csv](results/degree_matched_control.csv)：相同正/负系数的全局与查询条件公式固定对照；登记时点及范围见 [matched_global_protocol.json](matched_global_protocol.json)，不参与主选择。
- [all_fixed_grid_metrics.csv](results/all_fixed_grid_metrics.csv)：四类检查点的完整正式固定网格；不能据此回写目标选参。
- [b0_fixed_relation_diagnostic.json](results/b0_fixed_relation_diagnostic.json)、[b0_operator_trace.json](results/b0_operator_trace.json)：固定关系表示后的批量数值定位，主B0结果保持首次固定运行。
- [time_memory.csv](results/time_memory.csv)：代理、共享图准备、正式网格成本与 PyTorch peak allocated。

图组宏平均等权，配对区间只反映固定图集/检查点条件下的查询抽样误差。多重探索比较未作显著性校正，11图不独立。256条上限使源图名义掩码率可能饱和，不能把相同划分当独立的密度敏感性证据。LCB不是正式任务非劣效保证。
