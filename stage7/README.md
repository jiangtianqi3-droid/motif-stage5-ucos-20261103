# 阶段七：分布鲁棒的目标观测图自监督拓扑适配

本目录独立于只读的阶段五、阶段六。仅使用既有T-4D门控权重进行推理时α选择，不训练权重，不使用实体校准或语义编码器。实验结果与研究结论见 `report.md`；最终完成情况以 `results/verification.json` 为准。

## 数据与模型

- 保持阶段五根目录及 `stage6/` 研究资产不变，SHA256快照见 `preserved_assets.json`。
- 重用 `stage6/observed_loader.py` 的封闭观测数据类型、`stage6/proxy_graph.py` 的拓扑/CSR构建器和 `stage6/models.py` 的真实检查点加载器。
- 官方B0、阶段五T-4D、已恢复的历史B1/B2来源和权重哈希见只读 `stage6/model_manifest.json`。α=0引用同一T-4D检查点，不能称为旧B1。
- 新数据只将观测事实写入 `observations/*.npz`。下载包与观测词表分开保存，选择进程禁止访问下载包、历史排名和正式查询。
- 新数据的历史暴露审查、预训练重合、资源兼容性分别见 `results/historical_exposure_search.json`、`results/pretraining_observation_overlap.json`、`results/new_graph_preflight.json`。官方归纳/OOKB划分与外部转导划分上的跨图零样本评测分别统计；不把HM的实体重合情形称为严格实体不相交归纳。

## 环境

沿用WSL Ubuntu-24.04 + Python3.9.23 + PyTorch2.1.0/cu118 + PyG2.4.0 + Triton2.1.0 + NumPy1.26.4 + SciPy1.13.1，RTX3060 Laptop 6GB。没有新增Python依赖。Windows原有Python负责网络下载和最终统计/制图：Python3.9.10、NumPy2.0.2、SciPy1.13.1、Matplotlib3.9.4；与WSL推理环境分别记录于results/analysis_environment.json和figure_environment.json。

`run.sh` 从自身位置确定工程路径。可通过 `DRTA_ENV`、`DRTA_CUDA_HEADERS`、`DRTA_TMPDIR`、`DRTA_CC`、`DRTA_CXX` 修改既有环境路径。全部实验入口支持 `--stage5`、`--history`、`--official`、`--observations`、`--source-root`、`--batch-size`；小批量用于诊断，正式各方法使用同一冻结推理路径。

## 顺序与复现

在工程根目录运行；下列示例是WSL命令。`audit`、协议、选择计划和冻结文件采用独占创建，防止覆盖。复现应在新的阶段七输出副本中执行，保留原有实验目录以供核对。

```bash
bash stage7/run.sh -m stage7.run audit
bash stage7/run.sh -m stage7.run prepare
bash stage7/run.sh -m stage7.exposure_search
bash stage7/run.sh -m stage7.named_overlap
bash stage7/run.sh -m unittest discover -s stage7/tests -v
bash stage7/run.sh -m stage7.run numeric
bash stage7/run.sh -m stage7.run mechanism
bash stage7/after_mechanism.sh
```

Windows网络下载可使用原有Python运行 `python -B -m stage7.run prepare`，避免WSL本地代理问题。`exposure_search` 的历史路径清单只用于此机器上的审计，不进入模型推理。

`after_mechanism.sh` 顺序执行源图标准验证、数值机制验证、并列排名检查、机制汇总、观测图资源兼容性检查、协议冻结、所有目标适配、全局选择冻结、正式输入下载、正式评测。不会执行阶段五的 `initialize.py`。下载正式文本的Windows Python路径可用 `DRTA_FETCH_PYTHON` 指定；也可在冻结后手动执行：

```bash
python -B -m stage7.formal_fetch
```

只有 `frozen.json` 存在并通过哈希检查，正式评测才加载目标查询。全部目标α选定前，不允许运行正式评测。每个α都执行实际非线性前向；不插值最终实体分数。

本次封存后首次启动正式评测时，发现Windows生成的历史文件清单使用反斜杠，WSL将其识别为文件名字符；此前尚未打开正式查询。使用独立的 `path_compat.py` 仅在SHA文件读取边界统一路径分隔符，保留所有冻结代码字节及选参结果。适配器先重新校验全部历史文件，记录于 `results/path_compat_formal.json`。从已冻结状态恢复本次正式评测及分析可运行：

```bash
bash stage7/continue_formal.sh
```

也可单独运行 `bash stage7/run.sh -m stage7.path_compat formal` 或 `verify`。此适配不改变模型、图、排名或协议。全新复现的Windows资产清单仍应使用该入口。

当前 `after_mechanism.sh` 已在流程入口调用路径适配器，并在正式运行后加入α0关系路径诊断；旧失败的日志仍保留。

完成正式评测后，统计/制图/报告使用Windows既有Python；这里的`python`应指具备上述统计依赖及Microsoft YaHei字体的Windows环境。完整性核验在WSL使用 `bash stage7/run.sh -m stage7.path_compat verify`：

```bash
python -B -m stage7.run analyze
python -B -m stage7.plot
python -B -m stage7.run report
```

本轮额外的NELL-v1关系层数值诊断在正式模型运行全部完成后执行，可单独复现：`bash stage7/run.sh -m stage7.alpha0_audit`（支持相同路径参数）。它比较共同权重、CSR α0与官方关系前向，只输出诊断记录，不替换冻结的模型排名或选择。

Windows统计、制图不需要PyTorch；完整性核验与CPU单元测试也可在WSL同一环境执行。中断后，代理与正式推理依据已存哈希恢复；同一选择计划不会用确认结果重新搜索。

## 设计要点

四环境A自然频率、B关系均衡、C端点度数3×3分层、D关系模体支持三分层，共享物理掩码图，尽量只改变查询分布。物理掩码按唯一正向事实真实删除5%/10%/15%，独立于推理查询预算。掩码事实的重复拷贝及其逆边全部删除，未掩码重复观测保持原数目。生成森林、关系覆盖和剩余度数2的约束从不放宽；不足时记录实际数量并回退。原阶段六采样基线保留原全局去重行为，相关差异单独记录。

四环境查询数匹配，所有密度使用共同5%隐藏池中的查询；全局选择/确认事实互斥。环境间和密度间存在重叠，不能把12个环境当成12个独立样本。A是在连通性/度数等约束下的自然事实采样，不能精确恢复被保护的低频关系分布。主要选择器最大化选择集12环境最小LCB，仅最终候选进入确认；任一环境确认LCB不大于0即回退α0。κ=1.64是经验阈值，不保证正式任务不掉分。

除所要求的固定、均值、确认LCB、随机、样本权重与长尾权重对照，还预注册仅10%密度的四环境LCB/均值，以及仅A环境的三密度LCB，用于区分多环境与多密度两个因素。旧阶段六冻结拓扑参数在新数值路径上的重放只用于历史回归，不作为新增图谱的可用参数。

固定逐查询实体线性投影的GEMM形状、全局固定关系编号对、FP32且关闭TF32；不改变权重、数学公式或悲观并列排名。该数值修复单独回归，不能计入拓扑适配增益。

## 产物

`protocol.json` 冻结研究规则；`frozen.json` 冻结逐目标/训练种子/代理种子的选择；`splits/` 保留真实掩码与查询身份；`adaptation/plans/` 在确认前封存候选；`adaptation/ranks/`、`evaluation/` 保存双向整数排名及哈希；`logs/` 保存运行日志。`results/` 包含逐图指标、代理环境、密度敏感性、配对Bootstrap、训练种子差异、选择稳定性、暴露审查和完整性核验。置信区间以原始事实为单位，先合并头尾RR，再平均训练种子，重采样5000次；不构成分数集成。

报告读取research_decision.json中由真实结果复核后撰写的结论。全新复验若数值或数据结果不同，应先复核并更新该结论，再生成报告；不能自动沿用文字结论或据正式结果改动已冻结参数。
