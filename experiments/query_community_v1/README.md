# 查询感知强关联社区模块 v1

**正式全量实验已于北京时间2026-11-09 00:57完成。** 四图共6,701条事实、13,402个方向查询，A/B/C/D/E3a/E3b各K=10/30、M=5/10，共321,648份原始证据、96组指标和384个配对区间。原始28项正确性测试及19,538项历史资产SHA256复核通过。

完整结论见 [中文实验报告](report.md)。四维拓扑在部分图上降低枢纽占比，但未建立证据路径召回的显著提升，NELL的枢纽占比还会上升，不能声称方法普遍有效。

数据归档与恢复说明见 [完整实验数据](artifacts/data_archive_v1/README.md)，独立观察图和离线标签清单见 [输入快照](artifacts/inputs/manifest.json)。分卷保留全部逐查询文件字节和实际门控输出；中文报告、最终统计和案例图也可直接在仓库中查看。

从 `baseline-4d-gate-frozen-20261108` 创建，分支 `experiment/query-community-v1`。历史源码、权重、数据、排名均只读；所有新增文件位于本目录。冻结清单同时覆盖阶段一至阶段七的分散资产及未提交差异备份。

环境复用 WSL Ubuntu-24.04：`/mnt/d/ultra-phase1-env/env`，PyTorch 2.1.0+cu118、NumPy 1.26.4、SciPy 1.13.1、NetworkX 3.2.1；不新增外部模型调用。主图算法为稀疏 Louvain，不依赖 Leiden/igraph。

```powershell
# 仅安装到本阶段目录，不改变历史WSL环境：
wsl -d Ubuntu-24.04 -- /mnt/d/ultra-phase1-env/env/bin/python -m pip install --target /mnt/d/motif-phase4-work/query-community-v1-20261108/experiments/query_community_v1/cache/dependencies --cache-dir /mnt/d/motif-phase4-work/query-community-v1-20261108/experiments/query_community_v1/cache/pip -r /mnt/d/motif-phase4-work/query-community-v1-20261108/experiments/query_community_v1/config/requirements-phase.txt

# 自动测试（Windows）：
python -m pytest D:/motif-phase4-work/query-community-v1-20261108/experiments/query_community_v1/tests -q

# 实际WSL运行环境测试（-s避开DrvFS临时文件FD捕获限制）：
wsl -d Ubuntu-24.04 -- bash /mnt/d/motif-phase4-work/query-community-v1-20261108/experiments/query_community_v1/run.sh -m pytest tests -s -q --junitxml=results/unit_tests_wsl.xml

# 冒烟：四张真实图，每图无答案随机选1条事实、两个查询方向、全部方法和K/M预算。
wsl -d Ubuntu-24.04 -- bash /mnt/d/motif-phase4-work/query-community-v1-20261108/experiments/query_community_v1/run.sh scripts/run_community_experiments.py --smoke

# 全量运行；异常退出后执行同一命令断点续跑：
wsl -d Ubuntu-24.04 -- bash /mnt/d/motif-phase4-work/query-community-v1-20261108/experiments/query_community_v1/run.sh scripts/run_community_experiments.py

# 只读状态：
python D:/motif-phase4-work/query-community-v1-20261108/experiments/query_community_v1/scripts/status.py

# 固定无答案抽样试验（结果位于不同指纹目录，不替代默认全量）：
wsl -d Ubuntu-24.04 -- bash /mnt/d/motif-phase4-work/query-community-v1-20261108/experiments/query_community_v1/run.sh scripts/run_community_experiments.py --limit 128
```

全量默认评估四图全部历史协议测试事实、两个方向，A/B/C/D/E3a/E3b 六条实际流程，K∈{10,30}、每候选M∈{5,10}，每查询共享≤50节点/100边。需要长时运行时应使用 `scripts/launch_background.sh`；启动后停止自动轮询，用户主动询问时才运行 status.py。

## 数据边界

`load_observed` 仅加载阶段六 TSDC 的 observation-only NPZ，验证原始哈希与完整逆边；签名加载先核对该观察图。阶段六bundle的raw是全零占位，适配器从阶段四canonical缓存恢复真实raw/count并验证z一致，不能用占位数组计算频次。原始 entity/关系 ID、edge_id 与方向均保留，多关系仅在全局聚类投影合并。图、权重及路径检索函数没有答案参数。

候选来自阶段七 `B0.rrf.npy` 的未过滤全候选排名，基线不是重新训练模型。原始 logits 没有保存，`base_score` 明确标为 `reciprocal_raw_rank`，不是 logit。相同分数按实体 ID 排列，`raw_tie_rank` 保留历史悲观并列秩。输出不含答案字段；独立 diagnostics.py 仅在图谱全部检索写完之后读取真值、检查所有隐藏验证/测试边及其逆边，然后离线评价。

实际 T-4D 门控由冻结 checkpoint 的六层 RelHCNet 前向hook提取，模型参数前后哈希一致；无替代常数、无重新训练、不执行实体模型。检查点/门控缓存均绑定观察图及检查点哈希。

## 方法和分数

三种边权与两种拓扑子模式均完整实现。固定纯拓扑共现定义 `m~q ⇔ q在m非空槽位中出现`。无支持关系严格按epsilon分母公式得到rho=0，不伪造证据。拓扑映射及公式详见生成的 report.md。

PPR 是2或3次稀疏截断迭代，alpha=.2为**重启概率**；孤立质量返回种子，不声称稳态收敛。反向PPR重新行归一化反向图。交汇分数与有向路径分数独立导出；路径在原始观察图上精确做≤3跳前向/反向连接，避免beam提前删掉桥接边。

**发送端单一全局度因子在行归一化中会抵消**，对PPR的影响只能来自关系分布等非行常数项。度阻尼仍影响原始边权乘积路径排序和子图预算。测试专门固定这一不变量。原始路径权重跨模式量纲不同，应在各自候选内比较；同时保留转移概率乘积。

`pi_h[R]` 基于独立于候选的头部2-hop局部边，`anchor_presence_7[7]` 是模体关系槽位的局部类型覆盖率代理，不是实体级实例计数。`no_evidence` 原样导出。

## 断点恢复和报告

每次运行的配置、代码及依赖写入 `results/runs/<fingerprint>/`。不同代码/配置产生新目录，已完成查询原子缓存，原始结果不覆盖；异常保存回溯，运行锁阻止并发冲突。剩余空间低于8 GiB时安全退出保留断点，不清理任何历史文件。完整运行完成后校验19,538项冻结资产，并自动生成中文 `report.md`、逐查询离线诊断、配对Bootstrap、全部K/M对照表、真实案例图与对应原边/路径表。

Bootstrapping 按原始事实聚类、共同重采两个方向。目标图已有历史暴露，不称首次盲测。不可将冒烟样本、描述性可视化或相关性当作方法已有效的结论。

`freeze_manifest.json` 是历史资产的外部依赖清单，Git标签固定已存在的干净归档提交；`preserved_changes/` 单独保留原来两个脏仓库的修改。保留原始位置和清单，不修改其属性、不移动文件，也不把未经核实的临时变更混入基线。

## 当前实验的本地看护

本次全量运行 `d48ae432a2533cdaaec0` 配有独立本地看护进程，每1小时检查一次进程身份、状态时间戳、进度、剩余磁盘和完成记录。健康检查只落盘；退出、长时间静默或完成时向当前用户发送Windows通知，通知记录保存在 `results/monitor/<run>/`，不调用任何外部模型。

仅针对意外进程退出或明确的临时CUDA/IO错误自动断点续跑，最多2次；续跑前逐文件检查原运行程序、配置与冻结清单哈希，确保继续使用原查询缓存。代码断言、隐藏边检查失败、空间不足、未知错误或源码发生变化时保留现场并通知，不盲目重启，不终止任何活跃实验。运行完成后本次看护自动结束。看护代码位于 `scripts/monitor/`，不会改变已在运行的实验指纹。

```powershell
# 仅当看护未运行时启动；已有看护会拒绝重复启动。
wsl -d Ubuntu-24.04 -- bash /mnt/d/motif-phase4-work/query-community-v1-20261108/experiments/query_community_v1/scripts/monitor/launch_watch.sh --run d48ae432a2533cdaaec0 --pid 33 --interval 3600
```

查看 `results/monitor/d48ae432a2533cdaaec0/status.json` 和 `checks.jsonl` 可核对实际检查时间、下一次检查及任何恢复动作。停止看护可在该目录创建 `STOP` 文件；它只停止看护，不停止实验。电脑休眠期间本地进程暂停，恢复唤醒后继续检查。
