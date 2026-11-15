# ConceptNet100k 零样本结构与语义协同

独立分支 `experiment/conceptnet-zeroshot-boost`，从官方仓库提交 `30b2ff3` 建立 worktree。原工作区的两个未提交文件保持原样。历史阶段七代码、B0/B1/B2 权重、英文词表、划分与拓扑签名均只读使用；运行前后核对 SHA256。

所有新增源码、依赖、缓存、日志、状态和结果保存在本目录。WSL Python 复用 `/mnt/d/ultra-phase1-env/env`，缺失的文本依赖仅安装到 `cache/deps`；不修改共享环境。C 盘空间不足，因此工作区位于 D 盘。

```bash
bash scripts/run.sh scripts/check_invariants.py
bash scripts/run.sh scripts/run_conceptnet_boost.py --smoke
bash scripts/run.sh scripts/run_conceptnet_boost.py
bash scripts/run.sh scripts/status.py
```

冒烟从验证、测试各抽8条事实，合计32个双向查询；另抽16条观察事实验证自监督掩码流程。全量自监督500事实、验证1200事实、测试1200事实。GNN批次2；其他计分批次16。全候选特征按块保存在磁盘，不创建全量内存张量。权重搜索顺序读取缓存块，非负坐标网格搜索两轮。中断后复用同一内容指纹的完整块；部分写入文件不会当作完成块使用。

三元门控保留B2的原标准化签名与门控，先按类型保留前30%，聚合时乘0.15（避免被门控归一化抵消）；D*-Plus固定alpha=.90。均值锚定使用未加起点豁免的全图均值；首层稀疏源边raw权重改为1后仍按相同均值归一化。历史Triton发送端是存储边第二行，使用实际消息方向的关系局部度数。

P0-Prior无目标标签选参；P0-SelfSup先同时掩蔽500事实及其逆边，重算所有观察图特征；P1-ValCalib使用目标验证标签。特征生成器不接收truth；测试真值仅用于离线Filtered排名。英文词表属于公开候选元数据；数值形状的原始概念合法，但绝不将内部ID转换成伪文本。

该基准已有历史测试暴露；新运行不是首次盲测。冻结B2保留其历史源训练协议与局限。0.18–0.22和10–25分钟仅是任务目标，实际指标、时间和资源在结果中如实记录。结构参数先固定默认值，未执行18组合全网格；报告明确披露。不得保证语义共识一定将答案排到前三。

实时状态：`results/status.json`。全量完成报告：`report_conceptnet_boost.md`。每次运行记录输入、代码指纹、锁定权重、逐查询排名、Top50、四指标及2000次事实配对置信区间、度数与可达性诊断。
