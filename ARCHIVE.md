# 阶段五私有归档说明

本仓库保存2026-11-03完成的MOTIF新阶段五代码和实验结果。原实验目录未修改。

- [完整中文报告](report.md)
- [宏平均CSV](results/macro_seed_robustness.csv)
- [逐图配对区间](results/paired_bootstrap.csv)
- [固定查询组配对区间](results/query_stratified_bootstrap.csv)
- [结果汇总JSON](results/summary.json)
- [最终核验](results/verification.json)

归档包含全部本阶段代码、runtime代码、21条训练轨迹、全部已选/未选检查点与恢复检查点、275份排名及元数据、查询掩码、语义向量、日志、报告和CSV/JSON。编译缓存、临时缓存和Python字节码未上传；这些内容可重新生成。

21条轨迹共10,500次参数更新。最终核验通过不代表方法效果达到预期：源端真实余弦对应优于乱序，但目标6/11图乱序点估计更高；T-4D和U-Cos在locked、historical整体低于B0；HubCal源选lambda=0。完整结论和数值波动说明见报告。

`ARCHIVE_MANIFEST.json`记录原文件SHA256；`.gitattributes`禁止换行转换，保留冻结哈希。原始代码仍使用当时的Windows/WSL目录布局。重新训练需要旧阶段源图折叠缓存、拓扑缓存、官方初始检查点及编码器缓存，路径和哈希见`EXTERNAL_DEPENDENCIES.json`、`frozen.json`及原README；这些历史依赖未重复打包。本阶段全部训练结果与排名已保存，可直接检查。

第三方MOTIF/ULTRA依赖与出处见[上游README](upstream/MOTIF_README.md)。此仓库是私有研究归档，没有为第三方代码或数据新增授权声明。
