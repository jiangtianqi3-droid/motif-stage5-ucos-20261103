# 实验目录

| 实验 | 入口 | 主要依赖 |
| --- | --- | --- |
| 11 图零样本评测 | [universal_11g_v1](universal_11g_v1/README.md) | 阶段七非对称基准、冻结图谱与权重、ConceptNet 词面适配器 |
| ConceptNet100k 专项 | [conceptnet_zeroshot_v1](conceptnet_zeroshot_v1/README.md) | 冻结阶段七骨干与 ConceptNet 词表 |

每个实验目录内，`src/` 是实现，`scripts/` 是运行和核验入口。11 图实验的 `monitor/` 提供 Windows 进度窗口和原运行的诊断脚本。权重、数据和运行环境的准备方式见各实验说明，生成的缓存、日志和指标保存在本地运行目录。
