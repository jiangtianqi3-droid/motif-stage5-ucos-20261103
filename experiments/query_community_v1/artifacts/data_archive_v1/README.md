# 完整实验数据归档

本目录保存阶段 `query_community_v1` 的逐文件无损分卷。新代码、中文报告、最终统计表和案例图直接保存在同一 Git 分支；大量原始证据以可校验的压缩卷保存，避免数十万小文件直接进入 Git。

包含正式运行 `d48ae432a2533cdaaec0` 的全部 **321,648 条逐查询证据记录**，以及冒烟/失败运行记录、各图诊断和配对区间、协议与依赖版本、真实案例、日志、看护与暂停/恢复记录、实际冻结门控输出和签名缓存审计。失败记录保留原状，不混作正式结果。

`archive_manifest.json` 记录每卷 SHA256、文件数与体量；`index/*.json.zst` 记录每个原文件的相对路径、字节数和 SHA256。每卷在打包时均已完整解压并逐文件核对哈希。恢复得到原始文件字节，原实验文件从未移动、覆盖或删除。单卷不超过40 MiB。

## 校验和恢复

在仓库根目录执行：

```powershell
python -m pip install --target experiments/query_community_v1/cache/archive_dependencies -r experiments/query_community_v1/scripts/archive/requirements.txt

# 仅解压校验全部压缩内容，不写原始结果：
python experiments/query_community_v1/scripts/archive/restore_data.py --verify-only

# 恢复到 experiments/query_community_v1 下原有相对路径：
python experiments/query_community_v1/scripts/archive/restore_data.py
```

也可使用 `--destination <独立目录>` 恢复到新目录；`--parts part-00001.tar.zst` 只校验或恢复指定卷。已有文件只有内容哈希相同才会跳过，内容不同会报错，绝不覆盖。每卷是独立 tar.zst，无需先拼接。

## 输入与边界

`../inputs/` 另外直接保存四图的合法观察图、完整四维签名、关系词表、原始划分文件及独立的离线目标数组，路径和哈希见其 `manifest.json`。`offline_targets_ONLY.npz` 和原始验证/测试文件只供离线核查，严禁作为候选种子或加入观察图。

历史模型检查点和源码沿用当前仓库已有归档与 `freeze_manifest.json` 中的只读依赖；编译缓存、安装的第三方依赖和 Python 字节码不入档。需要重新执行算法时，仍须按阶段 README 配置对应的 MOTIF/WSL 环境；恢复和核查已保存数据只需要 Python 与上面的固定压缩库。
