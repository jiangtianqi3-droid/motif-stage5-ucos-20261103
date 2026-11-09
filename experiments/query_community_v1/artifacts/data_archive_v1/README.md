# 数据提交范围与本地归档工具

**按用户最新要求，本次不上传全部约43.6 GB原始实验文件，也不上传对应压缩分卷。** 完整逐查询证据、实际门控缓存、日志及已经生成的部分分卷继续保留在本地，未删除、未移动。`parts/`、`index/`及分卷清单已加入Git忽略规则，避免后续误提交。

仓库采用精简提交，保留本阶段全部代码、正式中文报告、完整96组指标、384个配对区间、逐查询数值诊断、协议和核验记录、真实案例图、四图输入快照，以及四图各一条查询在C/D/E3a/E3b下的16份完整候选证据示例（约1.3 MB）。示例直接复制既有正式结果，哈希和源路径见 `../evidence_examples/manifest.json`；完整数据范围见 `../publication_scope.json`。

本地全量数据包含正式运行 `d48ae432a2533cdaaec0` 的全部 **321,648 条逐查询证据记录**、冒烟/失败记录、看护与暂停/恢复记录以及实际冻结门控输出。仓库没有逐条上传这些原始记录；失败运行的诊断元数据单独保留，不混作正式结果。

`scripts/archive/` 仍保留无损归档、校验和恢复工具，供将来明确需要全量导出时使用。只有自行完成全量归档后才会生成 `archive_manifest.json` 和 `index/*.json.zst`；**当前仓库不提供完整分卷，不能直接执行全量恢复**。工具按每卷最多40 MiB打包，并逐文件解压核对SHA256。

## 未来自行完成全量归档后的校验和恢复

以下命令仅适用于已有完整分卷的本地副本，在仓库根目录执行：

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
