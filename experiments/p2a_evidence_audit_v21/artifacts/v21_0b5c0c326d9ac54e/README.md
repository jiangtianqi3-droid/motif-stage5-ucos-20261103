# V2.1 · 100 查询正式实验归档

运行 `0b5c0c326d9ac54ea8af36ed4a291e93daf8c44d06d545ee4a05c848b9f9138e`，代码 `6c30d004a106bef0ee4a31562c897799f9b768b9`。FBIngram_25 固定 50 条测试事实、两个方向，共 100 条查询。全部模型参数、全图关系表示、全图 D* 统计及实体空间冻结。API 请求为 0。

- [完整中文报告与七幅图](report_p2a.md)
- [充分性统计](sufficiency/sufficiency_metrics.csv)与[逐查询结果](sufficiency/sufficiency_per_query.csv)
- [必要性统计](necessity/necessity_metrics.csv)与[逐查询结果](necessity/necessity_per_query.csv)
- [10 次局部匹配随机删除全部明细](necessity/random_deletion_details.csv.gz)
- [固定查询](propagation/fixed_queries.csv)、[T0–T4 审计](propagation/t0_t4_audit.csv)与[原评分输入](propagation/scoring_inputs.json.gz)
- [路径聚合指标](aggregation/aggregation_metrics.csv)与[逐查询聚合](aggregation/aggregation_per_query.csv)
- [Bootstrap 与配对比较](statistics.json)及[方法间差异](paired_method_differences.csv)
- [全部 100 个查询检查点](query_checkpoints/)
- [冻结证明](freeze_proof.json)、[全实体基线验证](baseline_verification.json)、[关系缓存批次](relation_cache_schedule.json)
- [运行配置](audit_config.yaml)、[环境与来源](provenance.json)、[测试日志](unit_tests.log)
- [归档文件清单与哈希](publication_manifest.json)

每个 .gz 文件均可用 Python gzip 解压为原始同名 JSON、CSV 或日志。publication_manifest.json 同时保留发布字节、解压字节及原始文件 SHA256，除报告图片相对路径转换外，全部数值记录均保持原字节。complete.json 保留原运行完整清单，包括可重算且留在本地的关系表示缓存。

示例解压：

```python
import gzip
from pathlib import Path
p = Path("necessity/random_deletion_details.csv.gz")
p.with_suffix("").write_bytes(gzip.decompress(p.read_bytes()))
```
