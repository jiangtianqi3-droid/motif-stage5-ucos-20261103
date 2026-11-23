"""Publish a verified compact, lossless snapshot of the completed V2.1 results."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_PLAIN = 1024 * 1024
EXCLUDED = {'full_relation_embeddings.npz'}


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def native_path(value):
    if value.startswith('/mnt/') and len(value) > 7:
        if __import__('os').name == 'nt':
            return Path(value[5].upper() + ':' + value[6:])
    return Path(value)


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Published destination has different bytes: {path}')
        return
    with path.open('xb') as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    status_path = ROOT / 'outputs/status.json'
    status = read_json(status_path)
    if status['state'] != 'complete' or status['completed_queries'] != status['total_queries']:
        raise ValueError('A fully completed experiment is required')
    run_dir = args.run_dir or native_path(status['run_directory'])
    if run_dir.name != status['run_id']:
        raise ValueError('Run ID does not match the final status')
    completion = read_json(run_dir / 'complete.json')
    if completion['state']['run_id'] != status['run_id']:
        raise ValueError('Completion manifest belongs to another run')
    checkpoints = sorted((run_dir / 'query_checkpoints').glob('*.json'))
    if len(checkpoints) != status['total_queries']:
        raise ValueError('Missing completed query checkpoints')
    for rel, expected in completion['files'].items():
        source = run_dir / rel
        if not source.is_file() or digest_file(source) != expected:
            raise ValueError(f'Completion checksum failed: {rel}')
    output = ROOT / 'artifacts' / ('v21_' + status['run_id'][:16])
    records = []
    excluded = []

    def publish(source, relative, data=None, compress=False):
        content = source.read_bytes() if data is None else data
        source_hash = digest_file(source)
        stored = gzip.compress(content, compresslevel=9, mtime=0) if compress else content
        destination = relative + '.gz' if compress else relative
        write_new(output / destination, stored)
        # Verify the complete bytes, including every random draw and checkpoint record.
        recovered = gzip.decompress((output / destination).read_bytes()) if compress else (output / destination).read_bytes()
        if recovered != content:
            raise ValueError(f'Published round-trip differs: {destination}')
        records.append(dict(file=destination, bytes=len(stored), sha256=digest_bytes(stored),
                            source=relative, source_bytes=source.stat().st_size,
                            source_sha256=source_hash, decoded_sha256=digest_bytes(content),
                            compression='gzip' if compress else None,
                            transformation='relative image links' if data is not None else None))

    for source in sorted(run_dir.rglob('*')):
        if not source.is_file():
            continue
        rel = source.relative_to(run_dir).as_posix()
        if rel in EXCLUDED:
            excluded.append(dict(file=rel, bytes=source.stat().st_size, sha256=digest_file(source),
                                 reason='frozen full-graph relation representation cache; recomputable'))
            continue
        if source.suffix in ('.partial', '.tmp'):
            continue
        compress = rel.startswith('query_checkpoints/') or (
            source.stat().st_size > MAX_PLAIN and source.suffix in ('.csv', '.json', '.log'))
        publish(source, rel, compress=compress)
    for source, rel in ((status_path, 'final_status.json'),
                        (ROOT / 'configs/audit_config.yaml', 'audit_config.yaml'),
                        (ROOT / 'logs/formal-20261123-113422.log', 'logs/formal.log'),
                        (ROOT / 'logs/formal-20261123-113422.err.log', 'logs/formal.stderr.log')):
        if source.is_file():
            publish(source, rel, compress=source.suffix == '.log')
    report = ROOT / 'report_p2a.md'
    text = report.read_text(encoding='utf-8')
    prefix = f'outputs/{status["run_id"]}/'
    if prefix not in text or '查询 100 条' not in text:
        raise ValueError('The report does not match the completed 100-query run')
    publish(report, 'report_p2a.md', data=text.replace(prefix, '').encode('utf-8'))

    bundle = dict(schema='motif-v21-publication-v1', run_id=status['run_id'],
                  experiment='P2-A Stage 1 V2.1', run_commit=status['git_commit'],
                  completed_at=status['last_update'], queries=status['completed_queries'],
                  original_facts=50, llm_api_requests=status['llm_api_requests'],
                  decision=status['decision'], records=records, excluded=excluded,
                  original_completion_manifest='complete.json',
                  verified_original_files=len(completion['files']),
                  lossless_compressed_files=sum(r['compression'] == 'gzip' for r in records),
                  total_stored_bytes=sum(r['bytes'] for r in records))
    write_new(output / 'publication_manifest.json',
              (json.dumps(bundle, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    description = (
        '# V2.1 · 100 查询正式实验归档\n\n'
        f'运行 `{status["run_id"]}`，代码 `{status["git_commit"]}`。'
        'FBIngram_25 固定 50 条测试事实、两个方向，共 100 条查询。'
        '全部模型参数、全图关系表示、全图 D* 统计及实体空间冻结。API 请求为 0。\n\n'
        '- [完整中文报告与七幅图](report_p2a.md)\n'
        '- [充分性统计](sufficiency/sufficiency_metrics.csv)与[逐查询结果](sufficiency/sufficiency_per_query.csv)\n'
        '- [必要性统计](necessity/necessity_metrics.csv)与[逐查询结果](necessity/necessity_per_query.csv)\n'
        '- [10 次局部匹配随机删除全部明细](necessity/random_deletion_details.csv.gz)\n'
        '- [固定查询](propagation/fixed_queries.csv)、[T0–T4 审计](propagation/t0_t4_audit.csv)与[原评分输入](propagation/scoring_inputs.json.gz)\n'
        '- [路径聚合指标](aggregation/aggregation_metrics.csv)与[逐查询聚合](aggregation/aggregation_per_query.csv)\n'
        '- [Bootstrap 与配对比较](statistics.json)及[方法间差异](paired_method_differences.csv)\n'
        '- [全部 100 个查询检查点](query_checkpoints/)\n'
        '- [冻结证明](freeze_proof.json)、[全实体基线验证](baseline_verification.json)、[关系缓存批次](relation_cache_schedule.json)\n'
        '- [运行配置](audit_config.yaml)、[环境与来源](provenance.json)、[测试日志](unit_tests.log)\n'
        '- [归档文件清单与哈希](publication_manifest.json)\n\n'
        '每个 .gz 文件均可用 Python gzip 解压为原始同名 JSON、CSV 或日志。'
        'publication_manifest.json 同时保留发布字节、解压字节及原始文件 SHA256，'
        '除报告图片相对路径转换外，全部数值记录均保持原字节。'
        'complete.json 保留原运行完整清单，包括可重算且留在本地的关系表示缓存。\n\n'
        '示例解压：\n\n'
        '```python\nimport gzip\nfrom pathlib import Path\n'
        'p = Path("necessity/random_deletion_details.csv.gz")\n'
        'p.with_suffix("").write_bytes(gzip.decompress(p.read_bytes()))\n```\n'
    )
    write_new(output / 'README.md', description.encode('utf-8'))
    # The final manifest is the pointer for later publications and verification.
    pointer = dict(run_id=status['run_id'], report=(output.relative_to(ROOT) / 'report_p2a.md').as_posix(),
                   manifest=(output.relative_to(ROOT) / 'publication_manifest.json').as_posix(),
                   manifest_sha256=digest_file(output / 'publication_manifest.json'))
    pointer_path = ROOT / 'artifacts/latest.json'
    encoded = (json.dumps(pointer, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    write_new(pointer_path, encoded)
    print(json.dumps(dict(directory=str(output), files=len(records), queries=len(checkpoints),
                          stored_mib=round(bundle['total_stored_bytes'] / 1024 ** 2, 2),
                          verified_original_files=len(completion['files'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
