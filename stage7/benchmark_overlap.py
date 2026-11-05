"""Provenance-only audit of official benchmark train vs inference graphs.

Never imported by adaptation; does not read any validation/test query file.
"""
import zipfile
from stage7.prepare import fetch
from stage7.util import ROOT,read,dump,sha,log


def parse(text):
    return {tuple(line.split()) for line in text.splitlines() if line.strip()}


def main():
    meta=read(ROOT/'observations/manifest.json')
    hm=ROOT/'downloads/HM_1k_official_benchmark_train.txt'
    fetch('https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/ad1b20bb290d28cf8a19357ea059077248e25d45/data/Hamaguchi-BM_both-1000/train/train.txt',hm)
    datasets=[]
    hrec=meta['HM_1k_new']
    datasets.append(('HM_1k_new','official_Hamaguchi_source',parse(hm.read_text(encoding='utf8')),
        parse((ROOT/hrec['download_file'].replace('\\','/')).read_text(encoding='utf8')),sha(hm)))
    wrec=meta['WikiTopicsMT4_health_new']
    with zipfile.ZipFile(ROOT/wrec['download_file'].replace('\\','/')) as z:
        member='MTDEA_datasets/WikiTopics-MT4/wikidata_healthv2-trans/train.txt'
        train=z.read(member)
        import hashlib
        datasets.append(('WikiTopicsMT4_health_new','runtime_health_transductive_sibling',parse(train.decode('utf8')),
            parse(z.read(wrec['observation_member']).decode('utf8')),hashlib.sha256(train).hexdigest()))
        member='MTDEA_datasets/WikiTopics-MT4/wikidata_mt4v2-trans/train.txt'
        train=z.read(member)
        datasets.append(('WikiTopicsMT4_health_new','published_MT4_mixture',parse(train.decode('utf8')),
            parse(z.read(wrec['observation_member']).decode('utf8')),hashlib.sha256(train).hexdigest()))
    rows=[]
    for key,variant,train,obs,source_sha in datasets:
        te={x for h,r,t in train for x in (h,t)};oe={x for h,r,t in obs for x in (h,t)}
        tr={r for h,r,t in train};orr={r for h,r,t in obs}
        rows.append(dict(key=key,source_variant=variant,benchmark_train_facts=len(train),inference_observed_facts=len(obs),
            benchmark_train_entities=len(te),inference_entities=len(oe),shared_entities=len(te&oe),
            shared_relations=len(tr&orr),shared_facts=len(train&obs),benchmark_train_sha256=source_sha,
            source='Published benchmark source graph; not used by Stage7 selector or model training',formal_queries_loaded=False))
    dump(ROOT/'results/benchmark_train_inference_overlap.json',dict(rows=rows,
        warning='Official inductive/OOKB categories do not imply fully disjoint entities or independent knowledge provenance.'))
    log('benchmark_train_inference_audit',rows=rows)


if __name__=='__main__':main()
