"""Fetch bytes; materialize observation files only, never held-out queries."""
import urllib.request
import zipfile
import time
import numpy as np
from stage7.util import ROOT,read,dump,sha,log
from stage6.observed_loader import from_forward

CANDIDATES={
 'HM_1k_new':dict(family='WordNet',type='inductive_entity',public_n=9899,
    observation='https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/ad1b20bb290d28cf8a19357ea059077248e25d45/data/Hamaguchi-BM_both-1000/test/test-graph.txt',
    test='https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/ad1b20bb290d28cf8a19357ea059077248e25d45/data/Hamaguchi-BM_both-1000/test/test-fact.txt',
    risk='New split results; shares WordNet with backbone pretraining and historical WN. Not a new independent knowledge source.'),
 'WikiTopicsMT4_health_new':dict(family='Wikidata',type='inductive_entity_relation',public_n=10000,
    archive='https://reltrans.s3.us-east-2.amazonaws.com/MTDEA_data.zip',
    observation_member='MTDEA_datasets/WikiTopics-MT4/wikidata_healthv2-ind/observe.txt',
    test_member='MTDEA_datasets/WikiTopics-MT4/wikidata_healthv2-ind/test.txt',
    risk='New topic/version results; shares Wikidata provenance with CoDEx source and historical WK. Not independent of that family.'),
 'AristoV4_external_new':dict(family='Aristo-science',type='cross_graph_zero_shot_external_transductive_split',public_n=42016,
    archive='https://zenodo.org/records/5942560/files/aristo-v4.zip?download=1',
    observation_member='train',test_member='test',valid_member='valid',
    risk='New knowledge source. Published split is transductive; cross-graph zero-shot evaluation does not create a standard disjoint-entity inductive benchmark.'),
 'UMLS_external_new':dict(family='UMLS-biomedical',type='cross_graph_zero_shot_external_transductive_split',public_n=135,
    observation='https://raw.githubusercontent.com/pykeen/pykeen/5beaada8623a7bc368db9f29a7a43ddd237888a6/src/pykeen/datasets/umls/train.txt',
    valid='https://raw.githubusercontent.com/pykeen/pykeen/5beaada8623a7bc368db9f29a7a43ddd237888a6/src/pykeen/datasets/umls/valid.txt',
    test='https://raw.githubusercontent.com/pykeen/pykeen/5beaada8623a7bc368db9f29a7a43ddd237888a6/src/pykeen/datasets/umls/test.txt',
    risk='New biomedical source and prospective results. Small dense published transductive split; external cross-graph zero-shot experiment, not a standard disjoint-entity inductive split.')}


def fetch(url,path):
    if path.exists():
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=90) as r, path.with_suffix('.part').open('wb') as f:
                while True:
                    data=r.read(1024*1024)
                    if not data: break
                    f.write(data)
            path.with_suffix('.part').replace(path)
            return
        except Exception:
            if attempt==2: raise
            time.sleep(2)


def run(args):
    from datetime import datetime,timezone
    manifest={}
    folder=ROOT/'observations'
    folder.mkdir(exist_ok=True)
    for key,rec in CANDIDATES.items():
        if (folder/(key+'.json')).exists():
            manifest[key]=read(folder/(key+'.json'))
            continue
        try:
            dst=ROOT/'downloads'/(key+'.zip' if 'archive' in rec else key+'_observed.txt')
            fetch(rec.get('archive',rec.get('observation')),dst)
            if 'archive' in rec:
                with zipfile.ZipFile(dst) as z:
                    member=rec['observation_member']
                    if member not in z.namelist():
                        matches=[x for x in z.namelist() if x.endswith(member)]
                        assert len(matches)==1,(member,z.namelist()[:8])
                        member=matches[0]
                    text=z.read(member).decode('utf8')
            else:
                text=dst.read_text(encoding='utf8')
            # Observation-first vocabularies. Published candidate count reserves
            # anonymous isolated IDs without inspecting any held-out fact.
            entities,relations={},{}
            facts=[]
            for line in text.splitlines():
                if not line.strip(): continue
                h,r,t=line.split('\t') if '\t' in line else line.split()
                for x in (h,t):
                    if x not in entities: entities[x]=len(entities)
                if r not in relations: relations[r]=len(relations)
                facts.append([entities[h],relations[r],entities[t]])
            n=max(rec['public_n'],len(entities))
            graph=from_forward(np.asarray(facts,dtype=np.int64),n,2*len(relations),{'source':rec.get('observation',rec.get('archive'))})
            file=folder/(key+'.npz')
            np.savez_compressed(file,edge_index=graph.edge_index,edge_type=graph.edge_type,n=graph.n,R=graph.R)
            value=dict(**rec,key=key,file=file.name,sha256=sha(file),n=n,R=graph.R,
                observed_facts=len(graph.forward),observed_entities=len(entities),anonymous_public_candidates=n-len(entities),
                download_sha256=sha(dst),download_file=str(dst.relative_to(ROOT)),observed_sha256=graph.key,
                formal_queries_loaded=False,created_utc=datetime.now(timezone.utc).isoformat())
            dump(folder/(key+'_vocab.json'),dict(entities=entities,relations=relations),exclusive=True)
            dump(folder/(key+'.json'),value,exclusive=True)
            manifest[key]=value
            log('new_observation_ready',key=key,n=n,R=graph.R,facts=len(graph.forward))
        except Exception as e:
            log('new_observation_error',key=key,error=repr(e))
            manifest[key]=dict(**rec,key=key,status='unavailable',error=repr(e))
    dump(folder/'manifest.json',manifest)
    dump(ROOT/'results/target_exposure_audit.json',dict(candidates=manifest,
        excluded_prior_structural_exposure=['Metafam','ILPC2022SmallInductive','WikiTopicsMT1_tax','WikiTopicsMT2_org','WikiTopicsMT3_art','FBNELL'],
        searched_result_roots=['ultra-phase1-work/results','motif-phase2-work/results','motif-phase3-work/results','motif-phase4-work historical stage1-6'],
        claim='Prospective test graphs with provenance dependencies. HM/WikiTopics share source families; Aristo/UMLS are new external sources with published transductive splits. No claim of three completely independent standard inductive benchmarks.',
        pretraining_sources=['FB15k237','WN18RR','CoDExMedium']))
