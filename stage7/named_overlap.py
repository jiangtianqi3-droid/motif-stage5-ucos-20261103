"""Named-token overlap audit uses observed facts and pretraining facts only."""
import re
import zipfile
from stage7.util import ROOT,read,dump,sha,log


def canon(value):
    h,r,t=value
    def token(x):
        x=x.strip('<>')
        x=re.sub(r'^https?://www.wikidata.org/entity/','',x)
        x=re.sub(r'^https?://www.wikidata.org/prop/direct/','',x)
        x=re.sub(r'^https?://rdf.freebase.com/ns/','/',x)
        if x.startswith('/m.'):x=x.replace('/m.','/m/',1)
        return x
    return tuple(map(token,(h,r,t)))


def parse(text):
    return {canon(line.split('\t') if '\t' in line else line.split()) for line in text.splitlines() if line.strip()}


def run():
    source_root=ROOT.parents[1]/'datasets/audit/pretraining'
    source=set();sources=[]
    for p in source_root.glob('*train.txt'):
        rows=parse(p.read_text(encoding='utf8'));source |=rows
        sources.append(dict(path=str(p),sha256=sha(p),facts=len(rows)))
    assert sources
    entities={v for h,r,t in source for v in (h,t)}
    relations={r for h,r,t in source}
    rows=[]
    for key,rec in read(ROOT/'observations/manifest.json').items():
        if rec.get('status')=='unavailable':continue
        path=ROOT/rec['download_file'].replace('\\','/')
        if 'archive' in rec:
            with zipfile.ZipFile(path) as z:
                text=z.read(rec['observation_member']).decode('utf8')
        else:text=path.read_text(encoding='utf8')
        obs=parse(text)
        en={v for h,r,t in obs for v in (h,t)};rn={r for h,r,t in obs}
        rows.append(dict(key=key,family=rec['family'],observed_facts=len(obs),
            syntactic_fact_overlap=len(obs&source),syntactic_entity_overlap=len(en&entities),
            syntactic_relation_overlap=len(rn&relations),formal_queries_loaded=False,
            warning='Only declared namespace syntax normalized; zero lexical overlap does not prove semantic independence.'))
    dump(ROOT/'results/pretraining_observation_overlap.json',dict(sources=sources,rows=rows))
    log('pretraining_observation_overlap',rows=rows)


if __name__=='__main__':run()
