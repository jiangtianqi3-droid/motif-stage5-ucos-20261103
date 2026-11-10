"""Ground truth is confined to split preparation and offline metric computation."""
from collections import defaultdict
import numpy as np
from .util import HISTORY, DATA, historical_package, read, dump

def positive_index(triples, R):
    index=defaultdict(set)
    for h,r,t in triples:
        h,r,t=map(int,(h,r,t)); index[h,r].add(t); index[t,(r+R//2)%R].add(h)
    return index

class Dataset:
    def __init__(self,key,g,output):
        self.key=key
        adapter=historical_package()
        folder=HISTORY/'topology-semantic-stage4-20261102/semantic'
        provenance=read(folder/'target_mapping_provenance.json')[key]
        vocab=read(folder/'relation_mappings.json')[key]
        self.relations=[v['original_name'] for v in vocab[:g.R//2]]
        rel={s:i for i,s in enumerate(self.relations)}; entities={}
        raw=adapter.runtime_path(provenance['raw'])
        def parse(name):
            out=[]
            with (raw/name).open(encoding='utf-8') as f:
                for line in f:
                    h,r,t=line.split()
                    for token in (h,t):
                        if token not in entities: entities[token]=len(entities)
                    out.append([entities[h],rel[r],entities[t]])
            return np.asarray(out,dtype=np.int64).reshape(-1,3)
        observed=parse(provenance['fact_file'])
        valid,test=[parse(n) for n in (['valid_ind.txt','test_ind.txt'] if provenance['kind']=='grail' else ['inf_valid.txt','inf_test.txt'])]
        assert np.array_equal(observed[:,[0,2]].T,g.edge_index[:,:len(observed)])
        assert np.array_equal(observed[:,1],g.edge_type[:len(observed)])
        assert len(entities)==g.n
        g.assert_hidden_absent(np.concatenate([valid,test]))
        overlap=set(map(tuple,valid)) & set(map(tuple,test))
        # Duplicate validation facts are not legal independent test observations.
        if overlap: raise ValueError(f'Validation/test duplicate facts: {len(overlap)}')
        with np.load(DATA/f'{key}.npz',allow_pickle=False) as z:
            legacy=z['queries']
            assert np.array_equal(legacy,np.concatenate([valid,test]) if provenance['kind']=='grail' else test)
        self.names=list(entities); self.g=g; self.truth={'valid':valid,'test':test}
        self.public={split:np.stack([a[:,[0,1]],np.c_[a[:,2],(a[:,1]+g.R//2)%g.R]],axis=1) for split,a in self.truth.items()}
        self.observed=positive_index(observed,g.R)
        self.candidate_masks={'valid':self.observed,'test':positive_index(np.concatenate([observed,valid]),g.R)}
        self.filters={'valid':positive_index(np.concatenate([observed,valid]),g.R), 'test':positive_index(np.concatenate([observed,valid,test]),g.R)}
        dump(output,dict(dataset=key,valid_facts=len(valid),test_facts=len(test),legacy_facts=len(legacy),
                         observation_only=True,direct_inverse_leakage=0,split_overlap=0,
                         validation_evaluation_filter='observed+valid; test truths excluded from selection',
                         test_evaluation_filter='observed+valid+test; offline only',
                         candidate_filter='valid: observed; test: observed+valid; no test truth or current answer reinstatement',
                         legacy_protocol='Grail legacy valid+test pooled; official test reported separately'))

    def target(self,split,i,d): return int(self.truth[split][i,2 if d==0 else 0])

    def rank(self,split,i,d,scores,ids,order=None):
        target=self.target(split,i,d); h,q=map(int,self.public[split][i,d])
        mask=np.ones(len(scores),dtype=bool)
        exclude=list(self.filters[split].get((h,q),()))
        mask[exclude]=False; mask[target]=False
        if order is None or list(order)==list(range(len(ids))):
            return int(1+np.count_nonzero(mask & (scores>=scores[target])))
        updated=scores.copy(); updated[np.asarray(ids)[order]]=scores[ids]
        rank=int(1+np.count_nonzero(mask & (updated>=updated[target])))
        if target not in ids:
            assert rank==int(1+np.count_nonzero(mask & (scores>=scores[target])))
        return rank

def metrics(ranks):
    r=np.asarray(ranks,dtype=float)
    return dict(MRR=float(np.mean(1/r)),Hits1=float(np.mean(r<=1)),Hits3=float(np.mean(r<=3)),Hits10=float(np.mean(r<=10)))

def bootstrap(base, target, repetitions=2000):
    # Both directions are sampled together for each original fact.
    a=np.asarray(base).reshape(-1,2); b=np.asarray(target).reshape(-1,2)
    funcs={'MRR':lambda r:1/r,'Hits1':lambda r:r<=1,'Hits3':lambda r:r<=3,'Hits10':lambda r:r<=10}
    out={}
    for name,func in funcs.items():
        delta=(func(b).astype(float)-func(a).astype(float)).mean(axis=1)
        rng=np.random.default_rng(1024); draws=[]
        # Chunked resampling bounds temporary arrays independently of dataset size.
        for at in range(0,repetitions,32):
            draws.extend(rng.choice(delta,size=(min(32,repetitions-at),len(delta)),replace=True).mean(axis=1).tolist())
        out[name]=dict(delta=float(delta.mean()),lo=float(np.quantile(draws,.025)),hi=float(np.quantile(draws,.975)),replicates=repetitions,clusters=len(delta))
    return out
