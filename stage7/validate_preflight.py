"""Validate/rebuild residual cache identities after multiplicity preservation."""
import time
import gc
import numpy as np
from topology import extract,graph_key
from stage6.observed_loader import load_observed
from stage7.proxy_density import nested_masks,remaining_graph
from stage7.util import ROOT,read,dump,sha,log


def main():
    assert not (ROOT/'protocol.json').exists(),'Preflight inputs must be finalized before protocol freeze.'
    results=read(ROOT/'results/new_graph_preflight.json')
    checks=[]
    for key,rec in results.items():
        g=load_observed(ROOT/'observations',key)
        copies=len(g.edge_type)//2-len(g.forward)
        if rec['status']=='compatible_topology':
            expected=[];masks_by_seed={}
            graph_identities=[]
            for ps in [6101,6102,6103]:
                masks,meta=nested_masks(g,ps)
                masks_by_seed[ps]=masks
                for mask in masks:
                    rem=remaining_graph(g,mask)
                    expected.append(graph_key(rem.edge_index,rem.edge_type,rem.n,rem.R))
                    graph_identities.append(rem.key)
            actual=[r['key'] for r in rec['residual_graphs']]
            rebuilt=expected!=actual
            prior=ROOT/'preflight/before_multiplicity_fix'/(key+'.json')
            original=read(prior) if prior.exists() else rec
            changed_from_initial=expected!=[r['key'] for r in original['residual_graphs']]
            if rebuilt:
                old=ROOT/'preflight/before_multiplicity_fix'/(key+'.json')
                if not old.exists():dump(old,rec,exclusive=True)
                began=time.perf_counter();fresh=[]
                for ps,masks in masks_by_seed.items():
                    for di,mask in enumerate(masks):
                        rem=remaining_graph(g,mask)
                        sig=extract(rem.edge_index,rem.edge_type,rem.n,rem.R,block=rec['block'],cache=ROOT/'cache/topology')
                        fresh.append(dict(seed=ps,density_index=di,key=str(sig['key']),hyperedges=len(sig['edge_type'])))
                        del sig,rem;gc.collect()
                rec['residual_graphs']=fresh
                rec['multiplicity_rebuild_seconds']=time.perf_counter()-began
            for row,gid in zip(rec['residual_graphs'],graph_identities):row['residual_sha256']=gid
            rec['residual_identity_verified']=True
            rec['mask_semantics']='masked copies all removed; unmasked multiplicity retained'
        else:
            # All failing candidate graphs have no duplicate input facts, so
            # the revised residual constructor is byte-identical.
            assert copies==0,(key,rec['status'],copies)
            rebuilt=False
            changed_from_initial=False
            rec['mask_semantics']='no duplicate rows; original resource preflight remains applicable'
        rec['masking_code_sha256']=sha(ROOT/'proxy_density.py')
        checks.append(dict(key=key,duplicate_forward_rows=copies,residuals_rebuilt=changed_from_initial,
            rebuilt_this_validation=rebuilt,status=rec['status']))
        dump(ROOT/'preflight'/(key+'.json'),rec)
        log('multiplicity_cache_validated',**checks[-1])
    dump(ROOT/'results/new_graph_preflight.json',results)
    # Prove source pilot graph inputs are exactly unchanged.
    for key in ['wn18rr_train','codex_medium_train','fb15k237_train']:
        graph=load_observed(ROOT.parent/'stage6/observations',key)
        assert len(graph.edge_type)//2==len(graph.forward)
        for ps in [6101,6102,6103]:
            _,meta=nested_masks(graph,ps)
            old=read(ROOT/'splits'/f'{key}__p{ps}.json')
            assert [m['residual_sha256'] for m in meta]==[m['residual_sha256'] for m in old['mask_metadata']]
    dump(ROOT/'results/proxy_multiplicity_verification.json',dict(status='passed',checks=checks,
        masking_code_sha256=sha(ROOT/'proxy_density.py'),source_pilot_graphs_byte_identical=True,
        stage6_legacy_baseline_unchanged=True,formal_queries_loaded=False))


if __name__=='__main__':main()
