"""Observation-only exact topology compatibility test in a bounded CPU process."""
import argparse
import subprocess
import sys
import time
from stage7.util import ROOT,read,dump,log,sha


def worker(key,block):
    import resource
    # WSL has 7.4GiB RAM + 2GiB swap. Leave space for the host and Python libs.
    budget=7*1024**3
    resource.setrlimit(resource.RLIMIT_AS,(budget,budget))
    from stage6.observed_loader import load_observed
    from topology import extract
    import gc
    from stage7.proxy_density import nested_masks,remaining_graph
    from stage6.observed_loader import from_forward
    began=time.perf_counter()
    try:
        g=load_observed(ROOT/'observations',key)
        sig=extract(g.edge_index,g.edge_type,g.n,g.R,block=block,cache=ROOT/'cache/topology')
        hyperedges=len(sig['edge_type']);topology_key=str(sig['key'])
        del sig;gc.collect()
        rebuilt=[]
        f=g.forward
        for seed in [6101,6102,6103]:
            masks,_=nested_masks(g,seed)
            for di,mask in enumerate(masks):
                import numpy as np
                rem=remaining_graph(g,mask,f)
                sig=extract(rem.edge_index,rem.edge_type,rem.n,rem.R,block=block,cache=ROOT/'cache/topology')
                rebuilt.append(dict(seed=seed,density_index=di,key=str(sig['key']),residual_sha256=rem.key,hyperedges=len(sig['edge_type'])))
                del sig,rem;gc.collect()
        value=dict(status='compatible_topology',key=key,n=g.n,R=g.R,hyperedges=hyperedges,block=block,
            topology_key=topology_key,residual_graphs=rebuilt,observation_sha256=g.key,seconds=time.perf_counter()-began,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            address_space_budget=budget,formal_labels_loaded=False)
    except (MemoryError,RuntimeError) as e:
        value=dict(status='resource_failure',key=key,block=block,error=repr(e),seconds=time.perf_counter()-began,
            address_space_budget=budget,formal_labels_loaded=False,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    dump(ROOT/'preflight'/(key+f'__block{block}.json'),value,exclusive=True)


def run():
    results={}
    for key,rec in read(ROOT/'observations/manifest.json').items():
        if rec.get('status')=='unavailable':continue
        path=ROOT/'preflight'/(key+'.json')
        if not path.exists():
            attempts=[]
            for block in [8,2,1]:
                attempt=ROOT/'preflight'/(key+f'__block{block}.json')
                if not attempt.exists():
                    p=subprocess.run([sys.executable,'-m','stage7.preflight','--worker',key,'--block',str(block)])
                    if not attempt.exists():
                        dump(attempt,dict(status='process_failure',key=key,block=block,returncode=p.returncode,formal_labels_loaded=False),exclusive=True)
                value=read(attempt);attempts.append(dict(**value,record_sha256=sha(attempt)))
                if value['status']=='compatible_topology':break
            dump(path,dict(**value,attempts=attempts),exclusive=True)
        results[key]=read(path)
        log('new_graph_preflight',**results[key])
    dump(ROOT/'results/new_graph_preflight.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--block',type=int,default=8);a=p.parse_args()
    worker(a.worker,a.block) if a.worker else run()
