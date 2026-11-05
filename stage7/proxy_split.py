import numpy as np
from topology import extract
from stage6.observed_loader import from_forward
from stage6.proxy_split import make_split as legacy_split
from stage7.proxy_density import nested_masks,remaining_graph
from stage7.proxy_environments import environments
from stage7.util import ROOT,identity


def make_split(graph,seed):
    masks,meta=nested_masks(graph,seed)
    full=extract(graph.edge_index,graph.edge_type,graph.n,graph.R,cache=ROOT/'cache/topology')
    envs,description=environments(graph,masks[0],full['edge_index'],seed)
    old_rem,old_queries,cut,old_meta=legacy_split(graph,seed)
    arrays={'facts':graph.forward,'legacy_queries':old_queries}
    for i,hidden in enumerate(masks):
        arrays['hidden'+str(i)]=hidden
    for name,v in envs.items():
        for phase,ids in v.items():
            arrays[name+'_'+phase]=ids
    info=dict(seed=seed,graph_sha256=graph.key,mask_metadata=meta,description=description,
         environment_query_hashes={name:{ph:identity(graph.forward[ix]) for ph,ix in v.items()} for name,v in envs.items()},
         legacy=dict(cut=cut,metadata=old_meta))
    return arrays,info,old_rem


def residual(graph,arrays,density_index):
    return remaining_graph(graph,arrays['hidden'+str(density_index)],arrays['facts'])
