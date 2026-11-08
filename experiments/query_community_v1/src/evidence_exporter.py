"""Versioned evidence export, rejecting fabricated or disconnected paths."""
import json
import numpy as np
from .graph_adapter import dump

def validate_evidence(record,graph):
    budget=record['budget'];subgraph=record['subgraph']
    assert len(subgraph['nodes'])<=budget['max_nodes'] and len(subgraph['edge_ids'])<=budget['max_edges']
    for c in record['candidate_evidence']:
        paths=c['evidence_paths']
        assert (c['status']=='no_evidence')==(len(paths)==0)
        for path in paths:
            assert len(path['triples'])==len(path['edge_ids'])==len(path['edges'])>0
            h=record['query_info']['head']
            for triple,e in zip(path['triples'],path['edge_ids']):
                assert e in subgraph['edge_ids']
                assert triple==[int(graph.edge_index[0,e]),int(graph.edge_type[e]),int(graph.edge_index[1,e])]
                assert triple[0]==h
                h=triple[2]
            assert h==c['entity']
    json.dumps(record,allow_nan=False)
    return True

def export_json(path,record,graph):
    validate_evidence(record,graph);dump(path,record)

def export_jsonl(path,records,graph):
    from pathlib import Path
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        for record in records:
            validate_evidence(record,graph)
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)
