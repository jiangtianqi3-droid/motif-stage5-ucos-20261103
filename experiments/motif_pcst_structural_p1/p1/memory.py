"""Consumer-facing P1 structural memory, with no label fields."""
from src.io_utils import PRIMARY
from src.graph_loader import fact


def export(g, query_id, h, r, candidates, raw, selected, values, slots, checkpoint, observation):
    # Prefer the query and actual visited candidate endpoints, then path intermediates.
    precedence = [h] + [c for c in candidates if c in selected['paths']]
    precedence += selected['nodes']
    entities = list(dict.fromkeys(precedence))[:slots]
    ordered = [fact(g,i) for i in selected['ids']]
    return dict(schema='motif-structural-memory-v1',query_id=query_id,query_head=h,query_relation=r,
                selected_entity_ids=selected['nodes'],selected_edge_ids=selected['ids'],
                observations=[[f.id,f.src,f.relation,f.dst] for f in ordered],
                selected_relation_ids=[f.relation for f in ordered],evidence_paths=selected['paths'],
                motif_features=dict(source='B2_T3_frozen_s1024_gate_six_layer_mean_on_query_active_relation_motifs',
                                    per_edge=selected['motif']),
                node_prizes=selected['node_prizes'],edge_prizes=selected['bonuses'],edge_costs=selected['costs'],
                candidate_ids=candidates,candidate_structural_scores=[float(raw[c]) for c in candidates],
                score_kind='reciprocal_raw_rank',candidate_evidence_scores=values,
                memory_entity_ids=entities+[-1]*(slots-len(entities)),
                memory_mask=[True]*len(entities)+[False]*(slots-len(entities)),
                entity_embedding_ref=dict(kind='frozen_query_conditioned_recompute',model=PRIMARY,
                    checkpoint_parameter_sha256=checkpoint,observation_array_sha256=observation,
                    extraction='entity_model.bellmanford output node_feature before final MLP gather',
                    gather_index='memory_entity_ids on valid slots',padding='zero after gather'))
