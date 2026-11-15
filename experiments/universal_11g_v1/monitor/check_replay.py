"""Check the original full-size batch containing the one failed anchor query."""
import json

import numpy as np
import torch

from backbone import bidirectional, truth_index
from evaluator import resources
from micro_probe_calibrator import filtered_ranks
from protocol_data import final_labels
from regime_inspector import ROOT, STAGE7, observation, records, save_json
from stage7_replay import Stage7Replay

torch.set_num_threads(4)
rec = next(r for r in records() if r['key'] == 'ConceptNet100k_external')
z = observation(rec)
facts, alltrue = final_labels(rec)
relation_count = z['R'] // 2
truth = truth_index(alltrue, relation_count)
start = (1327 // 2 // 16) * 16
with np.load(STAGE7 / 'evaluation' / rec['key'] / 'B0+Dstar.npz') as archive:
    expected = archive['ranks'][start:start+16].reshape(-1).copy()
raw = np.load(ROOT / 'cache' / '910227618ce1db25d677' / rec['key'] / 'test' / f'{start:07d}.B0_1024.npy')
q = bidirectional(facts[start:start+16], relation_count)
previous = filtered_ranks(raw, q, truth)
model = Stage7Replay(z, 'B0', 1024)
with torch.no_grad():
    # Recreate the order in which Stage-7 first populated the relation cache.
    relation_model = model.model.relation_model
    for offset in range(0, start, 16):
        relations = torch.as_tensor(facts[offset:offset+16, 1], device='cuda')
        for relation_ids in (relations, relations + relation_count):
            relation_model(model.data.relation_hypergraph, relation_ids)
replayed = filtered_ranks(model.scores(facts[start:start+16]), q, truth)
model.close()
result = dict(historical=int(expected[1327 - start * 2]), micro_batch=int(previous[1327 - start * 2]),
              replay=int(replayed[1327 - start * 2]), batch_matches=bool(np.array_equal(replayed, expected)), **resources())
save_json(ROOT / 'results/replay_check.json', result)
print(json.dumps(result), flush=True)
assert result['batch_matches']
