"""Bounded feature shards and scoring; all files are under the new run cache."""
import gc
from pathlib import Path
import numpy as np
import torch
from backbone import Backbone, bidirectional
from context_features import Context
from micro_probe_calibrator import fuse, filtered_ranks
from evaluator import resources

def save_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.save(f, value)
    temp.replace(path)

def feature_shards(z, facts, lex, directory, log):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    facts_per_shard = 16 if z['n'] > 30000 else 128
    shards = []
    context = None
    for at in range(0, len(facts), facts_per_shard):
        part = facts[at:at + facts_per_shard]
        prefix = directory / f'{at:07d}'
        auxpath = prefix.with_suffix('.aux.npy')
        q = bidirectional(part, z['R'] // 2)
        if not auxpath.exists():
            if context is None:
                context = Context(z, lex)
            save_array(auxpath, context.score(q[:, :2]))
            log('auxiliary_features', done=min(at + len(part), len(facts)), total=len(facts))
        shards.append(dict(prefix=prefix, facts=part, queries=q))
        resources()
    del context
    gc.collect()
    return shards

def backbone_shards(z, shards, family, seed, log, adaptive=False, replay=False):
    tag = f'{family}_{seed}' + ('_adaptive' if adaptive else '')
    if replay:
        assert not adaptive
        tag += '_stage7'
        paths = [s['prefix'].with_suffix('.' + tag + '.npy') for s in shards]
        if all(p.exists() for p in paths):
            return tag
        from stage7_replay import Stage7Replay
        model = Stage7Replay(z, family, seed)
        # Populate the relation cache in the same order, including existing shards.
        for shard, path in zip(shards, paths):
            score = model.scores(shard['facts'])
            if path.exists():
                assert np.array_equal(np.load(path), score), f'Replay changed saved scores: {path}'
            else:
                save_array(path, score)
            log('backbone_features', family=family, seed=seed, adaptive=False, shard=shard['prefix'].name, replay=True)
        model.close()
        return tag
    model = None
    for shard in shards:
        path = shard['prefix'].with_suffix('.' + tag + '.npy')
        if not path.exists():
            if model is None:
                model = Backbone(z, family, seed, adaptive=adaptive)
            save_array(path, model.scores(shard['facts']))
            log('backbone_features', family=family, seed=seed, adaptive=adaptive, shard=shard['prefix'].name)
        resources()
    if model is not None:
        model.close()
    return tag

def evaluate_shards(shards, tag, truth, degree, weights, pure=False):
    result = []
    for shard in shards:
        raw = np.load(shard['prefix'].with_suffix('.' + tag + '.npy'))
        aux = np.load(shard['prefix'].with_suffix('.aux.npy'))
        q = shard['queries']
        scores = fuse(raw, aux, degree[q[:, 0]], weights, text_free=pure)
        result.append(filtered_ranks(scores, q, truth))
        del raw, aux, scores
        resources()
    return np.concatenate(result)

def fold_evaluator(folds, tag, pure):
    def evaluate(weights):
        return [float(np.mean(1 / evaluate_shards(f['shards'], tag, f['truth'], f['degree'], weights, pure))) for f in folds]
    return evaluate
