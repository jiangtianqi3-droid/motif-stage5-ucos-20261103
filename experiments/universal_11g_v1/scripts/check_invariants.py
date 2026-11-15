import numpy as np
from context_features import Context
from micro_probe_calibrator import fuse, filtered_ranks, micro_folds, calibrate
from regime_inspector import readable, save_json, ROOT

def main():
    raw = np.asarray([[1., np.nextafter(np.float32(1.), np.float32(2.)), 1.], [0., 0., 0.]], np.float32)
    aux = np.ones((2, 5, 3), np.float32)
    assert fuse(raw, aux, [1, 2], [0] * 5) is raw
    assert np.array_equal(fuse(raw, aux, [1, 2], [0, 0, 9, 0, 1], text_free=True), raw)
    q = np.asarray([[0, 0, 0], [1, 0, 1]])
    assert np.array_equal(filtered_ranks(raw, q, {}), [3, 3])
    facts = np.asarray([[i, 0, (i + 1) % 100] for i in range(100)])
    folds = micro_folds(facts, 100)
    assert len(folds) == 5 and all(len(f) == 5 for f in folds)
    assert len(set(np.concatenate(folds))) == 25
    guard = calibrate(lambda w: [.4] * 5)
    assert not guard['accepted'] and guard['weights'] == [0] * 5
    guard = calibrate(lambda w: [1., 1., .4, .4, .4] if w[0] else [.5] * 5)
    assert guard['self_gain'] > 0 and not guard['accepted'], 'Mean gain alone must not bypass 3-of-5 guard'
    guard = calibrate(lambda w: [.7] * 5, reference=[.8] * 5)
    assert not guard['accepted'], 'Adaptive fallback must compare against original anchor'
    p1 = calibrate(lambda w: [.7] if w[0] else [.5], guard=False)
    assert p1['accepted'] and p1['weights'][0] > 0
    for token in ['0', '1234', 'Q1008', 'P54', '/m/0102t4', '01089137']:
        assert not readable(token)
    assert readable('concept:animal:dog')
    z = dict(n=3, R=2, edge_index=np.asarray([[0, 1, 1, 2], [1, 2, 0, 1]]), edge_type=np.asarray([0, 0, 1, 1]),
             hyper_edge=np.zeros((3, 0), dtype=np.int64), hyper_type=np.zeros(0, dtype=np.int64), raw=np.zeros((0, 4)))
    context = Context(z)
    f = context.score(np.asarray([[0, 0], [2, 1]]))
    assert f.shape == (2, 5, 3) and np.isfinite(f).all()
    assert not f[:, 2].any() and np.array_equal(f[:, 3], f[:, 4])
    lexical = np.asarray([[1., 0.], [0., 1.], [.5, .5]], np.float32)
    dual = Context(z, lexical).score(np.asarray([[0, 0], [2, 1]]))
    assert np.array_equal(f[:, [0, 1, 3]], dual[:, [0, 1, 3]]), 'Lexical changes must not alter pure structural features'
    from reporting import values
    sample = np.asarray([[1, 2, 4, 8], [2, 4, 8, 16]])
    expected = (1 / sample).mean()
    assert np.isclose(values(sample)[:, 0].mean(), expected), 'Metric mean is not mean-rank scoring'
    save_json(ROOT / 'results/invariants.json', dict(passed=True, checks=['bitwise zero anchor', 'pure structural isolation', 'pessimistic ties', 'disjoint nonisolating masks', 'fallback', 'anonymous text rejection', 'finite sparse context']))
    print('All invariant checks passed')

if __name__ == '__main__':
    main()
