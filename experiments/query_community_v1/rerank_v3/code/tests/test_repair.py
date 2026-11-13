import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
spec=importlib.util.spec_from_file_location('strict_data',ROOT/'training/strict_data.py')
strict=importlib.util.module_from_spec(spec);spec.loader.exec_module(strict)
from src.selection import validation_partition,sample_query_groups,conservative_choice
from src.statistics import sign_flip,holm
from src.structural_reranker import grid
from src.offline import Dataset

def test_training_truth_is_invariant_to_heldout_labels():
    train=np.array([[0,0,1],[1,0,2]])
    heldout=np.array([[0,0,2]])
    expected=strict.truth_for_role(train,role='train')
    np.testing.assert_array_equal(strict.truth_for_role(train,heldout,role='train'),expected)
    np.testing.assert_array_equal(strict.truth_for_role(train,heldout+100,role='train'),expected)
    assert len(strict.truth_for_role(train,heldout,role='validation'))==3
    with pytest.raises(ValueError):strict.truth_for_role(train,heldout,role='test')

def test_truth_store_rejects_tampering(tmp_path):
    (tmp_path/'data').mkdir();path=tmp_path/'data/train.npy';np.save(path,np.array([[0,0,1]]))
    (tmp_path/'data/strict_manifest.json').write_text(json.dumps({'source':{'train':{'path':'data/train.npy','sha256':strict.sha(path)}}}))
    assert strict.load_truth(tmp_path,'source','train').shape==(1,3)
    np.save(path,np.array([[0,0,9]]))
    with pytest.raises(RuntimeError,match='fingerprint'):strict.load_truth(tmp_path,'source','train')

def test_connected_queries_and_both_directions_do_not_cross_roles():
    public=np.array([[[1,0],[2,3]],[[1,0],[4,3]],[[5,1],[6,4]],[[7,2],[8,5]],[[5,1],[9,4]]])
    p=validation_partition(public)
    keys=lambda role:{tuple(q) for i in p[role] for q in public[i]}
    assert set(p['structure'])|set(p['calibration'])==set(range(5))
    assert not keys('structure')&keys('calibration')
    assert (0 in p['structure'])==(1 in p['structure'])
    assert (2 in p['structure'])==(4 in p['structure'])
    permutation=np.array([4,1,3,0,2]);other=validation_partition(public[permutation])
    assert keys('structure')=={tuple(q) for i in other['structure'] for q in public[permutation[i]]}

def test_giant_component_does_not_force_leaking_calibration_set():
    public=np.array([[[1,0],[i+2,1]] for i in range(100)])
    p=validation_partition(public)
    assert p['calibration']==[] and p['largest_component']==100

def test_llm_sampling_is_unique_and_file_order_independent():
    rows=[(i,0,i%40,i%3) for i in range(200)]
    a=sample_query_groups(rows,limit=32);b=sample_query_groups(list(reversed(rows)),limit=32)
    assert a==b and len(a)==32 and len({key for key,aliases in a})==32
    assert any(aliases[0][0]>=32 for key,aliases in a)

def test_query_balanced_selection_is_invariant_to_duplicate_answers():
    values=np.array([[.5,.6] if i%2 else [.5,.4] for i in range(40)])
    groups=[(i,0) for i in range(40)]
    first=conservative_choice(values,groups)
    duplicate=conservative_choice(np.concatenate([values,np.repeat(values[:1],20,axis=0)]),groups+[groups[0]]*20)
    assert first['chosen']==duplicate['chosen']==0
    np.testing.assert_allclose(first['means'],duplicate['means'])

def test_insufficient_calibration_falls_back_even_on_large_apparent_gain():
    result=conservative_choice(np.tile([.1,1.],(20,1)),[(i,0) for i in range(20)],min_groups=32)
    assert result['chosen']==0 and result['reason']=='insufficient_unique_validation_queries'
    assert len(list(grid()))==3

def test_exact_rare_change_test_and_multiple_comparison_control():
    result=sign_flip(np.array([1/6-1/5,1/4-1/5,1/3-1/5,1/1-1/2,1/1-1/2]))
    assert result['exact'] and result['p']==.125
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
    assert sign_flip(np.zeros(100))['p']==1.

def test_calibration_labels_cannot_change_structure_selection_filter():
    from types import SimpleNamespace
    ds=object.__new__(Dataset);ds.g=SimpleNamespace(R=2);ds.observed={}
    ds.truth={'valid':np.array([[0,0,1],[0,0,2]])}
    ds.public={'valid':np.array([[[0,0],[1,1]],[[0,0],[2,1]]])}
    ds.filters={'valid':{(0,0):{1,2}}}
    partition={'structure':[0],'calibration':[1]}
    scores=np.array([0.,1.,2.,3.]);ids=np.arange(4)
    ds.configure_validation_roles(partition)
    first=ds.rank('valid',0,0,scores,ids,validation_role='structure')
    assert first==3
    ds.truth['valid'][1,2]=3;ds.configure_validation_roles(partition)
    assert ds.rank('valid',0,0,scores,ids,validation_role='structure')==first
    with pytest.raises(ValueError,match='Explicit validation role'):ds.rank('valid',0,0,scores,ids)
