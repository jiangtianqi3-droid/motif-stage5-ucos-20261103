import unittest
import subprocess
import sys
import numpy as np
from stage6.observed_loader import from_forward
from stage7.proxy_density import nested_masks,remaining_graph
from stage7.proxy_environments import sample,environments
from stage7.proxy_split import residual
from stage7.confirmation import paired_rr,lcb,accept
from stage7.robust_selector import choose,ALPHAS
from stage7.util import ROOT,identity,read,sha


def graph():
    facts=np.array([(h,r,(h+d)%100) for h in range(100) for d in [1,2,3] for r in range(4)],dtype=np.int64)
    return from_forward(np.r_[facts,facts[:4]],100,8,{'synthetic':True})


class Core(unittest.TestCase):
    def test_nested_masks(self):
        g=graph();m,meta=nested_masks(g,6101)
        self.assertTrue(set(m[0])<=set(m[1])<=set(m[2]))
        self.assertEqual([len(x) for x in m],[60,120,180])
        self.assertEqual([x['added'] for x in meta],[60,60,60])
        self.assertEqual(len(g.forward),1200)

    def test_direct_inverse_and_indices(self):
        g=graph();m,_=nested_masks(g,6102)
        a={'facts':g.forward,'hidden0':m[0]};rem=residual(g,a,0)
        aug=set(zip(rem.edge_index[0],rem.edge_type,rem.edge_index[1]))
        for h,r,t in g.forward[m[0]]:
            self.assertNotIn((h,r,t),aug);self.assertNotIn((t,r+4,h),aug)
        self.assertEqual((g.n,g.R),(rem.n,rem.R))

    def test_mask_hash_and_determinism(self):
        g=graph();a,_=nested_masks(g,6101);b,_=nested_masks(g,6101);c,_=nested_masks(g,6102)
        self.assertEqual(identity(a[0]),identity(b[0]));self.assertNotEqual(identity(a[0]),identity(c[0]))

    def test_preserve_unmasked_duplicate_copies(self):
        f=np.array([[0,0,1],[0,0,1],[1,0,2],[1,0,2],[1,0,2],[2,0,0]])
        g=from_forward(f,3,2,{})
        rem=remaining_graph(g,[0],g.forward)
        raw=np.c_[rem.edge_index[0],rem.edge_type,rem.edge_index[1]]
        self.assertEqual(int(np.all(raw==[1,0,2],axis=1).sum()),3)
        self.assertEqual(int(np.all(raw==[2,1,1],axis=1).sum()),3)
        self.assertEqual(int(np.all(raw==[0,0,1],axis=1).sum()),0)
        self.assertEqual(int(np.all(raw==[1,1,0],axis=1).sum()),0)

    def test_zero_mask_keeps_all_observation_rows(self):
        g=graph();rem=remaining_graph(g,[],g.forward)
        self.assertEqual(len(rem.edge_type),len(g.edge_type))
        self.assertTrue(np.array_equal(rem.forward,g.forward))

    def test_no_duplicate_graph_exact_legacy_residual_identity(self):
        g=from_forward(graph().forward,100,8,{})
        masks,_=nested_masks(g,6101)
        keep=np.ones(len(g.forward),dtype=bool);keep[masks[0]]=False
        old=from_forward(g.forward[keep],g.n,g.R,{})
        self.assertEqual(old.key,remaining_graph(g,masks[0]).key)

    def test_no_illegal_relaxation(self):
        f=np.array([(i,0,i+1) for i in range(20)])
        g=from_forward(f,21,2,{})
        m,meta=nested_masks(g,1)
        self.assertEqual([len(v) for v in m],[0,0,0])

    def test_environment_equal_n_disjoint(self):
        g=graph();m,_=nested_masks(g,6101)
        he=np.array([[0,1,2,3],[4,5,6,7],[-1,-1,-1,-1]])
        envs,_=environments(g,m[0],he,6101)
        self.assertEqual(len({len(v['selection']) for v in envs.values()}),1)
        self.assertEqual(len({len(v['confirmation']) for v in envs.values()}),1)
        s=set(np.concatenate([v['selection'] for v in envs.values()]));c=set(np.concatenate([v['confirmation'] for v in envs.values()]))
        self.assertFalse(s&c);self.assertTrue(s|c<=set(m[0]))

    def test_balanced_sampling_and_no_duplicate(self):
        labels=np.r_[np.zeros(90),np.ones(10)]
        s=sample(np.arange(100),labels,20,np.random.default_rng(2))
        self.assertEqual(int(labels[s].sum()),10);self.assertEqual(len(set(s)),20)

    def test_fact_paired_unit(self):
        r=np.array([[[1,4],[2,2]]])
        self.assertTrue(np.array_equal(paired_rr(r),[[.625,.5]]))

    def test_lcb_accept_fail(self):
        self.assertTrue(accept([lcb(np.ones(20)*.01)]))
        self.assertFalse(accept([lcb(np.zeros(20))]))
        self.assertFalse(accept([lcb(np.ones(3))]))
        self.assertFalse(accept([lcb(np.ones(20)),lcb(-np.ones(20))]))

    def test_worst_environment_selection(self):
        gain=[[np.zeros(30),np.ones(30)*.02,np.ones(30)*.1,np.zeros(30),np.zeros(30)] for _ in range(12)]
        gain[7][2]=-np.ones(30)*.01
        d,_=choose(gain,[30]*12)
        self.assertEqual(d['robust_lcb']['candidate_alpha'],.1)
        self.assertEqual(d['equal_mean']['candidate_alpha'],.25)
        self.assertEqual(d['four_env_lcb_10']['candidate_alpha'],.1)
        self.assertEqual(d['three_density_A_lcb']['candidate_alpha'],.25)

    def test_fallback_reference(self):
        gain=[[np.zeros(30)]+[-np.ones(30)*.01 for _ in range(4)] for _ in range(12)]
        d,_=choose(gain,[30]*12)
        self.assertEqual(d['robust_lcb']['candidate_alpha'],0.)

    def test_access_guard_subprocess(self):
        code="from stage7.isolation import enable;enable();open('stage6/evaluation/forbidden.npz','rb')"
        p=subprocess.run([sys.executable,'-c',code],cwd=str(ROOT.parent),capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('PermissionError',p.stderr)
        code="from stage7.isolation import enable;enable();open('stage7/downloads/new.zip','rb')"
        p=subprocess.run([sys.executable,'-c',code],cwd=str(ROOT.parent),capture_output=True,text=True)
        self.assertIn('PermissionError',p.stderr)

    def test_closed_observation_schema(self):
        from stage6.observed_loader import load_observed
        g=load_observed(ROOT.parent/'stage6/observations','NELLInductive_v1')
        self.assertEqual(set(g.__dataclass_fields__),{'edge_index','edge_type','n','R','provenance'})
        self.assertFalse(g.edge_index.flags.writeable)


if __name__=='__main__':unittest.main()
