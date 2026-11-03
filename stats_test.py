"""Analytic bootstrap oracles for empty, singleton and two-triple cases."""
import ast,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
tree=ast.parse((ROOT/'summarize.py').read_text(encoding='utf-8'))
fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='bootstrap_matrix')
space={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<bootstrap analytic tests>','exec'),space)
bootstrap=space['bootstrap_matrix']
assert bootstrap(np.empty((0,1)),1)==[(None,None,None)]
assert bootstrap(np.zeros((4,2)),1)==[(0.,0.,0.),(0.,0.,0.)]
assert np.allclose(bootstrap(np.array([[.125]]),1),[[.125,.125,.125]])
a=np.array([[1,2],[4,4]]);b=np.array([[2,2],[4,8]])
delta=((1/a).mean(1)-(1/b).mean(1))[:,None]
actual=bootstrap(delta,1)[0];expected=[.15625,.0625,.25]
assert np.allclose(actual,expected,atol=1e-14),(actual,expected)
(ROOT/'results/statistical_tests.json').write_text(json.dumps(dict(status='passed',scope='analytic toy oracles, not experimental graph results',empty_group=True,zero_difference=True,singleton_degenerate=True,paired_two_direction_triple_oracle=dict(expected=expected,actual=actual)),indent=2),encoding='utf-8')
print('ANALYTIC BOOTSTRAP TESTS PASSED')
