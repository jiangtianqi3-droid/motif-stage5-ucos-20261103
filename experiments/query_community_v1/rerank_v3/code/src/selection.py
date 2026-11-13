"""Label-free partitioning and conservative, query-balanced model selection."""
import hashlib
from collections import defaultdict
import numpy as np

SEED=20261112

def query_key(public,i,d):return tuple(map(int,public[int(i),int(d)]))

def validation_partition(public, max_facts_per_role=512, seed=SEED):
    """Keep both directions and all facts sharing a public query in one role."""
    public=np.asarray(public);n=len(public);parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i in range(n):
        for d in (0,1):
            key=query_key(public,i,d)
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
    components=defaultdict(list)
    for i in range(n):components[find(i)].append(i)
    # Stable by public content, not file ordering, then seeded tie-breaking.
    def component_key(rows):
        keys=sorted({query_key(public,i,d) for i in rows for d in (0,1)})
        return hashlib.sha256(repr((seed,keys)).encode()).hexdigest()
    ordered=sorted(components.values(),key=lambda c:(-len(c),component_key(c)))
    roles={'structure':[],'calibration':[]}
    for component in ordered:
        role=min(roles,key=lambda k:len(roles[k]));roles[role].extend(component)
    rng=np.random.default_rng(seed)
    for role,rows in roles.items():
        rows=sorted(rows)
        if len(rows)>max_facts_per_role:rows=rng.choice(rows,max_facts_per_role,replace=False).tolist()
        roles[role]=sorted(map(int,rows))
    keys={role:{query_key(public,i,d) for i in rows for d in (0,1)} for role,rows in roles.items()}
    assert not keys['structure']&keys['calibration']
    assert not set(roles['structure'])&set(roles['calibration'])
    return dict(**roles,seed=seed,components=len(components),largest_component=max(map(len,ordered),default=0),
                unique_queries={k:len(v) for k,v in keys.items()},query_overlap=0,fact_overlap=0,
                policy='whole connected fact components by shared public query; balanced roles; seeded bounded within-role sample')

def grouped_matrix(values,groups):
    values=np.asarray(values,float);buckets=defaultdict(list)
    for j,key in enumerate(groups):buckets[tuple(key)].append(j)
    if not buckets:return np.empty((0,values.shape[1]))
    return np.array([values[js].mean(0) for js in buckets.values()])

def conservative_choice(values,groups,min_groups=30,seed=SEED):
    """Paired one-SE simplicity rule, with 200 resamples as a stability diagnostic."""
    x=grouped_matrix(values,groups);n=len(x)
    if not n:return dict(chosen=0,groups=0,reason='no_independent_validation_queries',means=[],stability=[])
    means=x.mean(0);best=int(np.argmax(means));differences=x[:,best,None]-x
    se=differences.std(0,ddof=1)/np.sqrt(n) if n>1 else np.zeros(x.shape[1])
    eligible=np.flatnonzero(means[best]-means<=se+1e-12)
    chosen=int(eligible[0])
    reason='paired_one_standard_error_simplest'
    if n<min_groups:chosen=0;reason='insufficient_unique_validation_queries'
    counts=np.zeros(x.shape[1],dtype=int);rng=np.random.default_rng(seed)
    for _ in range(200):
        take=x[rng.integers(n,size=n)];mu=take.mean(0);b=int(np.argmax(mu))
        err=(take[:,b,None]-take).std(0,ddof=1)/np.sqrt(n) if n>1 else np.zeros(x.shape[1])
        c=int(np.flatnonzero(mu[b]-mu<=err+1e-12)[0]) if n>=min_groups else 0
        counts[c]+=1
    return dict(chosen=chosen,groups=n,reason=reason,means=means.tolist(),best_raw=best,
                paired_standard_errors=se.tolist(),stability=(counts/200).tolist(),resamples=200)

def sample_query_groups(rows,limit=None,seed=SEED):
    """rows=(i,d,head,relation); uniform unique-query sample, independent of row order."""
    groups=defaultdict(list)
    for i,d,h,r in rows:groups[int(h),int(r)].append((int(i),int(d)))
    keys=sorted(groups);rng=np.random.default_rng(seed)
    if limit is not None and len(keys)>limit:
        keys=[keys[j] for j in rng.choice(len(keys),limit,replace=False)]
    else:
        keys=[keys[j] for j in rng.permutation(len(keys))]
    return [(key,sorted(groups[key])) for key in keys]
