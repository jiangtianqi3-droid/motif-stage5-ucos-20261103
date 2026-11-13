from .llm_reranker import Client
from .util import dump, memory

def api_smoke(run):
    client=Client()
    messages=[dict(role='system',content='Return JSON only with exactly these keys: abstain (boolean), top_candidates (array), reasoning (string). There is no evidence: abstain must be true, top_candidates must be empty.'),
              dict(role='user',content='No knowledge graph evidence is available. Produce your JSON decision.')]
    first=client.call(messages,[])
    if first['parse_error'] or not first['decision']['abstain']:raise RuntimeError('API connectivity response failed schema/abstention check')
    calls=client.network_requests;second=client.call(messages,[])
    assert second['cache_hit'] and client.network_requests==calls
    dump(run/'api_smoke.json',dict(passed=True,model=first['model'],initial_network_requests=calls,replay_network_requests=0,
                                  rss_bytes=memory(),usage=first['usage'],decision=first['decision']))
