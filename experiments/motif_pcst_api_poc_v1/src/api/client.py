import json
from time import perf_counter, sleep

from src.api.budget_guard import BudgetExhausted, Ledger
from src.api.cache import history, completed
from src.api.provider_adapter import Provider, ProviderError, UncertainDelivery, ALLOWED_RETRY
from src.api.response_validator import parse
from src.evidence_serializer import estimate, MAX_INPUT
from src.io_utils import identity


def read_env(path):
    if not path.is_file():
        raise FileNotFoundError('experiment .env.local is absent')
    pairs = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, sep, value = line.partition('=')
            if sep:
                pairs[name.strip()] = value.strip()
    return pairs


class Client:
    # One process and one in-flight request. Runner takes a local filesystem lock.
    def __init__(self, provider, ledger, cooldown=sleep):
        self.provider = provider
        self.ledger = ledger
        self.cooldown = cooldown

    @classmethod
    def open(cls, env, ledger):
        return cls(Provider(read_env(env)), Ledger(ledger))

    def preflight(self):
        key = identity(['preflight models', self.provider.base, self.provider.model])
        if any(row['status'] == 'configured' for row in history(self.ledger, key)):
            return
        i = self.ledger.reserve(key, 0, 0)
        start = perf_counter()
        try:
            if self.provider.model not in self.provider.models():
                raise ValueError('configured model not present in provider model list')
        except (ValueError, ProviderError, UncertainDelivery):
            self.ledger.finish(i, 'preflight_failed', 0, 0, None, perf_counter()-start)
            raise
        self.ledger.finish(i, 'configured', 0, 0, None, perf_counter()-start)

    def rank(self, query_id, candidates, prompt, evidence, evidence_hash):
        input_cap = estimate(prompt)
        if input_cap > MAX_INPUT:
            raise ValueError('prompt too long')
        key = identity(dict(provider=self.provider.base, model=self.provider.model, query_id=query_id,
                            candidate_ids=list(candidates.items()), prompt_hash=identity(prompt),
                            evidence_hash=evidence_hash, temperature=self.provider.generation['temperature'],
                            generation_config=self.provider.generation))
        rows = history(self.ledger, key)
        found = completed(rows)
        if found is not None:
            return dict(found, cache_hit=True, cache_key=key)
        if any(r['status'] in ('pending','uncertain') for r in rows):
            return self.fallback(key, 'uncertain', rows)
        if any(r['status']=='fatal_http' for r in rows):
            raise RuntimeError('API request previously rejected. Check account/configuration before retrying.')
        for retry in range(len(rows), 1+self.provider.max_retries):
            # 512 is the configured per-call ceiling. Server usage releases the unused reservation.
            cap = self.provider.generation['max_tokens']
            i = self.ledger.reserve(key, input_cap, cap)
            start = perf_counter()
            try:
                result = self.provider.complete(prompt, cap)
            except ProviderError as exc:
                status = 'transient_http' if exc.status in ALLOWED_RETRY else 'fatal_http'
                self.ledger.finish(i, status, 0 if exc.status < 500 else None,
                                   0 if exc.status < 500 else None, None, perf_counter()-start)
                if status == 'fatal_http':
                    raise
            except UncertainDelivery:
                self.ledger.finish(i, 'uncertain', None, None, None, perf_counter()-start)
                return self.fallback(key, 'uncertain', history(self.ledger, key))
            else:
                response = parse(result, candidates, evidence)
                latency = perf_counter() - start
                usage = result.get('usage') or {} if isinstance(result, dict) else {}
                supplied_a, supplied_b = usage.get('prompt_tokens'), usage.get('completion_tokens')
                valid_a = supplied_a if type(supplied_a) is int and supplied_a >= 0 else None
                valid_b = supplied_b if type(supplied_b) is int and supplied_b >= 0 else None
                status = 'invalid' if response is None else 'abstain' if response['abstain'] else 'valid'
                value = dict(status=status, response=response, seconds=latency,
                             input_tokens=input_cap if valid_a is None else valid_a,
                             output_tokens=cap if valid_b is None else valid_b,
                             estimated=valid_a is None or valid_b is None)
                self.ledger.finish(i, status, valid_a, valid_b, json.dumps(value), latency)
                if response is not None:
                    return dict(value, cache_hit=False, cache_key=key)
            if retry < self.provider.max_retries:
                self.cooldown(2 ** retry)
        rows = history(self.ledger, key)
        return self.fallback(key, 'invalid' if any(r['status']=='invalid' for r in rows) else 'unavailable', rows)

    @staticmethod
    def fallback(key, status, rows):
        return dict(status=status, response=None, input_tokens=sum(r['input'] for r in rows),
                    output_tokens=sum(r['output'] for r in rows), seconds=sum(r['latency'] or 0 for r in rows),
                    estimated=True, cache_hit=False, cache_key=key)
