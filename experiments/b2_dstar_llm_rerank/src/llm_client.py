"""At-most-once persisted requests, bounded concurrency, strict identity fallback."""
import fcntl
import json
import os
import re
import time
import urllib.request
import urllib.error
import zipfile
import xml.etree.ElementTree as ET
from common_io import ROOT, dump, identity, read
from ranking import validate

def credentials():
    key = os.environ.get('LLM_API_KEY')
    if key: return key
    path = os.environ.get('LLM_CREDENTIAL_DOCX', '/mnt/d/下载/新阶段.md.docx')
    with zipfile.ZipFile(path) as z:
        value = ''.join(ET.fromstring(z.read('word/document.xml')).itertext())
    match = re.search(r'sk-[a-zA-Z0-9]+', value)
    if not match: raise RuntimeError('LLM credential unavailable')
    return match.group()

class Client:
    def __init__(self, cache=None):
        self.base = os.environ.get('LLM_API_BASE_URL', 'https://api.deepseek.com').rstrip('/')
        self.model = os.environ.get('LLM_MODEL_NAME', 'deepseek-flash')
        self.cache = cache or ROOT / 'cache/llm_responses'
        self.timeout = int(os.environ.get('LLM_TIMEOUT_SEC', '60'))

    def call(self, query_key, messages, allowed, local_tokens, budget=2500):
        payload = dict(model=self.model, messages=messages, temperature=0, max_tokens=400,
            response_format=dict(type='json_object'), thinking=dict(type='disabled'))
        fingerprint = identity(dict(query=query_key, base=self.base, payload=payload, allowed=allowed, schema=2))
        path = self.cache / (fingerprint+'.json'); path.parent.mkdir(parents=True, exist_ok=True)
        # Persistent in-flight journal avoids double billing after an ambiguous
        # network timeout/crash. Such requests fall back, never auto-resubmit.
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists(): return dict(read(path), cache_hit=True)
            key = credentials()
            fallback = dict(fingerprint=fingerprint, status='parse_or_validation_fallback',
                decision=None, usage={}, attempted=1, local_prompt_tokens=local_tokens,
                error='inflight_or_interrupted', cache_hit=False)
            dump(path, fallback)
            began = time.monotonic()
            try:
                req = urllib.request.Request(self.base+'/chat/completions', data=json.dumps(payload).encode(),
                    headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json'})
                with urllib.request.urlopen(req, timeout=self.timeout) as response: raw = json.load(response)
                content = raw['choices'][0]['message']['content']
                obj, error = validate(content, allowed)
                usage = raw.get('usage', {})
                actual = usage.get('prompt_tokens')
                # tiktoken is a proxy for this provider, never claim exact model tokenization.
                if type(actual) is not int or actual > budget:
                    error = 'token_budget_fallback'
                result = dict(fallback, decision=obj, status=error or ('abstain' if obj['abstain'] else 'reranked'),
                    usage=usage, raw_content=content, error=error, api_model=raw.get('model'),
                    token_count_delta=None if actual is None else actual-local_tokens)
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                result = dict(fallback, error=('HTTP '+str(exc.code)) if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__)
            result['seconds'] = time.monotonic()-began
            dump(path, result)
            return result
