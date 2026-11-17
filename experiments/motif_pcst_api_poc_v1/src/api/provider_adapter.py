import json
import urllib.error
import urllib.request
from urllib.parse import urlparse

ALLOWED_RETRY = {429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f'API HTTP {status}')


class UncertainDelivery(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Provider:
    def __init__(self, env):
        if env.get('LLM_API_TYPE') != 'openai_compatible':
            raise ValueError('unsupported LLM_API_TYPE')
        self.key = env.get('LLM_API_KEY', '')
        self.model = env.get('LLM_MODEL', '')
        self.base = env.get('LLM_API_BASE_URL', '').rstrip('/')
        parsed = urlparse(self.base)
        if not self.key or not self.model or parsed.scheme != 'https' or parsed.username or parsed.password:
            raise ValueError('invalid API key, model, or HTTPS endpoint configuration')
        if parsed.query or parsed.fragment or not parsed.hostname:
            raise ValueError('invalid API endpoint')
        self.timeout = int(env.get('LLM_TIMEOUT_SECONDS', 120))
        self.max_retries = int(env.get('LLM_MAX_RETRIES', 2))
        self.generation = dict(max_tokens=int(env.get('LLM_MAX_OUTPUT_TOKENS', 512)),
                               temperature=float(env.get('LLM_TEMPERATURE', 0)),
                               response_format={'type':'json_object'})
        if not 0 <= self.max_retries <= 2 or not 1 <= self.generation['max_tokens'] <= 512:
            raise ValueError('retry/output configuration exceeds the experiment limits')
        if parsed.hostname == 'api.deepseek.com':
            self.generation['reasoning_effort'] = 'none'
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, route, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base + '/' + route, data=body, headers={
            'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        try:
            with self.opener.open(req, timeout=self.timeout) as result:
                return json.load(result)
        except urllib.error.HTTPError as exc:
            raise ProviderError(exc.code) from None
        except (OSError, ValueError) as exc:
            raise UncertainDelivery(type(exc).__name__) from None

    def models(self):
        result = self.request('models')
        return {row['id'] for row in result.get('data', [])}

    def complete(self, messages, output_cap):
        return self.request('chat/completions', dict(self.generation, model=self.model,
                                                    messages=messages, max_tokens=output_cap))
