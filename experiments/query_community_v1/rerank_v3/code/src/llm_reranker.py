"""OpenAI-compatible requests; schema rejection is an exact identity fallback."""
import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from .util import ROOT, HISTORY, dump, identity, read

def credentials():
    # The supplied credential remains in its source document, never copied to code/logs.
    if not os.environ.get('LLM_API_KEY'):
        path=os.environ.get('LLM_CREDENTIAL_DOCX','/mnt/d/下载/新阶段.md.docx' if os.name!='nt' else 'D:/下载/新阶段.md.docx')
        with zipfile.ZipFile(path) as z:
            text=''.join(ET.fromstring(z.read('word/document.xml')).itertext())
        match=re.search(r'sk-[a-zA-Z0-9]+',text)
        if not match: raise RuntimeError('No LLM credential available')
        os.environ['LLM_API_KEY']=match.group()
    return os.environ['LLM_API_KEY']

def validate(content, allowed):
    try:
        obj=json.loads(content)
        if not isinstance(obj,dict) or type(obj.get('abstain')) is not bool: raise ValueError('abstain must be boolean')
        if not isinstance(obj.get('reasoning'),str): raise ValueError('reasoning missing')
        rows=obj.get('top_candidates')
        if not isinstance(rows,list): raise ValueError('top_candidates must be list')
        seen=set()
        for row in rows:
            i=row.get('candidate_id'); c=row.get('confidence')
            if type(i) is not int or i not in allowed or i in seen: raise ValueError('unknown or duplicated candidate')
            if isinstance(c,bool) or not isinstance(c,(int,float)) or not 0<=c<=1: raise ValueError('invalid confidence')
            seen.add(i)
        if not obj['abstain'] and not rows: raise ValueError('non-abstain without confidence')
        return obj, None
    except (ValueError,TypeError,KeyError,AttributeError) as exc:
        return dict(abstain=True,top_candidates=[],reasoning='Invalid response; retain structural ordering'),str(exc)

class Client:
    def __init__(self, cache=None):
        self.base=os.environ.get('LLM_API_BASE_URL','https://api.deepseek.com').rstrip('/')
        self.model=os.environ.get('LLM_MODEL_NAME','deepseek-flash')
        self.timeout=int(os.environ.get('LLM_TIMEOUT_SEC','30'))
        self.cache=cache or ROOT/'cache/llm_responses'; self.network_requests=0
        self.key=None

    def call(self,messages,allowed):
        payload=dict(model=self.model,messages=messages,response_format=dict(type='json_object'),thinking=dict(type='disabled'),temperature=0,max_tokens=800)
        fingerprint=identity(dict(base=self.base,payload=payload,allowed=sorted(allowed),schema=1))
        path=self.cache/f'{fingerprint}.json'
        historical=HISTORY/'route-a-rerank-20261109/cache/llm_responses'/f'{fingerprint}.json'
        if not path.exists() and historical.exists():
            result=read(historical);result['cache_hit']=True;return result
        if path.exists():
            result=read(path)
            result['cache_hit']=True
            return result
        if self.key is None: self.key=credentials()
        began=time.perf_counter(); response=None; failure=None; attempts=0
        # Initial attempt plus no more than three exponential-backoff retries.
        for attempt in range(4):
            attempts+=1
            try:
                request=urllib.request.Request(self.base+'/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
                self.network_requests+=1
                with urllib.request.urlopen(request,timeout=self.timeout) as f: response=json.load(f)
                break
            except urllib.error.HTTPError as exc:
                failure=f'HTTP {exc.code}'
                if exc.code not in (408,429,500,502,503,504): break
            except (OSError,ValueError) as exc: failure=type(exc).__name__
            if attempt<3: time.sleep(2**attempt)
        if response is None:
            raise RuntimeError(f'LLM transport unavailable after {attempts} attempts: {failure}')
        content=response.get('choices',[{}])[0].get('message',{}).get('content','')
        obj,error=validate(content,set(allowed))
        result=dict(fingerprint=fingerprint,cache_hit=False,attempts=attempts,seconds=time.perf_counter()-began,usage=response.get('usage',{}),
                    model=response.get('model',self.model),decision=obj,parse_error=error,raw_content=content)
        dump(path,result)
        return result

def confidence(result):
    d=result['decision']
    return {} if d['abstain'] else {c['candidate_id']:c['confidence'] for c in d['top_candidates']}
