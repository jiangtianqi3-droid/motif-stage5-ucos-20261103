import json,re,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
ROOT=Path(__file__).resolve().parent
REV='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
def clean(text):
    text=re.sub(r'([a-z])([A-Z])',r'\1 \2',text)
    return re.sub(r'\s+',' ',re.sub(r'[_/.:\-]+',' ',text)).strip().lower()
def transparent(text):
    return bool(re.search('[a-z]',text)) and not bool(re.fullmatch(r'(?:[pqrm]\s*)?\d+(?:\s+\d+)*|[a-f0-9]{16,}|(?:rel(?:ation)?|id)\s*\d+',text))
def main():
    if (ROOT/'frozen.json').exists():raise RuntimeError('frozen inputs cannot be regenerated')
    torch.set_num_threads(4)
    mapping=json.loads((ROOT/'semantic/relation_mappings.json').read_text(encoding='utf-8'))
    texts={};masks={}
    for key,rows in mapping.items():
        names=[clean(row['label']) for row in rows]
        texts[key]=[('inverse of '+n) if row['inverse'] else n for row,n in zip(rows,names)]
        masks[key]=[transparent(n) for n in names]
        assert masks[key][:len(rows)//2]==masks[key][len(rows)//2:]
    alltexts=sorted(set(t for arr in texts.values() for t in arr));index={t:i for i,t in enumerate(alltexts)}
    path=Path.home()/'.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots'/REV
    tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    model=AutoModel.from_pretrained(str(path),local_files_only=True).eval();vectors=[]
    with torch.no_grad():
        for at in range(0,len(alltexts),64):
            tokens=tok(alltexts[at:at+64],padding=True,truncation=True,max_length=128,return_tensors='pt')
            vectors.append(torch.nn.functional.normalize(model(**tokens).last_hidden_state[:,0],dim=-1).numpy())
    vec=np.concatenate(vectors);audit=[]
    for key,arr in texts.items():
        v=vec[[index[t] for t in arr]].astype('float32');assert v.shape[1]==384
        np.savez_compressed(ROOT/'semantic'/(key+'.npz'),vectors=v,transparent=np.asarray(masks[key]),texts=np.asarray(arr))
        audit.append(dict(key=key,relations=len(v),transparent=sum(masks[key]),opaque=len(v)-sum(masks[key]),max_unit_error=float(abs(np.linalg.norm(v,axis=1)-1).max())))
    (ROOT/'semantic/texts.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'results/encoder.json').write_text(json.dumps(dict(model='BAAI/bge-small-en-v1.5',revision=REV,pooling='normalized CLS',projection=False,generated_descriptions=False,encoder_weights={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in path.glob('*.safetensors')},graphs=audit),indent=2))
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'semantic').glob('*') if p.is_file()}
    (ROOT/'analysis/semantic_pretraining_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print('NAME ENCODING COMPLETE',len(alltexts),flush=True)
if __name__=='__main__':main()
