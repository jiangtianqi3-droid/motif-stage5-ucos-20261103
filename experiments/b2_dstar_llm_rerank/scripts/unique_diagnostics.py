from common_io import MODES, SEEDS, read

def summarize(directory):
    result={m:dict(unique_queries=0,triggered=0,abstain=0,validation_fallback=0,budget_fallback=0,
        no_evidence=0,prompt_tokens=0,completion_tokens=0) for m in MODES}
    for seed in SEEDS:
        for path in (directory/str(seed)).glob('*.json'):
            record=read(path)
            if 'modes' not in record:continue
            for mode,item in record['modes'].items():
                d=result[mode];d['unique_queries']+=1
                d['triggered']+=int(item.get('attempted',0)>0)
                d['abstain']+=int(item['status']=='abstain')
                d['validation_fallback']+=int(item['status']=='parse_or_validation_fallback')
                d['budget_fallback']+=int(item['status']=='token_budget_fallback')
                d['no_evidence']+=int(item['status']=='no_evidence')
                d['prompt_tokens']+=item.get('usage',{}).get('prompt_tokens',0)
                d['completion_tokens']+=item.get('usage',{}).get('completion_tokens',0)
    for d in result.values():
        d['trigger_rate']=d['triggered']/max(1,d['unique_queries'])
        d['abstain_rate']=d['abstain']/max(1,d['triggered'])
        d['fallback_rate']=d['validation_fallback']/max(1,d['triggered'])
    return result
