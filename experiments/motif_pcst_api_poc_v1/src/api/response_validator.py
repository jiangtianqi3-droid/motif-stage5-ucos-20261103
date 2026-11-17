import json

from src.ranking_fusion import validate_response


def parse(result, candidates, evidence):
    try:
        choice = result['choices'][0]
        obj = json.loads(choice['message']['content'])
        if choice.get('finish_reason') != 'stop' or not validate_response(obj, candidates, evidence):
            return None
        if obj.get('confidence') not in ('low', 'medium', 'high'):
            return None
        return obj
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return None
