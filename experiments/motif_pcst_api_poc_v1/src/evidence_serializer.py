"""A single answer-free prompt format shared across all arms."""
import random

from src.io_utils import SEED, identity

MAX_INPUT = 3000
SYSTEM = ('Rank candidates for an anonymous knowledge graph query using only the supplied observations. '
          'Return a JSON object with ranked_candidate_ids (all candidates exactly once), '
          'used_evidence_ids (only listed facts), abstain (boolean), and confidence '
          '(low, medium, or high). If evidence cannot justify a ranking, set abstain true. '
          'For example: {"ranked_candidate_ids":["C02","C01"],"used_evidence_ids":["E0001"],'
          '"abstain":false,"confidence":"low"}. ID values have no semantic meaning.')


def estimate(messages):
    # UTF-8 bytes plus framing margin is a conservative bound for these ASCII graph prompts.
    return sum(len(m['content'].encode('utf-8')) + 32 for m in messages) + 32


def candidate_map(query_id, candidates):
    shuffled = list(map(int, candidates))
    random.Random(identity([SEED, query_id])).shuffle(shuffled)
    return {f'C{i + 1:02}': t for i, t in enumerate(shuffled)}


def label(fact):
    return f'E{fact.id:04}'


def line(fact):
    return f'{label(fact)}: N{fact.src} --R{fact.relation}--> N{fact.dst}\n'


def messages(head, relation, candidates, evidence):
    prefix = 'Task: predict the missing tail.\n'
    prefix += f'Query: (N{head}, R{relation}, ?)\nCandidates:\n'
    prefix += ''.join(f'{c} = N{t}\n' for c, t in candidates.items())
    prefix += 'Observed evidence:\n'
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': prefix + (''.join(map(line, evidence)) or '(none)\n')}]
