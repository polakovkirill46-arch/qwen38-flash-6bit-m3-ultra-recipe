#!/usr/bin/env python3
"""Validate matched complete cache arms, then compare median request latencies."""
import argparse, json, statistics, math
from pathlib import Path

def model_identity(arm):
    names={event['model'] for row in arm['rows'] for kind in ('cold','exact','branch')
           for event in row[kind]['events'] if event.get('model') and not (event['model']=='keepalive' and not event.get('usage')
               and all(not c.get('delta',{}).get('content') and not c.get('finish_reason') for c in event.get('choices',[])))}
    if arm['config'].get('model'): names.add(arm['config']['model'])
    if len(names)!=1 or not all(isinstance(x,str) and x for x in names):
        raise ValueError('Require one consistent model identity')
    return names.pop()

def compare(before, candidate, after):
    arms = [before, candidate, after]
    if len({model_identity(arm) for arm in arms})!=1:raise ValueError('Mismatched model identity')
    for arm, mode in zip(arms, ['off', 'on', 'off']):
        cfg = arm['config']
        if not arm.get('complete') or cfg['expect_cache'] != mode or cfg['runs'] < 5 or len(arm['rows']) != cfg['runs']:
            raise ValueError('Require complete n>=5 off/on/off arms')
        for row in arm['rows']:
            for kind in ('cold', 'exact', 'branch'):
                r = row[kind]
                if r['completion_tokens'] != 256 or r['finish'] != 'length': raise ValueError('Incomplete output budget')
                if type(r['cached_tokens']) is not int or r['cached_tokens']<0:raise ValueError('Invalid cached token count')
                if type(r['prompt_tokens']) is not int or r['prompt_tokens']<=0 or r['cached_tokens']>r['prompt_tokens']:raise ValueError('Invalid prompt token count')
                for metric in ('ttft_s','total_s'):
                    value=r[metric]
                    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Invalid timing')
                if r['total_s']<r['ttft_s']:raise ValueError('Invalid timing order')
                expected_hit = mode == 'on' and kind != 'cold'
                if (r['cached_tokens'] > 0) != expected_hit: raise ValueError('Unexpected cache status')
    for arm in arms[1:]:
        for key in ('runs', 'prompt_seed', 'prefix_words', 'tokens'):
            if arm['config'][key] != before['config'][key]: raise ValueError('Mismatched configuration')
        for a, b in zip(before['rows'], arm['rows']):
            for kind in ('cold', 'exact', 'branch'):
                if a[kind]['prompt_tokens']!=b[kind]['prompt_tokens']:raise ValueError('Prompt token count mismatch')
                if a[kind]['messages'] != b[kind]['messages'] or a[kind]['text'] != b[kind]['text']:
                    raise ValueError('Payload/output mismatch')
    result = {}
    for kind in ('cold', 'exact', 'branch'):
        result[kind] = {}
        for metric in ('ttft_s', 'total_s'):
            medians = [statistics.median(r[kind][metric] for r in arm['rows']) for arm in arms]
            result[kind][metric] = {'before': medians[0], 'candidate': medians[1], 'after': medians[2],
                'reduction_pct_vs_before': 100*(1-medians[1]/medians[0]),
                'reduction_pct_vs_after': 100*(1-medians[1]/medians[2])}
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'candidate', 'after'): p.add_argument(name, type=Path)
    a = p.parse_args()
    print(json.dumps(compare(*(json.loads(x.read_text()) for x in (a.before, a.candidate, a.after))), indent=2))
