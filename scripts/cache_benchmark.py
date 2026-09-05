#!/usr/bin/env python3
"""Cache follow-up benchmark. Stdlib, complete SSE evidence, no GPU imports."""
import argparse, hashlib, json, pathlib, re, statistics, time, urllib.request
from cache_http import configure, request

MODEL = None
HERE = pathlib.Path(__file__).resolve().parent

def stopped():
    if (HERE / 'STOP').exists():
        raise RuntimeError('Safe stop requested')

def stream(messages, tokens):
    stopped()
    payload = dict(model=MODEL, messages=messages, temperature=0, presence_penalty=0,
                   frequency_penalty=0, repetition_penalty=1, max_tokens=tokens,
                   stream=True, stream_options={'include_usage': True},
                   chat_template_kwargs={'enable_thinking': False})
    req = request(payload)
    start = time.perf_counter(); first = None; events = []; text = ''; usage = {}; finish = None; done = False
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'): continue
            data = line[5:].strip()
            if data == '[DONE]': done = True; continue
            event = json.loads(data); events.append(event)
            if event.get('usage'): usage = event['usage']
            for choice in event.get('choices', []):
                content = choice.get('delta', {}).get('content') or ''
                if content and first is None: first = time.perf_counter()
                text += content
                finish = choice.get('finish_reason') or finish
    end = time.perf_counter()
    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens')
    if not done or first is None or not usage or cached is None or not finish:
        raise RuntimeError(f'Incomplete response evidence: done={done}, usage={usage}, finish={finish}')
    return dict(messages=messages, started_s=start, ended_s=end, ttft_s=first-start,
                total_s=end-start, completion_tokens=usage['completion_tokens'],
                cached_tokens=cached, prompt_tokens=usage['prompt_tokens'],
                decode_tps=(usage['completion_tokens']-1)/(end-first),
                finish=finish, text=text, usage=usage, events=events)

def nonce(seed, key):
    return hashlib.sha256(f'{seed}/{key}'.encode()).hexdigest()[:24]

def main():
    global HERE, MODEL
    p=argparse.ArgumentParser(); p.add_argument('--base-url',default='http://127.0.0.1:8024/v1'); p.add_argument('--model',required=True); p.add_argument('--api-key-env',default='OPENAI_API_KEY')
    p.add_argument('--label',required=True)
    p.add_argument('--runs',type=int,default=5); p.add_argument('--prompt-seed',default='r3-cache-confirm-v2')
    p.add_argument('--prefix-words',type=int,default=12288); p.add_argument('--tokens',type=int,default=256)
    p.add_argument('--expect-cache',choices=['on','off'],default='off')
    p.add_argument('--window',type=pathlib.Path,default=HERE); a=p.parse_args(); a.mode="cache"
    configure(a.base_url,a.api_key_env); MODEL=a.model
    HERE=a.window.resolve(); HERE.mkdir(parents=True,exist_ok=True)
    if a.runs < 5 or a.prefix_words < 12288 or a.tokens != 256 or pathlib.Path(a.label).name != a.label: p.error('Require n>=5, prefix>=12288 words, 256 tokens, plain label')
    out=dict(config={**vars(a),'window':str(HERE)},rows=[],complete=False)
    dest=HERE/(a.label+'.json')
    if dest.exists(): raise RuntimeError(f'Refusing to overwrite {dest}')
    def save(): dest.write_text(json.dumps(out,indent=2))
    stream([{'role':'user','content':'Reply with OK.'}],8)
    for i in range(a.runs):
        if a.mode == 'cache':
            prefix=f'Run {nonce(a.prompt_seed,i)}. Reference notes:\n'+('alpha bravo charlie delta '*(a.prefix_words//4))
            def msg(tail): return [{'role':'user','content':prefix+'\n'+tail}]
            cold=stream(msg('Count upward from 1 to 1000. Only output integers separated by spaces.'),a.tokens)
            exact=stream(cold['messages'],a.tokens)
            branch=stream(msg('Count upward from 2 to 1000. Only output integers separated by spaces.'),a.tokens)
            checks={name:[int(x) for x in re.findall(r'\d+',r['text'])[:10]]==list(range(start,start+10))
                    for name,r,start in [('cold',cold,1),('exact',exact,1),('branch',branch,2)]}
            row=dict(cold=cold,exact=exact,branch=branch,exact_output_equal=cold['text']==exact['text'],count_checks=checks)
            out['rows'].append(row); save()
            if not all(checks.values()): raise RuntimeError('Count oracle failed; cancel cache arm')
            if a.expect_cache=='off' and any(r['cached_tokens']!=0 for r in (cold,exact,branch)):
                raise RuntimeError('Cache contamination in no-cache arm')
            if a.expect_cache=='on' and (cold['cached_tokens']!=0 or any(r['cached_tokens']<=0 for r in (exact,branch))):
                raise RuntimeError('No confirmed prefix hit: cancel cache arm')
            if any(r['completion_tokens']!=a.tokens for r in (cold,exact,branch)):
                raise RuntimeError('Output budget not reached')
        print(a.mode,i,'complete',flush=True)
    if a.mode=='cache':
        out['summary']={k:statistics.median(r[k]['ttft_s'] for r in out['rows']) for k in ('cold','exact','branch')}
    out['complete']=True; save(); print('COMPLETE',flush=True)

if __name__=='__main__': main()
