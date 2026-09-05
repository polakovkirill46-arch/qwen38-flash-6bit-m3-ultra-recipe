#!/usr/bin/env python3
"""Cached continuation, divergent branch, and tool checks."""
import argparse, hashlib, json, pathlib, time, urllib.request
from cache_http import configure, request as http_request
TOOL={'type':'function','function':{'name':'get_weather','description':'Get weather for a city.','parameters':{'type':'object','properties':{'city':{'type':'string'},'units':{'type':'string','enum':['celsius','fahrenheit']}},'required':['city','units'],'additionalProperties':False}}}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--base-url',default='http://127.0.0.1:8024/v1'); p.add_argument('--model',required=True); p.add_argument('--api-key-env',default='OPENAI_API_KEY'); p.add_argument('--runs',type=int,default=3)
 p.add_argument('--label',required=True); p.add_argument('--window',type=pathlib.Path,required=True)
 p.add_argument('--expect-cache',choices=['on','off'],required=True); p.add_argument('--prompt-seed',default='r3-cache-quality-confirm-v2')
 p.add_argument('--prefix-words',type=int,default=12288); p.add_argument('--reference',type=pathlib.Path)
 a=p.parse_args()
 if a.runs<3 or a.prefix_words<12288 or pathlib.Path(a.label).name!=a.label:p.error('Require n>=3, prefix>=12288 words, plain label')
 configure(a.base_url,a.api_key_env)
 a.window.mkdir(parents=True,exist_ok=True); dest=a.window/(a.label+'.json')
 if dest.exists():raise RuntimeError('Refusing to overwrite '+str(dest))
 out={'config':{**vars(a),'window':str(a.window),'reference':str(a.reference) if a.reference else None},'rows':[],'complete':False}
 reference=json.loads(a.reference.read_text()) if a.reference else None
 if reference and (not reference.get('complete') or reference['config']['prompt_seed']!=a.prompt_seed or reference['config']['prefix_words']!=a.prefix_words):raise RuntimeError('Invalid reference')
 def save():dest.write_text(json.dumps(out,indent=2))
 def request(case,run,messages,expected,*,tools=False,force=False,warm=False):
  if (a.window/'STOP').exists():raise RuntimeError('Safe stop requested')
  payload={'model':a.model,'messages':messages,'temperature':0,'presence_penalty':0,'frequency_penalty':0,'repetition_penalty':1,'max_tokens':128,'chat_template_kwargs':{'enable_thinking':False}}
  if tools:payload.update(tools=[TOOL],tool_choice={'type':'function','function':{'name':'get_weather'}} if force else 'auto')
  start=time.perf_counter()
  req=http_request(payload)
  with urllib.request.urlopen(req,timeout=300) as r:response=json.load(r)
  elapsed=time.perf_counter()-start; msg=response['choices'][0]['message']; usage=response.get('usage',{}); cached=usage.get('prompt_tokens_details',{}).get('cached_tokens')
  semantic=None
  try:
   if force:
    calls=msg['tool_calls']; assert len(calls)==1
    semantic={'name':calls[0]['function']['name'],'arguments':json.loads(calls[0]['function']['arguments'])}
   else:semantic=json.loads(msg.get('content',''))
  except (ValueError,KeyError,AssertionError,TypeError):pass
  ok=semantic==expected and cached is not None and response['choices'][0].get('finish_reason')==('tool_calls' if force else 'stop')
  if not force:ok=ok and not msg.get('tool_calls')
  cache_ok=cached is not None and (cached==0 if a.expect_cache=='off' else (cached>0 if warm else True))
  row={'case':case,'run':run,'warm':warm,'pass':ok and cache_ok,'semantic':semantic,'expected':expected,'cached_tokens':cached,'total_s':elapsed,'payload':payload,'response':response}
  if reference:
   refs=[x for x in reference['rows'] if x['case']==case and x['run']==run]
   row['reference_equal']=len(refs)==1 and refs[0]['semantic']==semantic
   row['pass']=row['pass'] and row['reference_equal']
  out['rows'].append(row);save();print(case,run,'PASS' if row['pass'] else 'FAIL','cached',cached,flush=True)
  if not row['pass']:raise RuntimeError('Cache quality or cache-evidence failure')
  return msg
 for i in range(a.runs):
  nonce=hashlib.sha256(f'{a.prompt_seed}/{i}'.encode()).hexdigest()[:24]
  prefix=f'Run identifier {nonce}. Ignore reference filler. Secret marker is cedar-{7300+i}.\n'+('alpha bravo charlie delta '*(a.prefix_words//4))
  seed=[{'role':'user','content':prefix+'\nReturn only JSON with key marker and the secret marker as its value.'}]
  marker=f'cedar-{7300+i}'; first=request('seed',i,seed,{'marker':marker})
  history=seed+[{'role':'assistant','content':first['content']}]
  request('continuation',i,history+[{'role':'user','content':'Recall the secret marker from my first message. Return only JSON with key marker.'}],{'marker':marker},warm=True)
  request('branch_a',i,history+[{'role':'user','content':'For this branch replace the marker with amber-442. Return only JSON with key marker.'}],{'marker':'amber-442'},warm=True)
  request('branch_b',i,history+[{'role':'user','content':'This is a separate branch. Recall the original secret marker from my first message. Return only JSON with key marker.'}],{'marker':marker},warm=True)
  tool_seed=[{'role':'user','content':prefix+'\nDo not call a tool yet. Return exactly this JSON object: {"ready": true}. No other text.'}]
  ready=request('tool_seed',i,tool_seed,{'ready':True},tools=True)
  tool_history=tool_seed+[{'role':'assistant','content':ready['content']},{'role':'user','content':'Use get_weather for Tokyo in celsius.'}]
  call=request('forced_tool',i,tool_history,{'name':'get_weather','arguments':{'city':'Tokyo','units':'celsius'}},tools=True,force=True,warm=True)
  assistant={'role':'assistant','content':call.get('content'),'tool_calls':call['tool_calls']}
  follow=tool_history+[assistant,{'role':'tool','tool_call_id':call['tool_calls'][0]['id'],'content':'{"city":"Tokyo","temperature":21,"units":"celsius"}'},{'role':'user','content':'From that tool result return only JSON with keys city and temperature.'}]
  request('tool_result',i,follow,{'city':'Tokyo','temperature':21},tools=True,warm=True)
 out['complete']=True;out['passed']=sum(r['pass'] for r in out['rows']);save();print('COMPLETE',out['passed'],flush=True)

if __name__=='__main__':main()
