#!/usr/bin/env python3
"""Cold API prefill and ordinary generation. Run baseline and candidate sequentially."""
import argparse,json,pathlib,statistics,time,uuid,os
import stream_benchmark as b

def valid(row):
 return row['prompt_tokens']>0 and row['completion_tokens']==256 and row['usage'].get('prompt_tokens_details',{}).get('cached_tokens')==0 and row['prefill_tps']>0 and row['decode_tps']>0

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-url',default='http://127.0.0.1:8024/v1');p.add_argument('--model',required=True);p.add_argument('--out',required=True,type=pathlib.Path);p.add_argument('--runs',type=int,default=5);p.add_argument('--sizes',default='256,2048,8192');p.add_argument('--api-key-env',default='OPENAI_API_KEY');a=p.parse_args()
 if a.runs<3:p.error('At least 3 measured repetitions are required')
 if a.out.exists():p.error('Output exists; use a new filename')
 b.ENDPOINT=a.base_url.rstrip('/')+'/chat/completions';b.MODEL=a.model;b.API_KEY=os.environ.get(a.api_key_env,'')
 out={'model':a.model,'started':time.time(),'tokens':256,'status':'running','cells':{},'cases':{}}
 def save():a.out.write_text(json.dumps(out,indent=2))
 for size in map(int,a.sizes.split(',')):
  cell={'rows':[]};out['cells'][str(size)]=cell
  for i in range(a.runs+1):
   row=b.stream(b.make_prompt(size,uuid.uuid4().hex),256)
   if i:cell['rows'].append(row)
   else:cell['warmup']=row
   save()
   if not valid(row):raise RuntimeError('Incomplete/cached speed response; do not compare this run')
   print('prefill',size,i,round(row['prefill_tps'],2),round(row['decode_tps'],2),flush=True)
  cell['medians']={k:statistics.median(x[k] for x in cell['rows']) for k in ['prefill_tps','decode_tps','ttft_s','prompt_tokens']};save()
 prompts={'code':'Write a Python CSV parser with support for quoted fields and escaped double quotes. Include at least six unit tests. Output only Python code, without Markdown.','prose':'Explain how a database index speeds up queries. Cover lookups, range scans, writes, memory use, and an example. Write at least 500 words.','count':'Count upward from 1. Separate integers with spaces. Do not stop before 500. Output only the integers.'}
 for name,prompt in prompts.items():
  cell={'rows':[]};out['cases'][name]=cell
  for i in range(4):
   row=b.stream('Benchmark identifier '+uuid.uuid4().hex+'. Ignore this identifier.\n'+prompt,256)
   if i:cell['rows'].append(row)
   else:cell['warmup']=row
   save()
   if not valid(row):raise RuntimeError('Incomplete/cached workload response')
   print('generation',name,i,round(row['decode_tps'],2),flush=True)
  cell['median_decode_tps']=statistics.median(x['decode_tps'] for x in cell['rows']);save()
 out['status']='complete';save()
if __name__=='__main__':main()
