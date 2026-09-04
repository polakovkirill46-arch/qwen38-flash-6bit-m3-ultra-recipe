#!/usr/bin/env python3
"""Compare completed benchmark receipts. Verify identical checkpoint/hardware yourself."""
import argparse,json,statistics
from pathlib import Path
from benchmark import valid

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--min-runs',type=int,default=5);a=p.parse_args()
 b=json.loads(a.baseline.read_text());c=json.loads(a.candidate.read_text())
 if a.min_runs<3:p.error('At least 3 measured repetitions required')
 if b.get('status')!='complete' or c.get('status')!='complete':p.error('Incomplete receipt')
 print('Verify identical checkpoint, hardware and no concurrent traffic. Model IDs:',b['model'],c['model'])
 for kind,names,metric,n in [('cells',['2048','8192'],'prefill_tps',a.min_runs),('cases',['code','prose','count'],'decode_tps',3)]:
  for name in names:
   br=b[kind][name]['rows'];cr=c[kind][name]['rows']
   if min(len(br),len(cr))<n or not all(valid(x) for x in br+cr):p.error('Invalid measured rows: '+name)
   if kind=='cells':
    bp=statistics.median(x['prompt_tokens'] for x in br);cp=statistics.median(x['prompt_tokens'] for x in cr)
    if abs(cp/bp-1)>.01:p.error('Prompt sizes differ by more than1%: '+name)
   bv=statistics.median(x[metric] for x in br);cv=statistics.median(x[metric] for x in cr)
   print(f'{kind}/{name}: {bv:.2f} -> {cv:.2f} tok/s ({cv/bv-1:+.2%})')
if __name__=='__main__':main()
