import copy, json, os, pathlib, subprocess, sys, tempfile, unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from serve import cache_arguments, configure_cache
from compare_cache import compare
from cache_http import configure, request

class CacheTests(unittest.TestCase):
    def test_modes_and_space_paths(self):
        root=pathlib.Path('/tmp/runtime with spaces')
        self.assertEqual(cache_arguments(root,False),['--no-cache'])
        on=cache_arguments(root,True)
        self.assertNotIn('--no-cache',on)
        self.assertEqual(on[1],str(root/'prefix-cache'))
        self.assertEqual(on[-1],'8GB')
    def test_saved_overrides_and_rollback(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);(root/'base').mkdir();p=root/'base/settings.json'
            p.write_text(json.dumps({'auth':{'opaque':'keep'},'cache':{'enabled':False,'hot_cache_only':True,'gdn_sidecar_precision':'bf16','gdn_snapshot_storage':'embedded'}}))
            configure_cache(root,True);on=json.loads(p.read_text())
            self.assertTrue(on['cache']['enabled']);self.assertFalse(on['cache']['hot_cache_only'])
            self.assertEqual(on['cache']['gdn_sidecar_precision'],'fp32')
            self.assertEqual(on['cache']['gdn_snapshot_storage'],'auto')
            self.assertEqual(on['auth'],{'opaque':'keep'})
            configure_cache(root,False);self.assertFalse(json.loads(p.read_text())['cache']['enabled'])
            self.assertEqual(p.stat().st_mode & 0o777,0o600)
    def test_reject_cache_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);(root/'base').mkdir();(root/'prefix-cache').symlink_to('/tmp')
            with self.assertRaises(ValueError):configure_cache(root,True)
    def test_mutually_exclusive_cli(self):
        r=subprocess.run([sys.executable,str(ROOT/'scripts/serve.py'),'--state','unused','--prefix-cache','--no-cache'],capture_output=True)
        self.assertEqual(r.returncode,2)
    def test_auth_stays_out_of_payload(self):
        with patch.dict(os.environ,{'TEST_CACHE_KEY':'unit-test-only'}):configure('https://example.test/v1','TEST_CACHE_KEY')
        r=request({'model':'test'})
        self.assertEqual(r.get_header('Authorization'),'Bearer unit-test-only')
        self.assertNotIn(b'unit-test-only',r.data)
        with self.assertRaises(ValueError):configure('https://user:secret@example.test/v1','TEST_CACHE_KEY')
    def test_published_evidence_and_rejection(self):
        arms=[json.loads((ROOT/'results/cache'/n).read_text()) for n in ('baseline-before.json','candidate.json','baseline-after.json')]
        self.assertGreater(compare(*arms)['exact']['ttft_s']['reduction_pct_vs_after'],50)
        for kind in ('incomplete','cache','payload','output'):
            bad=copy.deepcopy(arms)
            if kind=='incomplete':bad[1]['complete']=False
            elif kind=='cache':bad[1]['rows'][0]['cold']['cached_tokens']=1
            elif kind=='payload':bad[1]['rows'][0]['exact']['messages']=[]
            else:bad[1]['rows'][0]['branch']['text']='wrong'
            with self.assertRaises(ValueError):compare(*bad)
    def test_comparison_rejects_invalid_measurements(self):
        arms=[json.loads((ROOT/'results/cache'/n).read_text()) for n in ('baseline-before.json','candidate.json','baseline-after.json')]
        for field,value in [('cached_tokens',-1),('cached_tokens',False),('prompt_tokens',42),('ttft_s',float('nan')),('total_s',float('inf')),('ttft_s',0),('total_s',-2)]:
            bad=copy.deepcopy(arms);bad[0]['rows'][0]['cold'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):compare(*bad)
        for config_model in (False,True):
            bad=copy.deepcopy(arms)
            if config_model:bad[1]['config']['model']='wrong-model'
            else:
                for row in bad[1]['rows']:
                    for kind in ('cold','exact','branch'):
                        for event in row[kind]['events']:
                            if 'model' in event:event['model']='wrong-model'
            with self.assertRaises(ValueError):compare(*bad)

    def test_complete_quality_evidence(self):
        for name in ('quality-off.json','quality-on.json'):
            d=json.loads((ROOT/'results/cache'/name).read_text())
            self.assertTrue(d['complete']);self.assertEqual(len(d['rows']),21)
            for row in d['rows']:
                self.assertTrue(row['pass']);self.assertEqual(row['semantic'],row['expected'])
                if name=='quality-off.json':self.assertEqual(row['cached_tokens'],0)
                elif row['warm']:self.assertGreater(row['cached_tokens'],0)
    def test_portable_help(self):
        for name in ('cache_benchmark.py','cache_quality.py','compare_cache.py','cache_policy_check.py'):
            subprocess.run([sys.executable,str(ROOT/'scripts'/name),'--help'],check=True,stdout=subprocess.DEVNULL)
