import json, pathlib, sys, tempfile, unittest, subprocess, shutil, os, struct, socket
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import outside, environment, write_json, sha, validate_numerical_receipt, model_snapshot, runtime_snapshot, verify_runtime
from prepare_model import validate_config,header
from serve import check_port
class RecipeTests(unittest.TestCase):
    def test_port_probe_rejects_live_listener(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0));server.listen(1)
            with self.assertRaises(OSError):check_port('127.0.0.1',server.getsockname()[1])
    def test_port_probe_allows_time_wait_restart(self):
        with socket.socket() as server, socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.settimeout(2);client.settimeout(2)
            server.bind(('127.0.0.1',0));port=server.getsockname()[1];server.listen(1)
            client.connect(('127.0.0.1',port));accepted,_=server.accept()
            # Server performs the active close and therefore owns TIME_WAIT.
            accepted.close();self.assertEqual(client.recv(1),b'')
        with socket.socket() as plain:
            with self.assertRaises(OSError):plain.bind(('127.0.0.1',port))
        check_port('127.0.0.1',port)
    def test_numerical_receipt_rejects_stale_or_fake_success(self):
        good={'success':True,'artifact_fingerprint':'abc','mlx':'0.32.2','weighted10_bit_exact':True,
              'native_imports':['decode_fast','glm_moe_dsa','qwen35_prefill'],
              'cells':[{'tokens':t,'mix_bit_exact':True,'normalization_relative_rms':0.0} for t in [3256,8192]]}
        validate_numerical_receipt(good,'abc')
        for updates in [{'success':False},{'artifact_fingerprint':'old'},{'cells':[]},{'weighted10_bit_exact':False}]:
            with self.assertRaises(RuntimeError):validate_numerical_receipt(dict(good,**updates),'abc')
        bad=json.loads(json.dumps(good));bad['cells'][0]['normalization_relative_rms']=float('nan')
        with self.assertRaises(RuntimeError):validate_numerical_receipt(bad,'abc')
    def test_native_and_helper_drift(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t);root=p/'state';repo=p/'repo'
            for d in [root/'source/omlx',root/'mlx',repo/'runtime',repo/'scripts']:d.mkdir(parents=True)
            for name in ['common.py','build.py','prepare_model.py','serve.py','numerical_checks.py']:(repo/'scripts'/name).write_text('test')
            native=root/'source/omlx/test.so';native.write_bytes(b'original')
            helper=repo/'runtime/helper.py';helper.write_text('original');python=p/'python';python.write_bytes(b'python')
            data={'python':str(python)}
            with patch('common.REPO',repo):
                data['runtime_snapshot']=runtime_snapshot(root,data);verify_runtime(root,data)
                native.write_bytes(b'changed')
                with self.assertRaises(RuntimeError):verify_runtime(root,data)
                native.write_bytes(b'original');helper.write_text('changed')
                with self.assertRaises(RuntimeError):verify_runtime(root,data)
    def test_view_link_redirection_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t);root=p/'state';target=p/'target';view=root/'models/model'
            target.mkdir();view.mkdir(parents=True);(root/'base').mkdir()
            names=['config.json','model.safetensors.index.json','shard.safetensors']
            before={};paths={}
            for name in names:
                f=target/name;f.write_bytes(b'original');st=f.stat()
                before[name]={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino};paths[name]=str(f.resolve())
                if name.endswith('.json'):(view/name).write_bytes(b'prepared')
                else:(view/name).symlink_to(f)
            (root/'draft.safetensors').write_bytes(b'draft');(view/'draft.safetensors').symlink_to(root/'draft.safetensors')
            (root/'base/model_settings.json').write_text('{}')
            receipt={'target':str(target),'model':'model','source_file_metadata':before,'source_paths':paths}
            model_snapshot(root,receipt)
            wrong=p/'wrong';wrong.write_bytes(b'original');(view/'shard.safetensors').unlink();(view/'shard.safetensors').symlink_to(wrong)
            with self.assertRaises(RuntimeError):model_snapshot(root,receipt)
    def test_exclusive_metadata_write(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'metadata.json';write_json(p,{'original':True})
            with self.assertRaises(FileExistsError):write_json(p,{'changed':True})
            self.assertEqual(json.loads(p.read_text()),{'original':True})
    def test_environment_drops_experiment_flags(self):
        with patch.dict(os.environ,{'FLASH_HC_FUSE':'1','OMLX_GDN_FUSED_G_BETA':'1','MLX_MAX_MB_PER_BUFFER':'9999'}):
            env=environment(pathlib.Path('/tmp/state'),{'app':'/Applications/oMLX.app'})
        for k in ['FLASH_HC_FUSE','OMLX_GDN_FUSED_G_BETA','MLX_MAX_MB_PER_BUFFER']:self.assertNotIn(k,env)
    def test_header_read_is_immutable(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'tensor';h=json.dumps({'a':{'dtype':'BF16','shape':[1],'data_offsets':[0,2]}}).encode()
            p.write_bytes(struct.pack('<Q',len(h))+h+b'\0\0');before=sha(p)
            self.assertEqual(header(p)[1]['a']['shape'],[1]);self.assertEqual(before,sha(p))
    def test_disjoint_paths(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)
            for a,b in [(p,p),(p/'child',p),(p,p/'child')]:
                with self.assertRaises(ValueError):outside(a,b)
            outside(p/'one',p/'two')
    def test_reject_other_model(self):
        with self.assertRaises(ValueError):validate_config({'model_type':'qwen4_exp','quantization':{'bits':4}})
    def test_reject_bad_header(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'bad';p.write_bytes(b'bad')
            with self.assertRaises(ValueError):header(p)
    def test_portability_and_no_binaries(self):
        for folder in ['scripts','runtime','patches']:
            for p in (ROOT/folder).rglob('*'):
                if not p.is_file() or '__pycache__' in p.parts:continue
                self.assertNotIn(p.suffix,['.so','.dylib','.safetensors','.metallib'])
                s=p.read_text();self.assertNotIn('/Users/',s);self.assertNotIn('Projects/experiments/',s)
    def test_qualified_environment(self):
        env=environment(pathlib.Path('/tmp/install with spaces'),{'app':'/Applications/oMLX.app'})
        self.assertEqual(env['FLASH_HC_NORM_THREADS'],'256')
        self.assertEqual(env['FLASH_ANE_CPU'],'0.125')
        self.assertNotIn('FLASH_HC_FUSE',env)
        self.assertEqual(env['FLASH_RECIPE_ROOT'],str(ROOT))
    def test_cli_help_without_mlx(self):
        for script in ['build.py','prepare_model.py','serve.py','numerical_checks.py']:
            subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--help'],check=True,stdout=subprocess.DEVNULL)
if __name__=='__main__':unittest.main()
