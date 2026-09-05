#!/usr/bin/env python3
"""CPU-only integration check; run with pinned oMLX source on PYTHONPATH."""
import argparse, json, tempfile
from pathlib import Path
from serve import configure_cache, cache_arguments

def main():
    p=argparse.ArgumentParser(description=__doc__);p.parse_args()
    from omlx.settings import CacheSettings, GlobalSettings
    with tempfile.TemporaryDirectory(prefix='cache-policy-check-') as directory:
        root=Path(directory).resolve();base=root/'base';base.mkdir()
        (base/'settings.json').write_text(json.dumps({'cache':{'enabled':False,'gdn_snapshot_storage':'embedded','gdn_ssd_split_enabled':False,'gdn_sidecar_precision':'bf16'}}))
        rows=[]
        for enabled in (True,False,True):
            configure_cache(root,enabled)
            args=argparse.Namespace(paged_ssd_cache_dir=str(root/'prefix-cache') if enabled else None,
                paged_ssd_cache_max_size='8GB' if enabled else None,no_cache=not enabled,
                hot_cache_max_size='0',max_concurrent_requests=1)
            settings=GlobalSettings(base_path=base)
            settings.save_cli_overrides(args)
            saved=json.loads((base/'settings.json').read_text())
            cache=CacheSettings.from_dict(saved['cache'])
            assert cache.enabled is enabled
            assert cache.get_gdn_snapshot_storage()=='auto'
            assert cache.get_gdn_ssd_split_enabled() is enabled
            assert cache.gdn_sidecar_state_dtype=='fp32'
            assert cache.hot_cache_max_size=='0' and not cache.hot_cache_only
            assert cache.ssd_cache_max_size=='8GB'
            assert cache.get_ssd_cache_dir(base)==root/'prefix-cache'
            assert saved['scheduler']['max_concurrent_requests']==1
            rows.append({'enabled':enabled,'snapshot_storage':'auto','resolved_ssd_sidecars':cache.get_gdn_ssd_split_enabled(),'precision':'fp32'})
        print(json.dumps({'success':True,'cpu_only':True,'cases':rows},indent=2))

if __name__=='__main__':main()
