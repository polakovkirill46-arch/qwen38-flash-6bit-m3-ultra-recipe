# Reproduce the prefix cache result

Build a fresh state directory from this revision using the main README. Older build receipts reject changed launcher code by design. Keep the exact model, MTP profile, sampling, client and machine identical. Stop other inference traffic and run only one server at a time.

`serve.py` defaults to cache off. Opt in with `--prefix-cache`; use `--no-cache` to roll back on the next restart. Stop the foreground server with Ctrl-C between arms. Neither option deletes cache files. The launcher pins the measured cache settings while preserving unrelated saved settings: 8GB SSD, hot cache 0, FP32 snapshots and `auto` storage (SSD sidecars when enabled). Both the saved cache policy and explicit on/off CLI overrides prevent a previously saved cache-enabled setting from silently controlling later launches.

Set `STATE` and `TARGET` as in the README. Run each server command in one terminal and its measurement commands in another. Use an empty evidence directory and a fresh prompt seed for a new campaign; keep that seed identical across all three arms. The example defaults reproduce the published fixtures, so use them only with a fresh cache state. The scripts refuse to overwrite results and never save API keys. Authenticated endpoints read `OPENAI_API_KEY` by default.

```bash
mkdir cache-evidence
# Arm 1: start this server, then run the two measurement commands.
python3 scripts/serve.py --state "$STATE" --no-cache
```

```bash
python3 scripts/cache_benchmark.py --model "$(basename "$TARGET")" --window cache-evidence --label before --expect-cache off
python3 scripts/cache_quality.py --model "$(basename "$TARGET")" --window cache-evidence --label quality-off --expect-cache off
```

Stop that server before starting the cache-enabled arm:

```bash
python3 scripts/serve.py --state "$STATE" --prefix-cache
```

```bash
python3 scripts/cache_benchmark.py --model "$(basename "$TARGET")" --window cache-evidence --label candidate --expect-cache on
python3 scripts/cache_quality.py --model "$(basename "$TARGET")" --window cache-evidence --label quality-on --expect-cache on --reference cache-evidence/quality-off.json
```

Stop that server before the final control:

```bash
python3 scripts/serve.py --state "$STATE" --no-cache
```

```bash
python3 scripts/cache_benchmark.py --model "$(basename "$TARGET")" --window cache-evidence --label after --expect-cache off
python3 scripts/compare_cache.py cache-evidence/before.json cache-evidence/candidate.json cache-evidence/after.json
```

`cache_benchmark.py` defaults to five sequences, 12,288 filler words and 256 output tokens. It requires zero cold cache hits and positive exact/branch hits in the on arm. `cache_quality.py` runs seven checks three times, requires correct JSON and finish reasons, and compares cached semantics with the uncached reference. Both preserve full synthetic request/response evidence. A file with `complete: false` is a failed or unfinished arm, not a usable result. Create `cache-evidence/STOP` to stop before the next request.

Cached state can include user prompt content. The cache resides at `$STATE/prefix-cache`; keep the state directory private. Stop the server before intentionally clearing your own cache. Starting with `--no-cache` disables reuse without deleting anything. The benchmark's repeated filler is a controlled fixture, not a claim about every real conversation. The shared prefix must reach a usable block boundary; the measured effective block was 8,192 tokens. Cache-hit counters are the deciding evidence.

To inspect the published numbers without a model:

```bash
python3 scripts/compare_cache.py results/cache/baseline-before.json results/cache/candidate.json results/cache/baseline-after.json
```

The launcher policy also passed a [CPU-only integration check](../results/cache/policy-check.json) against the pinned upstream settings code: on → off → on, preserving FP32 precision and resolving `auto` to SSD sidecars only when enabled. To rerun this check with the build environment's pinned oMLX source available on `PYTHONPATH`, execute `python3 scripts/cache_policy_check.py`. It uses a temporary settings directory and does not load a model or access your serving cache.
