# Qwen3.8 Flash Next 6-bit recipe for M3 Ultra

I wanted more speed from the 6-bit model on my Mac Studio without changing its stored weights. This repo contains the recipe, patches and measurements that worked on my machine.

It builds on [oMLX](https://github.com/jundot/omlx), [MLX](https://github.com/ml-explore/mlx) and credited community work. The contribution is a tested integration and reproducible benchmark. MTP, ANE offload and normalization fusion already existed. See [provenance](PROVENANCE.md).

## Measured results

Mac Studio, M3 Ultra, **60 GPU cores, 256 GiB RAM**. oMLX 0.6.4 with MLX 0.32.2 and rebuilt native extensions. Original target: `orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX`, `6-bit` directory.

| Test | Original tok/s | Recipe tok/s | Gain |
|---|---:|---:|---:|
| Cold prefill, about 3.3K input tokens | 819.49 | 939.25 | 14.6% |
| Cold prefill, about 12.5K input tokens | 849.78 | 1006.91 | 18.5% |
| Code generation | 30.90 | 62.21 | 101.3% |
| Prose generation | 30.87 | 48.15 | 56.0% |
| Counting generation | 30.92 | 70.68 | 128.6% |

Prefill uses 5 measured requests per context after warmup. Ordinary generation uses 3 per case after warmup. Each speed request generates 256 tokens. Prompt cache hits must be zero. The candidate was confirmed in a separate process/window. [Raw baseline](results/baseline.json) and [candidate](results/candidate.json) contain synthetic response text and timing evidence.

A separate clean installation of this public package reproduced 943.66 and 1006.42 tok/s prefill, with all 15 basic correctness checks passing. The final launcher also passed numerical checks and two consecutive server starts. See [clean-install evidence](results/clean-install.json); it records the launcher-only restart fix separately from the full speed run.

The generation baseline had MTP enabled in settings but no usable draft head. Much of the generation gain comes from repairing that setup. This is **not a 2x claim against an already working MTP setup**. Short-input prefill, about 570 tokens, stayed essentially flat: 520.27 to 516.61 tok/s. Results on other models, Macs and context sizes will differ.

## What you need

- Apple Silicon macOS. This profile was measured on the 60-core M3 Ultra with 256 GiB memory. Other machines are unqualified.
- The official [oMLX 0.6.4 app](https://github.com/jundot/omlx/releases/tag/v0.6.4), extracted locally. The installer uses its Python dependencies without editing the app.
- Native arm64 Python 3.11 **with development headers**, `uv`, `cmake`, and Xcode's Metal compiler. Check `xcrun --find metal`. The app's embedded Python alone does not contain the required headers.
- The exact target checkpoint and donor `mtp.safetensors`, obtained under their model terms. The target is about 200 GB; allow additional storage for the build and separate draft. Do not run this model beside another large GPU model.

Target: [orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX/tree/b44cd3338291ed8f7d0357198de9e14483023f7d), revision `b44cd3338291ed8f7d0357198de9e14483023f7d`, subdirectory `6-bit`. Some file requests require Hugging Face authentication/access. The package accepts an existing local target directory.

Draft: [Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed), revision `29ba90f82124961d0d902a9ea9bbb1034972af2f`, file `mtp.safetensors`. The historical input is pinned by SHA256 `3498cbee477938de1dd3a42242387bb8fa6ad921053af010543473ab020f717c`. A moving `main` download is not proof of the right input. Setup refuses a different file rather than silently changing the recipe.

No target or draft tensors are distributed here. Read the model terms. The donor uses Qwen Community License 1.0; this repository's software license does not relicense model weights.

## Build and run

Clone the repo. Set these paths for your machine. `STATE` must be a new directory outside the repo, app and model inputs.

```bash
git clone https://github.com/humanrouter/qwen38-flash-6bit-m3-ultra-recipe.git
cd qwen38-flash-6bit-m3-ultra-recipe

APP="/Applications/oMLX.app"
PYTHON311="$(uv python find 3.11)"
STATE="$HOME/qwen38-flash-recipe-runtime"
TARGET="$HOME/models/qwen38/6-bit"
DRAFT="$HOME/models/qwen38-draft/mtp.safetensors"

# Skip these downloads if you already have the exact files. Requires the hf CLI.
# Run hf auth login first if the model requires access.
hf download orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX \
  --revision b44cd3338291ed8f7d0357198de9e14483023f7d \
  --include "6-bit/*" --local-dir "$HOME/models/qwen38"
hf download Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed mtp.safetensors \
  --revision 29ba90f82124961d0d902a9ea9bbb1034972af2f \
  --local-dir "$HOME/models/qwen38-draft"

python3 scripts/build.py --app "$APP" --python "$PYTHON311" --state "$STATE"
python3 scripts/prepare_model.py --state "$STATE" --target "$TARGET" --draft "$DRAFT"
```

If Python 3.11 is not installed, install it first with `uv python install 3.11`. Install the other build prerequisites before running setup. The build downloads a SHA256-checked oMLX source archive, applies a checked patch and rebuilds 3 native extensions against MLX 0.32.2. It does not reuse incompatible MLX 0.32.0 binaries.

Stop your other inference server before these GPU checks. Keep it stopped while the recipe server runs.

```bash
python3 scripts/numerical_checks.py --state "$STATE"
python3 scripts/serve.py --state "$STATE" --port 8024
```

The server binds to `127.0.0.1` by default. The served model ID is your target directory's basename, for example `6-bit`. Check `/v1/models`. The scripts do not modify launchd, kill your services, or expose a server publicly.

## Compare on your own machine

Run a baseline against your original server first with other traffic stopped. Then stop that server, start the recipe and run the candidate. Do not load both models at once. Keep model, prompts, sampling, token count and client machine identical.

```bash
# Original server, before stopping it:
python3 scripts/benchmark.py --base-url http://127.0.0.1:8002/v1 \
  --model YOUR_ORIGINAL_MODEL_ID --out baseline.json

# Recipe server, after the original server has stopped:
python3 scripts/benchmark.py --base-url http://127.0.0.1:8024/v1 \
  --model "$(basename "$TARGET")" --out candidate.json

python3 scripts/compare.py baseline.json candidate.json
```

For an authenticated endpoint, set `OPENAI_API_KEY` or name a different environment variable with `--api-key-env`. The benchmark does not save the key. Use a new output filename for each run. Failed, cached or early-stopping requests do not count as valid speed results. Output marked `running` is incomplete.

Prefill here means prompt tokens divided by time to first streamed text, including API overhead. Generation means remaining tokens divided by time after the first text chunk. Repeated-word generation is supplemental; the table uses separate code, prose and counting prompts.

## What changed and what did not

The recipe adds a separate MTP draft with 7 corrected residual-gamma tensors, a measured 8192-token prefill chunk, PLE lookup changes, Qwen4 projection dispatch, dual-ANE GDN/CPU sharing, and specialized HC normalization/mixing and expert reduction. [The exact profile](runtime/qualified-env.json) and [patch](patches/qualified.patch) are included.

Stored target tensors remain unchanged through a symlinked model view. Runtime arithmetic is not bit-identical overall: the ANE path uses approximate INT8 working copies and FP16 projection metadata. Grouped normalization changes FP32 reduction order slightly. All 15 basic correctness checks passed on the measured system, but that is not broad model-quality or agent qualification. The recipe uses experimental private ANE APIs through oMLX.

The measured profile is single-request, cache-off, thinking-off, with a 160 GB memory ceiling and context cap of 393216. Long context up to that cap, concurrent traffic and cached-turn performance are not qualified here. Keep your existing service configuration for rollback. To roll back a local trial, stop this foreground server and restart your original server.

Some dormant experiment branches remain in the patch to preserve the tested source. They are unsupported and disabled. `serve.py` clears inherited tuning variables and loads only the qualified profile. Do not enable extra flags and treat the result as the measured recipe.

## Credits and license

Recipe code is Apache-2.0. Upstream components retain their own terms. See [third-party notices](THIRD_PARTY_NOTICES.md), [provenance](PROVENANCE.md) and the files in `licenses/`.

Run your own baseline and candidate. If you share results, include your GPU core count, memory, exact model, context sizes, sample counts and cache status.
