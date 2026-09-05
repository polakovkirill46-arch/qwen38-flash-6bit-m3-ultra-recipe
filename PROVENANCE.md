# Provenance and originality

A source and X review on September 4, 2026 found no evidence that this complete frozen recipe was a verbatim replica of another recipe. This is a bounded review, not proof of worldwide novelty.

The contribution is an independently developed integration and measured 6-bit M3 Ultra profile. The underlying methods already existed. The full measured gain cannot be assigned to a single patch; there is no complete component ablation.

| Component | Prior work | This recipe |
|---|---|---|
| QSA, scalar projections, PLE framework, Lightning MTP | [oMLX PR3244](https://github.com/jundot/omlx/pull/3244), jonathan308, and oMLX 0.6.4 | Pins the runtime and chooses dispatch/chunk settings for this checkpoint |
| Qwen4 ANE GDN offload | [oMLX PR3298](https://github.com/jundot/omlx/pull/3298), onthehub97 | Adapts serving GDNs, materializes working copies before worker handoff and integrates dispatch |
| CPU/ANE/GPU sharing | [oMLX PR2853](https://github.com/jundot/omlx/pull/2853), onthehub97 | Measures CPU fraction 0.125, 8 threads and padded sequence 3264 |
| Grouped RMS and HC fusion | [MTPLX PR391](https://github.com/youssofal/MTPLX/pull/391), davidtai | Separately written prefill-focused residual-gamma FP32 kernel and HC mix implementation |
| Top-10 expert combination | MLX reduction behavior and existing MoE implementations | Specializes scatter/product/reduction while matching tested MLX BF16 accumulation order |
| Draft compatibility | [MTPLX donor](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed) and oMLX residual-gamma model representation | Adapts 7 separate draft norms and pins the historical donor hash |

[mlx-serve](https://github.com/ddalcu/mlx-serve) and [Weschera's Mac recipe](https://github.com/Weschera/qwen38-flash-next-omlx-mac) are related work. My earlier [MTPLX PR423](https://github.com/youssofal/MTPLX/pull/423) ports oMLX QSA and is not independent corroboration of this result.

Grok CLI searched X for related recipes and kernel work. Source files and primary repository records, not social engagement, determine attribution. Published results on other quantizations or machines are not used to validate this recipe.

The grouped RMS algorithm, ANE offload and MTP are not new. The helpers use experimental private ANE APIs through oMLX. The overall arithmetic is not lossless: ANE working copies and normalization reduction order can change outputs. No universal speed or quality claim is made.

Target provenance was checked against local download metadata and the Hugging Face API: `orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX`, revision `b44cd3338291ed8f7d0357198de9e14483023f7d`, subdirectory `6-bit`. The donor is pinned to historical revision `29ba90f82124961d0d902a9ea9bbb1034972af2f`, file `mtp.safetensors`. Its Hugging Face LFS SHA256 matches the input enforced in `prepare_model.py`: `3498cbee477938de1dd3a42242387bb8fa6ad921053af010543473ab020f717c`. Local download metadata and the public repository tree independently identify these bytes.

## Production MTP update

Deferred PLE lookup follows an existing idea in [mlx-serve PR350](https://github.com/ddalcu/mlx-serve/pull/350), David Dalcu. This implementation uses a separately written, read-only mmap row gather scheduled as an MLX CPU primitive; its scheduling pattern is adapted from Apple's MIT-licensed MLX extension example. The source retains those credits. Compiled verification HC has related prior work in [MTPLX PR391](https://github.com/youssofal/MTPLX/pull/391). Fixed MTP depth is a serving configuration choice. We claim the tested integration and measured profile, not invention of these techniques.

The additional14.7–15.2% production generation result uses a working MTP baseline and five matched requests per workload against before/after controls. It adds no prefill improvement claim and no broad quality-equivalence claim. Stored target weights are unchanged; compiled HC can alter outputs.
