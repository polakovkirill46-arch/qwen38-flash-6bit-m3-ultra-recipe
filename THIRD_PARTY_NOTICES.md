# Third-party notices

Recipe code is licensed under Apache-2.0. This does not relicense dependencies or model weights.

- oMLX 0.6.4, copyright its contributors, Apache-2.0. The installer downloads its pinned source archive. `patches/qualified.patch` changes seven upstream Python files; hashes identify each original and changed file. Retain `licenses/Apache-2.0.txt` and upstream source headers.
- MLX 0.32.2, copyright Apple Inc., MIT. The HC sigmoid and expert reduction helpers follow MLX arithmetic/reduction semantics. See `licenses/MLX-MIT.txt`.
- mlx-vlm, copyright Prince Canuma, MIT. The Qwen4 compatibility model code is based on mlx-vlm code vendored by oMLX. See `licenses/mlx-vlm-MIT.txt`.
- mlx-serve, copyright David Dalcu, MIT. oMLX's native QSA kernel credits its K/V staging work. The native build retains upstream source and credit. See `licenses/mlx-serve-MIT.txt`.
- MTPLX and Youssof Altoukhi supply related MTP work and the separately obtained donor artifact. No MTPLX runtime or model tensors are distributed here. If you add MTPLX code, retain its Apache license and NOTICE, including its attribution requirements.

The ANE adapter is adapted from oMLX PR3298 by onthehub97 and uses the existing ANE/CPU backend from PR2853. HC/grouped RMS fusion has prior implementations in MTPLX PR391. These are credited techniques, not inventions claimed by this recipe. See PROVENANCE.md for source links.

The target model card labels itself Apache-2.0, while upstream Qwen and the donor use Qwen Community License 1.0. Obtain the tensors yourself under the applicable terms. No software license in this repository grants additional model rights. No Apple framework binaries or application bundle are distributed.
