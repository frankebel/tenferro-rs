# GPU: `vdot_read` is one cuBLAS conjugate dot (#2624)

## Decision

`CudaEngine::vdot_read` no longer composes `conj_read` -> `mul_read` ->
`reduce_axes`. It reads both operands' compact spans and issues one BLAS-1 call
on the engine's stream: `cublasSdot`/`cublasDdot` for a real dtype and
`cublasCdotc`/`cublasZdotc` for a complex one, which conjugates the left operand
by definition, so no conjugation copy exists to skip or to keep. This is the
contract #1733 established for the CubeCL engine and #2208 dropped with it,
restored on the engine that remains.

What that removes from a standalone call on two compact f64 vectors: a full
conjugation copy that is the identity on a real dtype, a full product tensor, a
separate reduction (and its partial-vector split on long shapes), and one
completion wait per removed launch.

Rejected alternatives:

- **Keep the composition and only skip the real-dtype conjugation copy.** The
  product tensor and the reduction stay, so two of the three costs remain.
- **A hand-written fused kernel that loads both operands and writes one scalar.**
  It competes with cuBLAS on the one shape where cuBLAS is already bandwidth
  bound, and it would need its own NVRTC source, artifact-cache key and a
  reduction tree. The vendor call is the same number of launches with no new
  kernel to maintain.
- **Skipping the destination's zero fill** (`alloc_uninit` + `assume_init`, the
  pattern `reduce_axes` uses for #2406) would remove one `cuMemsetD8Async` from
  the call, but the raw scope can only register an initialised `Buffer`
  (`RawResource::Buffer`); passing an uninitialised one needs a
  `RawResource::BufferUninit` in `t4a-cuda-runtime`, so the fill stays. Revisit
  if the engine's per-call overhead is attacked in general.
- **`norm_squared_read` and `axpby_read_into_accum`** still compose; they were
  not part of this request and their compositions are correct.

## Measurements

Host `primerose`: NVIDIA A100 80GB PCIe, driver 580.178.04, CUDA toolkit 12.6,
`nvidia-smi` reporting an idle device between runs (0 %), `torch 2.12.0+cu130`
available in the benchmark venv. Rust `1.100.0`, release profile.

The workload is `gpu/reuse_contraction/vdot_read_f64_n262144` (and its `c64`
sibling): two compact rank-1 operands of 262144 elements, one `vdot_read` per
iteration, steady state, host API plus one device synchronisation per call.

The numbers below come from a **diagnostic** (`vdot-probe`, kept outside the
repository) that issues the same call with the same sampling protocol as the
campaign's `benchmark_gpu_public_api`: one untimed priming call, 3 warmups, a
batch grown until it lasts 2 ms, then 15 samples of `batch_ms / ops_per_sample`,
outputs retained until the timer stops, median reported. Arm A is `aebe8cb3`
(the merge base) and arm B is `6081fea5`; both arms are the same binary source
built against each revision, run back to back in one window.

| case | A `aebe8cb3` (composition) | B `6081fea5` (one cuBLAS dot) | speedup |
| --- | --- | --- | --- |
| `vdot_read_f64_n262144` | 0.0629 / 0.0655 / 0.0661 / 0.0651 ms | 0.0214 / 0.0211 / 0.0208 / 0.0215 ms | 3.06x |
| `vdot_read_c64_n262144` | 0.0647 / 0.0662 / 0.0685 / 0.0688 ms | 0.0223 / 0.0220 / 0.0217 / 0.0231 ms | 3.03x |

Four interleaved A/B repetitions each; the table lists them in run order, and
the arm medians are 0.0652 ms -> 0.0213 ms (f64) and 0.0673 ms -> 0.0222 ms
(c64). Against `torch.vdot`'s 0.01017 ms on the same case, the fused path is
2.1x behind instead of 24.8x.

Two caveats the reader must have:

- The issue's baseline of 0.2518 ms is from pin `9a850b06`. On current `main`
  the composition already costs 0.0652 ms, 3.9x faster than that baseline, so
  most of the 24.8x gap was closed by earlier reduction work (#2406, #2474) and
  this change closes the remainder of the *launch* gap, not of a 0.25 ms cost.
  The ratio this change owns is 0.0652 -> 0.0213 ms.
- The machine was not quiet: two other agents were building and running GPU
  tests throughout, and `/proc/loadavg` sat around 26-28 (64 cores) for the
  measurement window. No benchmark process was running and the device was idle
  at 0-16 % at the start and end of every pair, but the absolute values include
  that host contention; both arms share it, which is why the pairs are
  interleaved rather than compared across sessions. The one outlier sample
  (arm A, first run: IQR upper 1.91 ms) is contention, not the operation.

## Verification

- `cargo test -p tenferro-gpu --features cuda --test blas1_cuda -- --ignored
  --test-threads=1` — 15 passed, 0 failed (1.46 s). Includes the real F64/F32
  transposed, offset and mixed-layout operands against the CPU oracle, rank-0
  operands, and the completion-wait count (one for compact operands, three when
  both operands are strided).
- `cargo test -p tenferro-gpu --features cuda --test seam_vendor -- --ignored` —
  5 passed, 0 failed (0.73 s).
- `cargo test -p tenferro-gpu --features cuda --lib vendor_narrowing` — 5
  passed; the new case refuses a dot longer than `i32::MAX` elements by name
  and value.
- `cargo clippy -p tenferro-gpu --features cuda --all-targets -- -D warnings
  -D clippy::missing_errors_doc -D clippy::missing_panics_doc` — clean;
  `cargo fmt --all --check` — clean.
- The CUDA environment for all of the above:
  `CUDA_PATH=/usr/local/cuda-12.6`,
  `LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12`,
  `TENFERRO_REQUIRE_GPU=1`.

## Review record

- Plan review, before implementing: `ollama-cloud/glm-5.3-flash:high`,
  **VERDICT: CHANGES** — adopted: a typed `Unavailable` when cuBLAS cannot be
  loaded (cudarc's `culib()` panics otherwise), the `i32`-length narrowing with
  a test and a CHANGELOG note, and the stale anchors. Declined: dropping
  `DotRequest` (the enum is what keeps the four entry points one call site).
- Diff review, after implementing: `openai/gpt-6.1-sol`, **VERDICT: CHANGES** —
  adopted: `Cublas::dot` is `pub(crate)` (it trusts its spans), a handle whose
  pointer mode cannot be proven is recreated instead of reused by the GEMM
  entries, the design note no longer claims `tenferro-gpu` reads
  `TENFERRO_CUBLAS_PATH` (it loads through cudarc's soname search), and the
  wait-count test is described as a completion-wait regression test rather than
  a launch count. No finding was left unfixed.
