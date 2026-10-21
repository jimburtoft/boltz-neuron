# Notes

What `boltz-neuron` changes to make Boltz-2 run on Inferentia2, and what to watch for.

## Fixes applied automatically

`scripts/setup_inf2.sh` patches the installed `torch_neuronx` (two lines in its `torch.nonzero`
lowering); the `boltz_neuron` package applies the rest when it starts. None of these change Boltz-2's
model code or weights. Status below is for Neuron SDK 2.33 (torch-neuronx 2.14.3) on Inferentia2.

| fix | where | find it by | why |
|---|---|---|---|
| `torch.nonzero` scatter dtype | setup | `NONZERO_DTYPE_SHIM` | the Neuron lowering mixes fp32 and bf16 in one scatter and fails on bf16 models |
| `torch.nonzero` count | setup | `BOOL_SUM_SHIM` | on Inferentia2 the count can come out too small at some lengths |
| int64 mode | package | `TORCH_NEURONX_INT64_MODE` | the default `native` mode returns wrong values for some int64 ops on Inferentia2; the package sets `packed` |
| platform target | package | `NEURON_PLATFORM_TARGET_OVERRIDE` | with `inf2` set, torch-neuronx 2.14 picks an RNG kernel that needs a newer NeuronCore; the package removes it |
| random numbers on CPU | package | `WORKAROUND(gen2-rng)` | on Inferentia2, `torch.randn(device="neuron")` recurses forever; random tensors are drawn on CPU and copied (same values for a given `--seed`) |
| outputs unpadded on CPU | package | `WORKAROUND(nonzero-index)` | on-device boolean indexing can return wrong indices at some lengths |
| tokens padded to a multiple of 32 | package | `BOLTZ_NEURON_PAD` | other token counts make one graph do ~N x 384 extra DMA transfers |
| 768 / 896 tokens padded to 832 / 928 | package | `WORKAROUND(trimul-shape)` | the compiler cannot handle these two sizes |
| residue ids mod 256 | package | `BOLTZ_NEURON_UID_WRAP` | keeps an atom-to-residue mask exact in bf16 above 256 residues |
| Lightning accelerator | package | `NeuronAccelerator` | Lightning has no Neuron device; the package registers one |
| precompile guards | package | `_enable_precompile_guard` | needed for `BOLTZ_NEURON_PRECOMPILE=1` to load correctly (Beta 5 build; precompile does not load on SDK 2.33) |

Search `boltz_neuron/preamble.py` and `scripts/setup_inf2.sh` for the name in the "find it by"
column. Each fix is meant to be removed once the underlying Neuron issue is fixed.

## Things to know

- **Not an AWS-supported target.** PyTorch Native (torch-neuronx 2.14) is supported on Trainium2/3.
  It runs on Inferentia2 with the fixes above; re-check them on each new SDK release.
- **Each input shape compiles once.** A shape is the padded token count, padded atom count and
  `--diffusion_samples`. The first run of a new shape takes minutes and about 40 GB of host RAM,
  so compile on an inf2.8xlarge and copy the cache (`scripts/build_cache.sh`).
- **Pass `--num_workers 0`**, or the DataLoader can time out during a compile.
- **`--max_parallel_samples` in Boltz 2.2.1 is a chunk count, not a cap.** The default (5) runs
  `--diffusion_samples 8` as 4 chunks of 2; the package leaves it alone (no measurable cost).
  `BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1` makes it a cap, which helps large inputs with many samples fit.
- **Large inputs:** up to 1,056 tokens per chip (ds=1); use `--max_msa_seqs 2048` above ~700 tokens.
- **Ligands / complexes and fk_steering are not validated.** The confidence head and the steering
  potentials do boolean indexing on device, which the `nonzero-index` issue can corrupt at some atom
  counts (protein-only inference does not reach it).
