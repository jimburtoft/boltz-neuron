# Notes

What `boltz-neuron` changes to make Boltz-2 run on Inferentia2, and what to watch for.

## Fixes applied automatically

`scripts/setup_inf2.sh` patches the installed `torch_neuronx`; the `boltz_neuron` package applies
the rest when it starts. None of these change Boltz-2's model code or weights.

| fix | where | find it by | why |
|---|---|---|---|
| `torch.nonzero` scatter dtype | setup | `NONZERO_DTYPE_SHIM` | the Neuron lowering mixes fp32 and bf16 in one scatter and fails on bf16 models |
| `torch.nonzero` count | setup | `BOOL_SUM_SHIM` | on Inferentia2 the count can come out too small at some lengths |
| gen2 RNG | setup + package | `GEN2_SHIM` | device RNG seeding requires a newer NeuronCore; not needed for inference |
| outputs unpadded on CPU | package | `WORKAROUND(nonzero-index)` | on-device boolean indexing can return wrong indices at some lengths |
| 768 / 896 tokens padded to 832 / 928 | package | `WORKAROUND(trimul-shape)` | the compiler cannot handle these two sizes |
| residue ids mod 256 | package | `BOLTZ_NEURON_UID_WRAP` | keeps an atom-to-residue mask exact in bf16 above 256 residues |
| Lightning accelerator | package | `NeuronAccelerator` | Lightning has no Neuron device; the package registers one |
| precompile guards | package | `_enable_precompile_guard` | needed for `BOLTZ_NEURON_PRECOMPILE=1` to load correctly |

Search `boltz_neuron/preamble.py` and `scripts/setup_inf2.sh` for the name in the "find it by"
column. Each fix is meant to be removed once the underlying Neuron issue is fixed.

## Things to know

- **Each input shape compiles once.** A shape is the padded token count, padded atom count and
  `--diffusion_samples`. The first run of a new shape takes minutes and about 40 GB of host RAM,
  so compile on an inf2.8xlarge and copy the cache (`scripts/build_cache.sh`).
- **Pass `--num_workers 0`**, or the DataLoader can time out during a compile.
- **`--max_parallel_samples` in Boltz 2.2.1 is a chunk count, not a cap.** The default (5) runs
  `--diffusion_samples 8` as 4 chunks of 2; the package leaves it alone (no measurable cost).
  `BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1` makes it a cap, which helps large inputs with many samples fit.
- **Large inputs:** up to 1,088 tokens per chip; cap `--max_msa_seqs` (4096) above ~700 tokens with
  deep MSAs.
