# boltz-neuron

Run **Boltz-2** biomolecular structure prediction on **AWS Inferentia2 (inf2)** and **Trainium2
(trn2)** with **PyTorch Native**, through the same command line as `boltz predict`.

The package wraps the stock [`boltz`](https://github.com/jwohlwend/boltz) v2.2.1 (unchanged) and
applies a validated configuration automatically when you run `python -m boltz_neuron`: bf16
inference, a compiled diffusion model, sequence-length padding to fast shapes, caches that survive
restarts, and the Inferentia2 fixes described in [NOTES.md](NOTES.md).

Runs on the public **AWS Neuron SDK 2.33** torch-neuronx DLAMI; no containers or private
packages. Validated for single protein chains on **inf2.xlarge / inf2.8xlarge** (one Inferentia2
chip, 64-1,056 tokens) and **trn2.3xlarge** (one Trainium2 device, 64-1,280 tokens). The same install
and the same commands work on both; the package detects the instance. (PyTorch Native is officially
supported on Trainium2/3; Inferentia2 works with the fixes in [NOTES.md](NOTES.md) but is not an
AWS-supported target.)

## Requirements

- An inf2 instance (inf2.xlarge to run, inf2.8xlarge or larger to compile new shapes), or a trn2
  instance (trn2.3xlarge runs and compiles), launched from
  **Deep Learning AMI Neuron torch-neuronx 2.14.0 (Ubuntu 24.04) SDK 2.33.0** (or the SDK 2.33
  multi-framework DLAMI). Look up the AMI ID for your region with:

  ```bash
  aws ssm get-parameter --region <region> \
      --name /aws/service/neuron/dlami/torch-neuronx-2.14.0/ubuntu-24.04/latest/image_id \
      --query Parameter.Value --output text
  ```

  The package uses the DLAMI's venv as is (`/opt/aws_neuronx_venv_torch_neuronx_2_14_0`: torch
  2.14.0, torch-neuronx 2.14.3, neuronx-cc 2.28.7631, nki 0.7.0) and adds Boltz-2 to it.
- A root volume of at least 100 GB (300 GB on the compile host).

## Quick start (inf2.xlarge or trn2.3xlarge)

```bash
# 1. Install (about 3 minutes)
git clone https://github.com/jimburtoft/boltz-neuron.git && cd boltz-neuron
bash scripts/setup_inf2.sh

# 2. Restore the prebuilt cache, so the example runs without compiling
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
CACHE=boltz_neuron_cache_v8_sdk233_ds1-8-25.tgz          # inf2
# CACHE=boltz_neuron_cache_v8_sdk233_trn2_ds1-8-25.tgz   # trn2
hf download jburtoft/boltz-neuron-cache $CACHE --repo-type dataset --local-dir ~/
bash scripts/restore_cache.sh ~/$CACHE

# 3. Predict (same arguments as `boltz predict`)
export NEURON_RT_VISIBLE_CORES=0
python -m boltz_neuron predict examples/8eil.yaml --out_dir out \
    --recycling_steps 3 --sampling_steps 50 --diffusion_samples 1 --num_workers 0
```

Do not set `NEURON_PLATFORM_TARGET_OVERRIDE` (earlier versions of this package needed `inf2`; on SDK
2.33 the package removes it, see NOTES.md). On trn2 the package runs at **LNC=1** (8 logical
NeuronCores per device, `NEURON_RT_VISIBLE_CORES=0`..`7`) because the default LNC=2 is less accurate in
bf16 (see [Trainium2](#trainium2-trn2)).

The output is a standard Boltz-2 mmCIF structure plus confidence files in `out/`.

**The prebuilt cache covers the example only** (186 tokens, `--diffusion_samples` 1, 8 or 25).
Other inputs compile on their first run, which needs more host RAM than an inf2.xlarge has; see
[Building a cache](#building-a-cache-for-your-inputs).

## Performance (Inferentia2)

Neuron SDK 2.33 (torch-neuronx 2.14.3), bf16, `--recycling_steps 3 --sampling_steps 50`, fk_steering
off, single protein chains without MSA, warm (already compiled). `ds` = `--diffusion_samples`.

### Latency: seconds per prediction, one NeuronCore

| tokens | ds=1 | ds=8 | ds=25 |
|---:|---:|---:|---:|
| 64 | 4.4 | 7.0 | 11.2 |
| 128 | 5.8 | 9.3 | 15.5 |
| 192 | 10.6 | 16.3 | 30.3 |
| 256 | 15.0 | 22.3 | 44.3 |
| 384 | 28.2 | 39.6 | 75.7 |
| 512 | 62.8 | 83.4 | 144.7 |
| 640 | 103.0 | 139.6 | 230.5 * |
| 960 | 261.7 | 346.6 | |
| 1,024 | 347.8 | | |
| 1,056 | 370.4 | | |

### Throughput: structures per second per Inferentia2 chip (one worker on each of the 2 cores)

| tokens | ds=1 | ds=8 | ds=25 |
|---:|---:|---:|---:|
| 64 | 0.344 | 1.65 | 3.54 |
| 128 | 0.276 | 1.34 | 2.74 |
| 192 | 0.168 | 0.89 | 1.57 |
| 256 | 0.127 | 0.68 | 1.10 |
| 384 | 0.069 | 0.39 | 0.65 |
| 512 | 0.032 | 0.19 | 0.34 |
| 640 | 0.019 | 0.11 | 0.21 * |
| 960 | 0.008 | 0.046 | |
| 1,024 | 0.006 | | |
| 1,056 | 0.005 | | |

\* with `BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1` (runs the samples in smaller groups to fit).

All measured on an **inf2.xlarge**: warm model time per prediction (median of 3 after 1 warm-up)
in one `predict` process, and for throughput one worker per NeuronCore
(`NEURON_RT_VISIBLE_CORES=0` and `=1`) running at the same time. The 192-token row is the
186-residue example padded to 192. One call with `--diffusion_samples 25` is far cheaper than 25
calls with `--diffusion_samples 1`.

Compared with the earlier PyTorch Native Beta 5 build of this package (same instance type, same
method), SDK 2.33 is 2-16% faster at 64-256 tokens, within 5% at 384-512 tokens, and 3-5% slower
at 640-1,024 tokens.

### Command-line time per input (inf2.xlarge, 186-token example)

One `boltz_neuron predict` command on a fresh inf2.xlarge after restoring the prebuilt cache,
including model load and output writing: about 72 s for ds=1 and ds=8 and 93 s for ds=25 (24, 32
and 52 s of that is the prediction itself). The first command after boot takes about 4 minutes
(the DLAMI's packages load from disk for the first time). To process many inputs, pass a
**directory** to one command: startup is paid once.

## Trainium2 (trn2)

Measured on a **trn2.3xlarge** (one Trainium2 device), Neuron SDK 2.33, same settings and method as
the Inferentia2 tables. The package runs at **LNC=1**: 8 logical NeuronCores, two per HBM bank.

### Latency: seconds per prediction, one logical NeuronCore (LNC=1)

| tokens | ds=1 | ds=8 | ds=25 |
|---:|---:|---:|---:|
| 64 | 4.3 | 6.8 | 10.5 |
| 128 | 5.4 | 8.6 | 13.8 |
| 192 | 10.2 | 15.5 | 26.6 |
| 256 | 13.3 | 20.0 | 43.7 |
| 384 | 25.9 | 36.5 | 67.4 |
| 512 | 50.8 | 68.6 | 118.8 |
| 640 | 94.9 | 129.9 | 210.0 * |
| 960 | 255.8 | 336.1 | 521.3 |
| 1,024 | 318.8 | 406.9 | |
| 1,056 | 366.7 | 479.1 | |

### Throughput: structures per second per Trainium2 device

| tokens | ds=1 | ds=8 | ds=25 | workers |
|---:|---:|---:|---:|---|
| 64 | 1.49 | 6.01 | 12.9 | 8 |
| 128 | 1.14 | 4.98 | 10.2 | 8 |
| 192 | 0.716 | 3.35 | 6.33 | 8 |
| 256 | 0.543 | 2.77 | 4.04 | 8 |
| 384 | 0.294 | 1.63 | 2.75 | 8 |
| 512 | 0.142 | 0.840 | 1.52 | 8 |
| 640 | 0.081 | 0.473 | 0.895 * | 8 |
| 960 | 0.016 | 0.095 | 0.193 | 4 |
| 1,024 | 0.013 | 0.079 | | 4 |
| 1,056 | 0.011 | 0.067 | | 4 |

\* with `BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1`. Up to 640 tokens, one worker runs on each of the 8 logical
cores (`NEURON_RT_VISIBLE_CORES=0`..`7`). From 960 tokens two workers no longer fit in one 24 GB HBM
bank, so run one worker per bank (cores 0, 2, 4, 6).

At 960 tokens and ds=25 the device is close to full (the runtime retries allocations, which makes that
cell slow).

### Input size limits (single protein chain, one worker)

| instance | ds=1 | ds=8 | ds=25 |
|---|---|---|---|
| inf2 (one NeuronCore, 16 GB) | **1,056** (1,088 fails) | 960 (1,024 fails) | 640 * (960 not tested) |
| trn2, LNC=1 (one logical core, one per 24 GB HBM bank) | **1,280** (higher not tested) | 1,056 (higher not tested) | 960 (higher not tested) |
| trn2, LNC=2 (one logical core = one 24 GB bank) | **1,280** (1,536 fails) | 1,056 (higher not tested) | 960 (higher not tested) |

"Fails" means the device memory runs out (`NRT_RESOURCE`). \* needs `BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1`.
Deep MSAs lower these; use `--max_msa_seqs 2048` above ~700 tokens.

### LNC=1 vs LNC=2 speed on trn2 (bf16)

| tokens | single worker, LNC=1 / LNC=2 (s) | per device, LNC=1 / LNC=2 (structures/s, ds=1) |
|---:|---|---|
| 192 | 10.2 / 7.5 | 0.716 (8 workers) / 0.523 (4) |
| 512 | 50.8 / 31.1 | 0.142 (8) / 0.129 (4) |
| 640 | 94.9 / 54.9 | 0.081 (8) / 0.073 (4) |
| 960 | 255.8 / 145.2 | 0.016 (4) / 0.028 (4) |
| 1,024 | 318.8 / 195.6 | 0.013 (4) / 0.020 (4) |
| 1,056 | 366.7 / 207.0 | 0.011 (4) / 0.019 (4) |

Up to 640 tokens LNC=1 gives more per device (8 workers); from 960 tokens only 4 LNC=1 workers fit, and
LNC=2 is 1.6-1.8x faster per device. **LNC=2 is not used by default because of the accuracy issue
below**; the LNC=2 numbers are for reference.

### Why LNC=1 on trn2

trn2 defaults to LNC=2 (4 logical cores). With SDK 2.33 and bf16, Boltz-2 compiled at LNC=2 is less
accurate: on two ~765-residue proteins (3W7T, 5A7M; MSA, 832 tokens) CA-RMSD to the crystal is
1.6-2.2 Å and lDDT-CA 0.84-0.86, against 0.61-0.89 Å and 0.980-0.983 at LNC=1 (and on Inferentia2).
fp32 at LNC=2 is accurate, so the difference is in the LNC=2 bf16 build. The package therefore sets
`NEURON_LOGICAL_NC_CONFIG=1` on trn2 unless you set it yourself (it warns for LNC=2 with bf16). LNC=1
also gives more throughput per device at 64-640 tokens (6-58% more than LNC=2), but less from 960 tokens
(see the table above). The LNC=2 accuracy issue is under investigation.

## Input size and padding

- **Largest input: 1,056 tokens per Inferentia2 chip, 1,280 per Trainium2 logical core**, at
  `--diffusion_samples 1`; lower with more samples (table in
  [Input size limits](#input-size-limits-single-protein-chain-one-worker)). On inf2, 1,088 tokens no
  longer fits (it did with the Beta 5 build).
- **Cap the MSA above ~700 tokens.** The loaded MSA uses device memory. Use `--max_msa_seqs 2048`
  (Boltz-2 uses 1,024 MSA rows inside the model by default).
- **Padding.** On Inferentia2, token counts that are not a multiple of 32 run up to 1.6x slower (one
  compiled graph does far more data movement). The package pads every input to the next multiple of
  32, and pads 768 and 896 tokens (sizes the compiler cannot handle) to 832 and 928. Pad tokens are
  masked; padded predictions match unpadded ones within the model's seed-to-seed variation, checked
  on real proteins at ~190, ~765 and ~890 residues.

## Accuracy

- bf16, padding and the caches were validated against fp32, unpadded and uncached runs. Caches
  give bit-identical output.
- On SDK 2.33, two ~765-residue proteins with MSAs (PDB 3W7T, 5A7M; `--max_msa_seqs 2048`, padded
  to 832 tokens, seeds 0 and 1) match the crystal structures at 0.62-0.92 Å CA-RMSD (lDDT-CA
  0.981-0.983) on Inferentia2, the same as the Beta 5 build (0.54-0.90 Å, 0.982-0.986), and at
  0.61-0.89 Å (lDDT-CA 0.980-0.983) on trn2 at LNC=1.
- Boltz-2's diffusion is stochastic: two runs that differ only in the seed can differ by several Å.
  Compare structures against that spread, not against zero.

## Caching and startup

| cache | what it skips | default |
|---|---|---|
| NEFF cache (`/tmp/neff_cache`) | compiling each graph (minutes per input shape) | on |
| AOTAutograd cache | re-lowering the diffusion model in each new process | on (`BOLTZ_NEURON_AOT_CACHE`) |

Caches match only an environment installed by this repo's `scripts/setup_inf2.sh` on the same SDK
release (the cache key includes a hash of the installed `torch_neuronx` sources).

`BOLTZ_NEURON_PRECOMPILE=1` (a dynamo precompile package that also skips re-tracing) worked with the
Beta 5 build but does not load on torch 2.14, so it is not recommended on SDK 2.33.

### Building a cache for your inputs

Compiling needs about 40 GB of host RAM per graph, so on Inferentia2 build on an **inf2.8xlarge**
and run on an inf2.xlarge. A trn2.3xlarge (124 GB) can compile for itself. Caches are specific to
the chip (Inferentia2 vs Trainium2) and, on trn2, the LNC setting:

```bash
# On an inf2.8xlarge (or the trn2 instance itself), after setup_inf2.sh: one input of every size you serve
bash scripts/build_cache.sh my_inputs/ 1,8,25     # -> ~/boltz_neuron_cache.tgz
scp ~/boltz_neuron_cache.tgz <xlarge>:~/

# On the inf2.xlarge, after setup_inf2.sh
bash scripts/restore_cache.sh ~/boltz_neuron_cache.tgz
```

A compiled shape is fixed by the padded token count, the padded atom count and `--diffusion_samples`,
so build all the shapes you need in one run.

## Configuration

Environment variables; the defaults are the validated configuration.

| variable | default | meaning |
|-----|---------|---------|
| `NEURON_PLATFORM_TARGET_OVERRIDE` | none | **leave unset** on SDK 2.33 (the package removes `inf2`) |
| `NEURON_LOGICAL_NC_CONFIG` | `1` on trn2 (set by the package) | `2` is less accurate with bf16 on trn2; not used on inf2 |
| `TORCH_NEURONX_INT64_MODE` | `packed` (set by the package) | `native` gives wrong int64 results on Inferentia2 |
| `BOLTZ_NEURON_PAD` | `32` | pad the token count to a multiple of this (`0` or `1` disables padding) |
| `BOLTZ_NEURON_PAD_MAX_ADD` | `-1` | pad only if it adds at most this many tokens (`-1` = no limit) |
| `BOLTZ_NEURON_BAD_SHAPES` | `768=832,896=928` | token counts the compiler cannot handle, and their replacement (`""` disables) |
| `BOLTZ_NEURON_DTYPE` | `bf16` | `fp32` disables bf16 |
| `BOLTZ_NEURON_COMPILE_STRUCTURE` | `1` | `0` disables compiling the diffusion model |
| `BOLTZ_NEURON_AOT_CACHE` | `1` | reuse the traced diffusion model across processes |
| `BOLTZ_NEURON_PRECOMPILE` | `0` | `1` saves/loads a dynamo precompile package (does not load on SDK 2.33) |
| `BOLTZ_NEURON_NUM_THREADS` | `2` | torch thread count when precompile is on |
| `BOLTZ_NEURON_DROP_DISTO` | `1` | drop a training-only feature before it reaches the device |
| `BOLTZ_NEURON_WRITER_CPU` | `1` | remove padding from outputs on CPU |
| `BOLTZ_NEURON_UID_WRAP` | `1` | keep residue ids exact in bf16 above 256 residues |
| `BOLTZ_NEURON_STEERING` | `off` | `on` enables fk_steering (much slower on Neuron) |

Always pass `--num_workers 0`: it avoids a DataLoader timeout while a shape compiles.

## License

MIT (this wrapper). Boltz-2 is MIT-licensed by its authors.
