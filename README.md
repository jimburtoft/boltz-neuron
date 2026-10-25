# boltz-neuron

Run **Boltz-2** biomolecular structure prediction on **AWS Inferentia2 (inf2)** and **Trainium2
(trn2)** with **PyTorch Native**, through the same command line as `boltz predict`.

The package wraps the stock [`boltz`](https://github.com/jwohlwend/boltz) v2.2.1 (unchanged) and
applies a validated configuration automatically when you run `python -m boltz_neuron`: bf16
inference, a compiled diffusion model, sequence-length padding to fast shapes, caches that survive
restarts, and the Inferentia2 fixes described in [NOTES.md](NOTES.md).

Runs on the public **AWS Neuron SDK 2.33** torch-neuronx DLAMI; no containers or private
packages. Validated on **inf2.xlarge / inf2.8xlarge / inf2.24xlarge** (64-1,056 tokens per
Inferentia2 chip) and **trn2.3xlarge** (64-1,408 tokens per Trainium2 logical core) for single protein
chains, protein-ligand complexes (including affinity prediction) and protein-protein complexes. The
same install and the same commands work on both; the package detects the instance. (PyTorch Native is officially
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
bash scripts/setup.sh

# 2. Restore the prebuilt cache, so the example runs without compiling
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
CACHE=boltz_neuron_cache_v8_sdk233_ds1-8-25.tgz          # inf2
# CACHE=boltz_neuron_cache_v9_sdk233_trn2_lnc2_ds1-8-25.tgz   # trn2 (LNC=1: ..._v8_sdk233_trn2_ds1-8-25.tgz)
hf download jburtoft/boltz-neuron-cache $CACHE --repo-type dataset --local-dir ~/
bash scripts/restore_cache.sh ~/$CACHE

# 3. Predict (same arguments as `boltz predict`)
export NEURON_RT_VISIBLE_CORES=0
python -m boltz_neuron predict examples/8eil.yaml --out_dir out \
    --recycling_steps 3 --sampling_steps 50 --diffusion_samples 1 --num_workers 0
```

Do not set `NEURON_PLATFORM_TARGET_OVERRIDE` (earlier versions of this package needed `inf2`; on SDK
2.33 the package removes it, see NOTES.md). On trn2 the package runs at the trn2 default **LNC=2**
(4 logical NeuronCores per device, `NEURON_RT_VISIBLE_CORES=0`..`3`, 24 GB each) with a fix for a
compiler issue at LNC=2 (see [Trainium2](#trainium2-trn2)).

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
and 52 s of that is the prediction itself). Since then the package skips Boltz-2's random weight
initialisation during checkpoint loading (it is overwritten anyway), which removes ~17 s per command
(192-token example: trn2 57 s -> 39 s; fresh inf2.xlarge 82/92 s -> 62/72 s at ds 8/25; identical output). The first command after boot takes about 4 minutes
(the DLAMI's packages load from disk for the first time). To process many inputs, pass a
**directory** to one command: startup is paid once.

### Larger instances

On an **inf2.24xlarge** (6 chips, 12 NeuronCores) with one worker per NeuronCore, throughput scales
linearly: 1.07 structures/s at ds=1 and 9.33 at ds=25 (192 tokens), 0.99-1.06x of six inf2.xlarge
chips; per-worker times match a single worker. Per structure, inf2.xlarge stays the cheapest.

### Inputs with an MSA

The tables above use single-sequence inputs. With an MSA (`--max_msa_seqs 2048`), one NeuronCore,
warm:

| input | tokens (padded) | ds=1 | ds=8 | ds=25 |
|---|---:|---:|---:|---:|
| PDB 3HTB (protein) | 192 | 14.2 s | | 33.8 s |
| PDB 1M17 (protein) | 352 | 30.4 s | | 76.9 s |
| PDB 3W7T | 832 | 204.6 s | | does not fit |
| PDB 6EA2 | 928 | 253.2 s | 333.1 s | does not fit |

At ~200 tokens the MSA adds about 3.6 s per prediction; above ~350 tokens it adds little. Above
~800 tokens with an MSA, ds=25 does not fit on one Inferentia2 core (use ds=8, or trn2).

## Trainium2 (trn2)

Measured on a **trn2.3xlarge** (one Trainium2 device), Neuron SDK 2.33, same settings and method as
the Inferentia2 tables. Default: **LNC=2**, 4 logical NeuronCores of 24 GB each, one worker per core.

### Latency: seconds per prediction, one logical NeuronCore (LNC=2)

| tokens | ds=1 | ds=8 | ds=25 |
|---:|---:|---:|---:|
| 64 | 4.0 | 7.2 | 8.0 |
| 128 | 4.5 | 8.6 | 10.0 |
| 192 | 7.5 | 10.6 | 16.7 |
| 256 | 9.4 | 13.8 | 25.8 |
| 384 | 16.5 | 23.1 | 49.5 |
| 512 | 31.1 | 41.5 | 66.5 ** |
| 640 | 54.6 | 73.8 | 116.1 * |
| 960 | 143.8 | 186.1 | 282.1 |
| 1,024 | 201.8 | 258.1 |  |
| 1,056 | 205.4 | 269.7 |  |

### Throughput: structures per second per Trainium2 device (4 workers, LNC=2)

| tokens | ds=1 | ds=8 | ds=25 |
|---:|---:|---:|---:|
| 64 | 0.984 | 4.14 | 11.61 |
| 128 | 0.889 | 3.53 | 9.14 |
| 192 | 0.535 | 2.89 | 5.81 |
| 256 | 0.427 | 2.26 | 3.85 |
| 384 | 0.242 | 1.37 | 2.00 |
| 512 | 0.129 | 0.764 | 1.49 ** |
| 640 | 0.073 | 0.432 | 0.857 * |
| 960 | 0.028 | 0.172 | 0.348 |
| 1,024 | 0.020 | 0.124 |  |
| 1,056 | 0.019 | 0.118 |  |

\* with `BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1`. \** 512 tokens x 25 samples (and other 448-576-token,
ds >= 16 shapes) does not compile with the LNC=2 accuracy fix (`NCC_ILSA902` / `NCC_ITEN406`); set
`BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1` there (runs the samples in chunks; 66.5 s). Measured with the
shipped default (LNC=2 + the accuracy fix): same as without the fix within 4%, except 128 tokens x 8
samples (8.6 s vs 6.8 s).

### Inputs with an MSA (trn2, LNC=2)

| input | tokens (padded) | ds=1 | ds=8 | ds=25 |
|---|---:|---:|---:|---:|
| PDB 3HTB (protein) | 192 | 7.8 s | | 17.6 s |
| PDB 1M17 (protein) | 352 | 17.6 s | | |
| PDB 3W7T | 832 | 106.3 s | | 211.4 s * |
| PDB 6EA2 | 928 | 141.9 s | 184.0 s | 282.3 s * |

### Larger trn2 instances

Not measured. A trn2.48xlarge has 16 Trainium2 devices; with one worker per logical core expect about
16x the per-device throughput above (inf2.24xlarge, measured, scales linearly with one worker per
core).

### Input size limits (single protein chain, one worker)

| instance | ds=1 | ds=8 | ds=25 |
|---|---|---|---|
| inf2 (one NeuronCore, 16 GB) | **1,056** (1,088 fails) | 960 (1,024 fails) | 640 * (960 not tested) |
| **trn2, LNC=2** (default; one 24 GB logical core) | **1,408** (1,536 fails) | **1,280** (1,408 fails) | **960** (1,024 fails) |
| trn2, LNC=1 (12 GB logical cores; one worker per HBM bank) | 1,280 (higher not tested) | 1,056 (higher not tested) | 960 (higher not tested) |

"Fails" means the device memory runs out (`NRT_RESOURCE`). \* needs `BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1`.
Deep MSAs lower these; use `--max_msa_seqs 2048` above ~700 tokens. Near the limits the runtime
retries allocations, which slows those runs.

### LNC=2 or LNC=1

| tokens | single worker, LNC=2 / LNC=1 (s) | per device, LNC=2 / LNC=1 (structures/s, ds=1) |
|---:|---|---|
| 192 | 7.5 / 10.2 | 0.535 (4 workers) / 0.716 (8) |
| 512 | 31.1 / 50.8 | 0.129 (4) / 0.142 (8) |
| 640 | 54.6 / 94.9 | 0.073 (4) / 0.081 (8) |
| 960 | 143.8 / 255.8 | 0.028 (4) / 0.016 (4) |
| 1,056 | 205.4 / 366.7 | 0.019 (4) / 0.011 (4) |

LNC=2 (default) has the lowest latency, the largest inputs and, from ~700 tokens, the most throughput.
Up to 640 tokens, `NEURON_LOGICAL_NC_CONFIG=1` (8 workers) gives 6-58% more structures per second per
device. Accuracy is the same at both settings (below). Caches are per LNC setting.

### LNC=2 accuracy fix

With SDK 2.33, one compiled graph in Boltz-2's diffusion model returns wrong values at LNC=2 for some
atom counts: when the atom/token one-hot matrix multiply is compiled into the same graph as the
windowed atom attention that follows it (e.g. 5,952-6,400 atoms; not 4,096, 6,144 or 8,192). On
two ~765-residue proteins this put predictions 1.6-2.2 Å from the crystal (lDDT-CA 0.84-0.86). The
same issue is present in the earlier PyTorch Native Beta 5 compiler. The package splits that graph
in two (`BOLTZ_NEURON_ATOM_GRAPH_BREAK=1`, set automatically on trn2 at LNC=2), which restores
accuracy with no measurable speed cost: **0.61-0.92 Å, lDDT-CA 0.979-0.982** on the same proteins
(two seeds each), the same as LNC=1 and Inferentia2. Details in [NOTES.md](NOTES.md).

## Input size and padding

- **Largest input: 1,056 tokens per Inferentia2 chip, 1,408 per Trainium2 logical core (LNC=2)**, at
  `--diffusion_samples 1`; lower with more samples (table in
  [Input size limits](#input-size-limits-single-protein-chain-one-worker)). On inf2, 1,088 tokens no
  longer fits (it did with the Beta 5 build): the SDK 2.33 compiler reserves 52-78% more device scratch
  for the diffusion model (measured on the same chip), which is also why 640-1,024 tokens are 3-5% slower.
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
  0.61-0.92 Å (lDDT-CA 0.979-0.983) on trn2 (LNC=2 with the fix, and LNC=1).
- ~890-residue proteins (PDB 6EA2, 3B34, MSA, padded 889/891 -> 928 tokens, seeds 0 and 1): 0.37-0.47 Å
  CA-RMSD (lDDT-CA 0.993-0.997) on Inferentia2 and 0.47-0.50 Å (0.991-0.996) on trn2, the same as the
  Beta 5 build (0.34-0.44 Å).
- **Ligands and complexes** (MSA, seeds 0 and 1, vs crystal; CPU fp32 = stock Boltz-2 on CPU):

  | target | metric | Inferentia2 | trn2 | CPU fp32 |
  |---|---|---|---|---|
  | 3HTB, T4 lysozyme + 2-propylphenol | ligand RMSD (pocket-aligned) | 0.33-0.35 Å | 0.34 Å | 0.34 Å |
  | | affinity (log10 IC50 / P(binder)) | -0.21, -0.14 / 0.41, 0.40 | -0.22 / 0.44 | -0.31 / 0.45 |
  | 1M17, EGFR kinase + erlotinib | ligand RMSD | 0.79-0.82 Å | 0.92 Å | 0.98 Å |
  | 1BRS, barnase-barstar | interface RMSD | 0.77-0.80 Å | 0.79 Å | 0.77 Å |

  Protein CA-RMSD for 3HTB and 1BRS is 0.25-0.37 Å everywhere. For 1M17 the whole-chain CA-RMSD is
  ~11.8 Å on every platform including CPU fp32: the model places the C-terminal tail differently from
  the crystal (the rest of the chain is 0.1-1.0 Å), so it is a property of Boltz-2, not the port.
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
| `NEURON_LOGICAL_NC_CONFIG` | trn2 default (`2`) | `1` = 8 smaller cores on trn2 (more throughput below ~700 tokens); not used on inf2 |
| `BOLTZ_NEURON_ATOM_GRAPH_BREAK` | `1` on trn2 at LNC=2, else `0` | splits one diffusion-model graph (LNC=2 compiler fix) |
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
