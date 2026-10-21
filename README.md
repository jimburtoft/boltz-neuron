# boltz-neuron

Run **Boltz-2** biomolecular structure prediction on **AWS Inferentia2 (inf2)** with
**PyTorch Native**, through the same command line as `boltz predict`.

The package wraps the stock [`boltz`](https://github.com/jwohlwend/boltz) v2.2.1 (unchanged) and
applies a validated configuration automatically when you run `python -m boltz_neuron`: bf16
inference, a compiled diffusion model, sequence-length padding to fast shapes, caches that survive
restarts, and the Inferentia2 fixes described in [NOTES.md](NOTES.md).

Runs on the public **AWS Neuron SDK 2.33** torch-neuronx DLAMI; no containers or private
packages. Validated on **inf2.xlarge** and **inf2.8xlarge** (one Inferentia2 chip, two NeuronCores)
for single protein chains of 64-1,056 tokens. (PyTorch Native is officially supported on Trainium2/3;
Inferentia2 works with the fixes in [NOTES.md](NOTES.md) but is not an AWS-supported target.)

## Requirements

- An inf2 instance (inf2.xlarge to run, inf2.8xlarge or larger to compile new shapes) launched from
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

## Quick start (inf2.xlarge)

```bash
# 1. Install (about 3 minutes)
git clone https://github.com/jimburtoft/boltz-neuron.git && cd boltz-neuron
bash scripts/setup_inf2.sh

# 2. Restore the prebuilt cache, so the example runs without compiling
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
hf download jburtoft/boltz-neuron-cache boltz_neuron_cache_v8_sdk233_ds1-8-25.tgz \
    --repo-type dataset --local-dir ~/
bash scripts/restore_cache.sh ~/boltz_neuron_cache_v8_sdk233_ds1-8-25.tgz

# 3. Predict (same arguments as `boltz predict`)
export NEURON_RT_VISIBLE_CORES=0
python -m boltz_neuron predict examples/8eil.yaml --out_dir out \
    --recycling_steps 3 --sampling_steps 50 --diffusion_samples 1 --num_workers 0
```

Do not set `NEURON_PLATFORM_TARGET_OVERRIDE` (earlier versions of this package needed `inf2`; on SDK
2.33 the package removes it, see NOTES.md).

The output is a standard Boltz-2 mmCIF structure plus confidence files in `out/`.

**The prebuilt cache covers the example only** (186 tokens, `--diffusion_samples` 1, 8 or 25).
Other inputs compile on their first run, which needs more host RAM than an inf2.xlarge has; see
[Building a cache](#building-a-cache-for-your-inputs).

## Performance

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

## Input size and padding

- **Largest input: 1,056 tokens per chip at `--diffusion_samples 1`.** With more samples the limit is
  lower: 960 tokens runs at ds=8, 1,024 tokens fails at ds=8 (device memory), 640 tokens runs at
  ds=25. 1,088 tokens no longer fits (it did with the Beta 5 build).
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
  0.981-0.983), the same as the Beta 5 build (0.54-0.90 Å, 0.982-0.986).
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

Compiling needs about 40 GB of host RAM per graph, so build on an **inf2.8xlarge** and run on an
inf2.xlarge:

```bash
# On an inf2.8xlarge, after setup_inf2.sh: a directory with one input of every size you serve
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
