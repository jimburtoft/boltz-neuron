# boltz-neuron

Run **Boltz-2** biomolecular structure prediction on **AWS Inferentia2 (inf2)** with
**PyTorch Native**, through the same command line as `boltz predict`.

The package wraps the stock [`boltz`](https://github.com/jwohlwend/boltz) v2.2.1 (unchanged) and
applies a validated configuration automatically when you run `python -m boltz_neuron`: bf16
inference, a compiled diffusion model, sequence-length padding to fast shapes, caches that survive
restarts, and the Inferentia2 fixes described in [NOTES.md](NOTES.md).

Validated on **inf2.xlarge** and **inf2.8xlarge** (one Inferentia2 chip, two NeuronCores) for single
protein chains of 64-1,088 tokens.

## Requirements

- An inf2 instance running the AWS **Deep Learning AMI Neuron (Ubuntu 24.04)**, SDK 2.32
  (`20260818`).
- Access to the **PyTorch Native Beta 5** container in Amazon ECR
  (`421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b`). `setup_inf2.sh`
  installs its Python stack (torch 2.12.1, torch-neuronx 2.12.3, neuronx-cc 2.27.2878, nki 0.6.0)
  into a local venv.

## Quick start (inf2.xlarge)

```bash
# 1. From a machine with AWS credentials, let the instance pull the PyTorch Native container:
TOKEN=$(aws ecr get-login-password --region us-east-1)
ssh <instance> "echo '$TOKEN' | sudo docker login --username AWS \
    --password-stdin 421672808698.dkr.ecr.us-east-1.amazonaws.com"

# 2. On the instance: install (about 20 minutes)
git clone https://github.com/jimburtoft/boltz-neuron.git && cd boltz-neuron
bash scripts/setup_inf2.sh

# 3. Restore the prebuilt cache, so the example runs without compiling
source ~/boltz_neuron_ws/workspace/venv/bin/activate
pip install -q huggingface_hub
hf download jburtoft/boltz-neuron-cache boltz_neuron_cache_v7_ds1-8-25.tgz \
    --repo-type dataset --local-dir ~/
bash scripts/restore_cache.sh ~/boltz_neuron_cache_v7_ds1-8-25.tgz

# 4. Predict (same arguments as `boltz predict`)
export NEURON_PLATFORM_TARGET_OVERRIDE=inf2 NEURON_RT_VISIBLE_CORES=0 NEURON_SCRATCHPAD_PAGE_SIZE=2048
python -m boltz_neuron predict examples/8eil.yaml --out_dir out \
    --recycling_steps 3 --sampling_steps 50 --diffusion_samples 1 --num_workers 0
```

The output is a standard Boltz-2 mmCIF structure plus confidence files in `out/`.

**The prebuilt cache covers the example only** (186 tokens, `--diffusion_samples` 1, 8 or 25).
Other inputs compile on their first run, which needs more host RAM than an inf2.xlarge has; see
[Building a cache](#building-a-cache-for-your-inputs).

## Performance

PyTorch Native Beta 5, bf16, `--recycling_steps 3 --sampling_steps 50`, fk_steering off, single
protein chains without MSA, warm (already compiled). `ds` = `--diffusion_samples`.

### Latency: seconds per prediction, one NeuronCore

| tokens | ds=1 | ds=8 | ds=25 |
|---:|---:|---:|---:|
| 64 | 5.1 | 8.3 | 12.0 |
| 128 | 6.2 | 10.5 | 16.1 |
| 192 | 11.3 | 17.8 | 32.6 |
| 256 | 15.3 | 23.7 | 47.7 |
| 384 | 27.9 | 39.7 | 75.3 |
| 512 | 63.8 | 85.3 | 151.5 |
| 640 | 99.9 | 133.2 | 223.7 * |
| 1,024 | 334.8 | 495.7 | |
| 1,088 | 375.2 | | |

### Throughput: structures per second per Inferentia2 chip (one worker on each of the 2 cores)

| tokens | ds=1 | ds=8 | ds=25 |
|---:|---:|---:|---:|
| 64 | 0.340 | 1.36 | 3.25 |
| 128 | 0.235 | 1.19 | 2.60 |
| 192 | 0.151 | 0.79 | 1.42 |
| 256 | 0.118 | 0.61 | 0.99 |
| 384 | 0.068 | 0.38 | 0.64 |
| 512 | 0.031 | 0.18 | 0.32 |
| 640 | 0.020 | 0.12 | 0.22 * |
| 1,024 | 0.006 | 0.03 | |
| 1,088 | 0.005 | | |

\* with `BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1` (runs the samples in smaller groups to fit).

All measured on an **inf2.xlarge**: warm model time per prediction (median of 3 after 1 warm-up)
in one `predict` process, and for throughput one worker per NeuronCore
(`NEURON_RT_VISIBLE_CORES=0` and `=1`) running at the same time. The 192-token row is the
186-residue example padded to 192. At 1,024 tokens and above two workers fit in device memory but
need most of the xlarge's 15 GB host RAM. One call with `--diffusion_samples 25` is far cheaper than
25 calls with `--diffusion_samples 1`.

### Command-line time per input (inf2.xlarge, 186-token example)

One `boltz_neuron predict` command, including model load and output writing:

| ds | default | `BOLTZ_NEURON_PRECOMPILE=1` |
|---:|---:|---:|
| 1 | 21-25 s | 13-14 s |
| 8 | 28-34 s | 22-25 s |
| 25 | 58-63 s | 52-55 s |

To process many inputs, pass a **directory** to one command: startup is paid once.

## Input size and padding

- **Largest input: 1,088 tokens per chip.** More diffusion samples lower that (at 1,024 tokens, ds=8
  fits and ds=25 does not).
- **Cap the MSA above ~700 tokens.** The loaded MSA uses device memory: at ~890 residues, 8,192 MSA
  sequences fail, 4,096 fail on some runs, and 2,048 runs reliably with unchanged accuracy (Boltz-2
  uses 1,024 MSA rows inside the model by default). Use `--max_msa_seqs 2048`.
- **Padding.** On Inferentia2, token counts that are not a multiple of 32 run up to 1.6x slower (one
  compiled graph does far more data movement). The package pads every input to the next multiple of
  32, and pads 768 and 896 tokens (sizes the compiler cannot handle) to 832 and 928. Pad tokens are
  masked; padded predictions match unpadded ones within the model's seed-to-seed variation, checked
  on real proteins at ~190, ~765 and ~890 residues.

## Accuracy

- bf16, padding and the caches were validated against fp32, unpadded and uncached runs. Caches and
  precompile give bit-identical output.
- On two ~765-residue proteins with MSAs (PDB 3W7T, 5A7M), predictions match the crystal structures
  at 0.54-0.93 Å CA-RMSD (lDDT-CA 0.978-0.986).
- Boltz-2's diffusion is stochastic: two runs that differ only in the seed can differ by several Å.
  Compare structures against that spread, not against zero.

## Caching and startup

| cache | what it skips | default |
|---|---|---|
| NEFF cache (`/tmp/neff_cache`) | compiling each graph (minutes per input shape) | on |
| AOTAutograd cache | re-lowering the diffusion model in each new process | on (`BOLTZ_NEURON_AOT_CACHE`) |
| Dynamo precompile package | re-tracing the diffusion model in each new process | off (`BOLTZ_NEURON_PRECOMPILE=1`) |

Precompile is off by default because the first precompile run of a shape not yet in the package is
slow (minutes). Build packages ahead of time with `build_cache.sh`, and rebuild after code changes.

Caches match only an environment installed by this repo's `scripts/setup_inf2.sh` at the same
commit.

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
| `NEURON_PLATFORM_TARGET_OVERRIDE` | none | **set to `inf2`** |
| `BOLTZ_NEURON_PAD` | `32` | pad the token count to a multiple of this (`0` or `1` disables padding) |
| `BOLTZ_NEURON_PAD_MAX_ADD` | `-1` | pad only if it adds at most this many tokens (`-1` = no limit) |
| `BOLTZ_NEURON_BAD_SHAPES` | `768=832,896=928` | token counts the compiler cannot handle, and their replacement (`""` disables) |
| `BOLTZ_NEURON_DTYPE` | `bf16` | `fp32` disables bf16 |
| `BOLTZ_NEURON_COMPILE_STRUCTURE` | `1` | `0` disables compiling the diffusion model |
| `BOLTZ_NEURON_AOT_CACHE` | `1` | reuse the traced diffusion model across processes |
| `BOLTZ_NEURON_PRECOMPILE` | `0` | `1` saves/loads a dynamo precompile package |
| `BOLTZ_NEURON_NUM_THREADS` | `2` | torch thread count when precompile is on |
| `BOLTZ_NEURON_DROP_DISTO` | `1` | drop a training-only feature before it reaches the device |
| `BOLTZ_NEURON_WRITER_CPU` | `1` | remove padding from outputs on CPU |
| `BOLTZ_NEURON_UID_WRAP` | `1` | keep residue ids exact in bf16 above 256 residues |
| `BOLTZ_NEURON_STEERING` | `off` | `on` enables fk_steering (much slower on Neuron) |

Always pass `--num_workers 0`: it avoids a DataLoader timeout while a shape compiles.

## License

MIT (this wrapper). Boltz-2 is MIT-licensed by its authors.
