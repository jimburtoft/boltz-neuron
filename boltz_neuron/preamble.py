"""
boltz_neuron.preamble -- apply the validated Inferentia2 configuration, then run Boltz-2.

Import this module (or run `python -m boltz_neuron`) BEFORE boltz builds its model. It:

  1. Patches torch.load for the Boltz checkpoint (weights_only=False + omegaconf allowlist).
  2. Applies the bf16 patch set (patches.apply_patches / apply_load_patches) so Boltz runs in
     bf16 with dtype propagation that is safe under torch.compile.
  3. Installs the Inferentia2 RNG workaround (random tensors drawn on CPU; see WORKAROUND(gen2-rng)).
  4. Patches the inference featurizer's max_tokens so the token axis is rounded up to the next
     multiple of BOLTZ_NEURON_PAD (default 32; see the PAD comment below), and moves the two
     token counts that do not compile (768, 896) to the next shape that does. Pad tokens are
     masked.
  5. Drops the unused `disto_target` feature (a [N,N,1,64] fp32 training target that no
     inference module reads).
  6. Feeds `ref_space_uid % 256` so the atom encoder's same-residue mask stays exact in bf16.
  7. Enables the bundled AOTAutograd cache, and optionally (BOLTZ_NEURON_PRECOMPILE=1) dynamo
     precompile, so new processes skip re-lowering / re-tracing the diffusion model.
  8. Hooks Boltz2.load_from_checkpoint so that AFTER the model is built it is:
       - forced to fk_steering OFF (unless BOLTZ_NEURON_STEERING=on),
       - wrapped with torch.compile(score_model, backend="neuron") under inference_mode,
       - cast to bf16 and moved to the neuron device.

Config via env (defaults = the validated configuration); see README "Configuration":
  NEURON_PLATFORM_TARGET_OVERRIDE  (SDK 2.33: leave unset; Beta 5: set to inf2 before import)
  BOLTZ_NEURON_PAD (32), BOLTZ_NEURON_PAD_MAX_ADD (-1 = no limit), BOLTZ_NEURON_DTYPE (bf16),
  BOLTZ_NEURON_BAD_SHAPES (768=832,896=928), BOLTZ_NEURON_WRITER_CPU (1),
  BOLTZ_NEURON_COMPILE_STRUCTURE (1), BOLTZ_NEURON_AOT_CACHE (1), BOLTZ_NEURON_PRECOMPILE (0),
  BOLTZ_NEURON_NUM_THREADS (2, precompile only), BOLTZ_NEURON_DROP_DISTO (1),
  BOLTZ_NEURON_UID_WRAP (1), BOLTZ_NEURON_STEERING (off)

Validated: Neuron SDK 2.33 torch-neuronx DLAMI (torch 2.14.0 / torch-neuronx 2.14.3 / neuronx-cc
2.28.7631 / nki 0.7.0) on inf2.xlarge (runtime) and inf2.8xlarge / inf2.24xlarge (compiling).
Earlier: PyTorch Native Beta 5 (torch 2.12.1 / torch-neuronx 2.12.3 / neuronx-cc 2.27.2878).
"""
import os
import sys

# patches.py ships alongside this module.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Dynamo precompile (opt-in): skip dynamo tracing entirely on repeat shapes. Must be decided BEFORE
# torch is imported -- torch reads TORCH_CACHING_PRECOMPILE at import, and torch_neuronx applies its
# precompile patches at import as well.
PRECOMPILE = os.environ.get("BOLTZ_NEURON_PRECOMPILE", "0") in ("1", "true", "on")
if PRECOMPILE:
    os.environ["TORCH_CACHING_PRECOMPILE"] = "1"


def _torch_neuronx_version():
    try:
        from importlib.metadata import version
        return tuple(int(x) for x in version("torch-neuronx").split(".")[:2])
    except Exception:
        return (0, 0)


# Neuron SDK 2.33 (torch-neuronx 2.14) on Inferentia2. Both settings are read once, when torch_neuronx
# loads, so they are made here, before torch is imported:
#  - int64 tensors: the default "native" mode passes int64 to the compiler as 64-bit integers, and on
#    Inferentia2 some eager int64 ops then return wrong values (e.g. torch.where on a [1,64,64] int64
#    tensor; Boltz-2's relative-position one_hot fails with "Class values must be non-negative").
#    "packed" mode computes on the low 32 bits, which holds every int64 value Boltz-2 uses.
#  - NEURON_PLATFORM_TARGET_OVERRIDE=inf2 is not needed (the runtime reports inf2 as "trn1"), and with
#    it set torch-neuronx 2.14 selects the NeuronCore-v3 RNG kernel, which asserts on Inferentia2.
os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "2048")
_TNX = _torch_neuronx_version()
if _TNX >= (2, 14):
    os.environ.setdefault("TORCH_NEURONX_INT64_MODE", "packed")
    if os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "").lower() in ("inf2", "trn1", "gen2"):
        os.environ.pop("NEURON_PLATFORM_TARGET_OVERRIDE")


def _instance_type():
    try:
        return open("/sys/devices/virtual/dmi/id/product_name").read().strip()
    except Exception:
        return ""


# Trainium2: run at LNC=1 (8 logical cores per device on trn2.3xlarge). With the trn2 default LNC=2,
# SDK 2.33's bf16 build of Boltz-2 loses accuracy: on two ~765-residue proteins CA-RMSD to the
# crystal went from 0.6-0.9 A to 1.6-2.2 A and lDDT-CA from 0.98 to 0.84-0.86 (fp32 at LNC=2, and bf16
# at LNC=1, are both accurate). LNC is read when the runtime starts, so it is set here. Set
# NEURON_LOGICAL_NC_CONFIG yourself to override (LNC=2 then logs a warning).
INSTANCE = _instance_type()
if INSTANCE.startswith("trn2"):
    if "NEURON_LOGICAL_NC_CONFIG" not in os.environ:
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    elif os.environ["NEURON_LOGICAL_NC_CONFIG"] == "2" and os.environ.get("BOLTZ_NEURON_DTYPE", "bf16") == "bf16":
        print("[boltz-neuron] WARNING: NEURON_LOGICAL_NC_CONFIG=2 with bf16 on trn2 is less accurate "
              "(see NOTES.md); use 1, or BOLTZ_NEURON_DTYPE=fp32", flush=True)

import torch
import torch.nn.functional as F

# Token-axis padding. On Inferentia2 one compiled graph in the diffusion model emits ~N*384 extra
# DMA descriptors unless N is a multiple of 32 (measured on every N tested, 54 of 54 from 186 to
# 1,088). Padding up to the next multiple of 32 (at most 31 masked tokens) removes them. Timed on
# an inf2.xlarge: 186->192 1.59x, 200->224 1.16x, 330->352 1.56x, 600->608 1.26x faster; 208->224
# and 464->480 within 2% (slightly slower). Net positive, so it is the default.
PAD = int(os.environ.get("BOLTZ_NEURON_PAD", "32"))
DTYPE = os.environ.get("BOLTZ_NEURON_DTYPE", "bf16")
COMPILE_STRUCTURE = os.environ.get("BOLTZ_NEURON_COMPILE_STRUCTURE", "1") not in ("0", "false", "no")
STEERING = os.environ.get("BOLTZ_NEURON_STEERING", "off") in ("on", "1", "true")
PAD_MAX_ADD = int(os.environ.get("BOLTZ_NEURON_PAD_MAX_ADD", "-1"))

# WORKAROUND(trimul-shape): token counts at which the trunk's triangle-multiplication matmul
# ([128,N,N] x [128,N,N] bf16, with its slice+transpose prologue) exceeds neuronx-cc's 5M
# instruction limit (NCC_EBVF030) on inf2. Measured with the op alone, N = 640..1088 step 32:
# only 768 (18.9M instructions) and 896 (25.7M) fail; every other N compiles on a smooth curve.
# These are shape-specific code-generation blow-ups, not a size limit. Inputs that land on one are
# padded to the next shape that compiles. Validated on real proteins against the BOLTZ_TRIMUL_KCHUNK
# fallback: 768 -> 832 (two ~765-residue proteins, 0.15-0.21 A apart, 1.53x faster) and 896 -> 928
# (two ~890-residue proteins, 0.13-0.20 A apart; seed-to-seed spread is larger in both cases).
# Remove when the compiler handles these shapes. Override with BOLTZ_NEURON_BAD_SHAPES="" (off) or
# a comma list of bad=good pairs.
_BAD_SHAPES_DEFAULT = "768=832,896=928"
BAD_SHAPES = {
    int(a): int(b)
    for a, b in (p.split("=") for p in os.environ.get("BOLTZ_NEURON_BAD_SHAPES", _BAD_SHAPES_DEFAULT).split(",") if "=" in p)
}
DROP_DISTO = os.environ.get("BOLTZ_NEURON_DROP_DISTO", "1") in ("1", "true", "on")
TARGET = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "")
_ACTIVE_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}.get(DTYPE, torch.bfloat16)


def _log(msg):
    print(f"[boltz-neuron] {msg}", flush=True)


def _install_gen2_rng_shim():
    """Random numbers on Inferentia2 (NeuronCore-v2).

    torch-neuronx's on-device RNG is an NKI kernel that needs NeuronCore-v3+. On inf2 the SDK 2.33
    release disables it (`_has_nki_rng = False`), and its fallback for an eager
    `torch.randn(..., device="neuron")` calls `torch.rand` on the device again, which recurses until
    Python's limit (the first symptom is an "Out of bounds access" NEFF error or a garbage tensor).
    Boltz-2 draws its diffusion noise in the eager sampler loop, so draw it on the CPU and copy it
    to the device: the same numbers for a given --seed, and no device RNG needed.
    WORKAROUND(gen2-rng). Remove when torch-neuronx has a working gen2 RNG fallback.
    """
    try:
        from torch_neuronx.python_ops.nki_kernels import rng_state as _rs
    except Exception as e:
        _log(f"gen2 RNG shim skipped: {e!r}")
        return
    if getattr(_rs, "_has_nki_rng", True):
        if any(g in TARGET.lower() for g in ("inf2", "trn1", "gen2")):
            # Older builds (PyTorch Native Beta 5) with the target override: the kernel exists but
            # rand_set_state asserts on gen2. Seeding is not needed for inference.
            _rs.set_rng_state = lambda *a, **k: None
            _log(f"gen2 RNG shim installed (set_rng_state disabled, target={TARGET})")
        return

    def _is_neuron(d):
        return d is not None and torch.device(d).type == "neuron"

    def _factory(orig):
        def f(*a, device=None, **k):
            if _is_neuron(device) and not torch.compiler.is_compiling():
                return orig(*a, device="cpu", **k).to(device)
            return orig(*a, device=device, **k)
        return f

    def _like(orig):
        def f(x, *a, device=None, **k):
            dev = device if device is not None else x.device
            if _is_neuron(dev) and not torch.compiler.is_compiling():
                return orig(x, *a, device="cpu", **k).to(dev)
            return orig(x, *a, device=device, **k)
        return f

    for name in ("randn", "rand", "randint"):
        setattr(torch, name, _factory(getattr(torch, name)))
    for name in ("randn_like", "rand_like"):
        setattr(torch, name, _like(getattr(torch, name)))
    _log("gen2 RNG shim installed (random tensors drawn on CPU, then moved to the device)")


def _install_pad_hook():
    """Round the token axis up to the next multiple of PAD at feature generation, then move any
    shape listed in BAD_SHAPES (see WORKAROUND(trimul-shape)) to its replacement."""
    if PAD <= 1 and not BAD_SHAPES:
        return
    try:
        import boltz.data.module.inferencev2 as _inf2
    except Exception as e:
        _log(f"pad hook skipped (no inferencev2): {e!r}")
        return
    _orig = _inf2.Boltz2Featurizer.process

    def _padded(self, tokenized, **kw):
        import math
        n = len(tokenized.tokens)
        mt = n
        if PAD > 1:
            up = int(math.ceil(n / PAD) * PAD)
            if up != n and (PAD_MAX_ADD < 0 or up - n <= PAD_MAX_ADD):
                mt = up
            elif up != n:
                _log(f"pad N {n}: next multiple of {PAD} is +{up - n} tokens (> {PAD_MAX_ADD}); running native")
        why = f"multiple of {PAD}"
        if mt in BAD_SHAPES:
            # WORKAROUND(trimul-shape): this token count does not compile; use the next good shape.
            why = f"N={mt} hits a compiler shape limit"
            mt = BAD_SHAPES[mt]
        if mt == n:
            return _orig(self, tokenized, **kw)
        kw["max_tokens"] = mt
        _log(f"pad N {n} -> {mt} (+{mt - n}, {why}; pad tokens masked)")
        return _orig(self, tokenized, **kw)

    _inf2.Boltz2Featurizer.process = _padded


def _install_drop_disto_hook():
    """Drop `disto_target` (a [N,N,1,64] fp32 distogram TRAINING target) from the features.

    The featurizer emits it but no inference module reads it, so it is dead HBM + host->device
    transfer that scales as N^2. Mirrors the Boltz-2 Neuron benchmark harness default
    (BOLTZ_DROP_DISTO_TARGET=1). Applied at the featurizer output so it never reaches collate or
    the device. Disable with BOLTZ_NEURON_DROP_DISTO=0.
    """
    if not DROP_DISTO:
        return
    try:
        import boltz.data.module.inferencev2 as _inf2
    except Exception as e:
        _log(f"disto drop skipped (no inferencev2): {e!r}")
        return
    _orig = _inf2.Boltz2Featurizer.process   # may already be the pad wrapper -- chain onto it

    def _no_disto(self, tokenized, **kw):
        feats = _orig(self, tokenized, **kw)
        if isinstance(feats, dict) and "disto_target" in feats:
            feats.pop("disto_target")
        return feats

    _inf2.Boltz2Featurizer.process = _no_disto
    _log("disto_target drop installed (dead inference feature)")


def _install_uid_wrap_hook():
    """Keep the atom encoder's same-residue mask exact in bf16 (inputs > 256 residues).

    AtomEncoder compares each query atom's `ref_space_uid` with its key-window atoms' uid after
    `to_keys(uid.float())`. Under bf16 `.float()` is bf16, which holds integers exactly only up
    to 256, so for > 256 residues neighbouring residue ids collide/round and the mask is wrong
    (CPU check: 7,544 of 26,670 pairs wrong at 384 residues). The uid is used ONLY for that
    equality, and a 128-atom key window spans far fewer than 256 residues, so `uid % 256` gives
    the identical mask while staying exact in bf16. Disable with BOLTZ_NEURON_UID_WRAP=0.
    """
    if os.environ.get("BOLTZ_NEURON_UID_WRAP", "1") in ("0", "false", "no") or DTYPE == "fp32":
        return
    try:
        import boltz.data.module.inferencev2 as _inf2
    except Exception as e:
        _log(f"uid wrap skipped (no inferencev2): {e!r}")
        return
    _orig = _inf2.Boltz2Featurizer.process

    def _wrap_uid(self, tokenized, **kw):
        feats = _orig(self, tokenized, **kw)
        if isinstance(feats, dict) and "ref_space_uid" in feats:
            feats["ref_space_uid"] = feats["ref_space_uid"] % 256
        return feats

    _inf2.Boltz2Featurizer.process = _wrap_uid
    _log("ref_space_uid %256 wrap installed (exact same-residue mask in bf16)")


def _install_writer_cpu_unpad():
    """WORKAROUND(nonzero-index): move each prediction to CPU before Boltz's writer unpads it.

    BoltzWriter does `coords[pad_mask.bool()]`. On inf2, torch.nonzero / boolean indexing on device
    returns the right number of rows but garbage index values at some lengths (all-true masks of
    5792, 5920, 5952, 5984, 6304, 6336 elements; 5,600-6,560 swept in steps of 32). The compiled
    scatter_add inside torch_neuronx's nonzero lowering corrupts the result; eager is correct.
    A 766-residue protein (5,948 atoms, padded to 5,952) crashed in the writer with
    "index 15856896 is out of bounds". The writer only formats output, so doing it on CPU costs
    nothing measurable. Remove when the Neuron nonzero lowering is fixed.
    Disable with BOLTZ_NEURON_WRITER_CPU=0.
    """
    if os.environ.get("BOLTZ_NEURON_WRITER_CPU", "1") in ("0", "false", "no"):
        return
    try:
        from boltz.data.write.writer import BoltzWriter
    except Exception as e:
        _log(f"writer CPU unpad skipped: {e!r}")
        return
    _orig = BoltzWriter.write_on_batch_end

    def _to_cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: _to_cpu(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(_to_cpu(v) for v in x)
        return x

    def _write(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        return _orig(self, trainer, pl_module, _to_cpu(prediction), batch_indices, batch, batch_idx,
                     dataloader_idx)

    BoltzWriter.write_on_batch_end = _write


def _install_trainer_hook():
    """Register a custom Lightning 'neuron' Accelerator and force the Trainer to use it.

    Boltz's CLI builds a Lightning Trainer with accelerator='gpu'; Lightning has no built-in
    neuron accelerator. Rather than fight it with accelerator='cpu' (which leaves tensors on
    CPU while the model is on neuron -> "input tensor is on cpu device, expected neuron"), we
    register a proper custom Accelerator reporting torch.device('neuron') and pass an explicit
    SingleDeviceStrategy(device=neuron). Lightning's device-transfer (move_data_to_device) is
    device-agnostic, so it then moves batches/model to the neuron device correctly.

    Two Lightning gotchas handled here:
      - strategy='auto' falls back to CPU for any non-CUDA/MPS accelerator -> pass the strategy
        explicitly with the neuron device.
      - precision='bf16-mixed' (boltz default) can assume cuda -> force '32-true' (we cast the
        model to bf16 ourselves, so no Lightning mixed-precision plugin is needed).
    """
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.accelerators import Accelerator, AcceleratorRegistry
    from pytorch_lightning.strategies import SingleDeviceStrategy

    _NEURON_DEV = torch.device("neuron", 0)

    class NeuronAccelerator(Accelerator):
        def setup_device(self, device): pass
        def teardown(self): pass
        @staticmethod
        def parse_devices(devices): return 1
        @staticmethod
        def get_parallel_devices(devices): return [torch.device("neuron", 0)]
        @staticmethod
        def auto_device_count(): return 1
        @staticmethod
        def is_available(): return True
        @staticmethod
        def name(): return "neuron"

    try:
        if "neuron" not in AcceleratorRegistry.available_accelerators():
            AcceleratorRegistry.register("neuron", NeuronAccelerator, description="AWS Neuron (PrivateUse1)")
    except Exception as e:
        _log(f"accelerator registry note: {e!r}")

    _orig_init = pl.Trainer.__init__

    def _init(self, *args, **kwargs):
        kwargs["accelerator"] = NeuronAccelerator()
        kwargs["devices"] = 1
        kwargs["strategy"] = SingleDeviceStrategy(device=_NEURON_DEV)
        kwargs["precision"] = "32-true"   # model is already bf16; no mixed-precision plugin
        return _orig_init(self, *args, **kwargs)

    pl.Trainer.__init__ = _init
    _log("Trainer -> custom NeuronAccelerator + SingleDeviceStrategy(neuron) + precision=32-true")


def _install_load_hook():
    """Wrap Boltz2.load_from_checkpoint: steering off, compile score_model, cast+move."""
    from boltz.main import Boltz2
    _orig_load = Boltz2.load_from_checkpoint.__func__  # it's a classmethod

    def _hooked(cls, *args, **kwargs):
        # Force steering off unless explicitly enabled.
        sa = kwargs.get("steering_args")
        if sa is not None and not STEERING:
            # steering_args is a dict here (asdict). Flip the fk flag off.
            if isinstance(sa, dict):
                sa["fk_steering"] = False
                sa["physical_guidance_update"] = False
        pa = kwargs.get("predict_args") or {}
        if isinstance(pa, dict) and pa:
            _log(f"predict_args: diffusion_samples={pa.get('diffusion_samples')} "
                 f"max_parallel_samples={pa.get('max_parallel_samples')}")
        model = _orig_load(cls, *args, **kwargs)
        model.eval()
        # compile_structure: wrap the diffusion score model with the neuron backend.
        if COMPILE_STRUCTURE and hasattr(model, "structure_module") and hasattr(model.structure_module, "score_model"):
            # Create the wrapper under inference_mode: Lightning's predict loop runs there, and with
            # precompile the saved guards must expect inference-tensor dispatch keys or every guard
            # fails on load and dynamo re-traces (then crashes, see _enable_precompile). Harmless
            # without precompile.
            with torch.inference_mode():
                model.structure_module.score_model = torch.compile(
                    model.structure_module.score_model, backend="neuron"
                )
            _log("compile_structure: score_model wrapped with torch.compile(backend='neuron')")
        # cast to bf16; Lightning's SingleDeviceStrategy(neuron) moves it to the device.
        model = model.to(_ACTIVE_DTYPE)
        _log(f"model cast to {DTYPE} (device move handled by Lightning NeuronAccelerator)")
        return model

    Boltz2.load_from_checkpoint = classmethod(_hooked)


def _enable_aot_cache():
    """Let warm processes reuse the traced score-model graph instead of re-lowering it.

    The neuron backend returns a serializable `NeuronOutputCode`, but PyTorch only writes an
    AOTAutograd cache entry for it when the *bundled* AOTAutograd cache is on (the non-bundled path
    needs an inductor FX-graph key the neuron backend never sets). With the default off, every
    process misses and redoes AOT dispatch + StableHLO lowering (~10 s of the first diffusion step).
    Turning it on: first score_model call 16.3 -> 6.5 s, 8eil predict 29 -> 19-20 s, output
    bit-identical. Entries go to TORCHINDUCTOR_CACHE_DIR (default /tmp/torchinductor_<user>).
    Disable with BOLTZ_NEURON_AOT_CACHE=0.
    """
    if os.environ.get("BOLTZ_NEURON_AOT_CACHE", "1") in ("0", "false", "no"):
        return
    torch._functorch.config.bundled_autograd_cache = True
    _log("bundled AOTAutograd cache enabled")


def _enable_precompile_guard():
    """Defensive fix for a torch precompile crash (`assert name not in scope`).

    A loaded precompile package leaves `__builtins_dict___<N>` in the module globals; if anything in
    that module is then traced fresh, dynamo's per-process unique-id counter (which restarts at 0)
    hands out the same name and asserts. Start the counter at a random large offset so fresh names
    never collide with names saved by an earlier process. Only matters with precompile on.
    """
    import itertools
    import random
    import torch._dynamo.bytecode_transformation as _bt
    _bt._unique_id_counter = itertools.count(random.randint(10**6, 10**9))


def apply():
    """Apply all optimizations. Call once, before boltz runs."""
    if PRECOMPILE:
        _enable_precompile_guard()
        # Dynamo's saved guards include torch.get_num_threads() (= physical cores: 16 on an
        # inf2.8xlarge, 2 on an inf2.xlarge). A package built on one size never matches on the
        # other, so pin it. Override with BOLTZ_NEURON_NUM_THREADS.
        torch.set_num_threads(int(os.environ.get("BOLTZ_NEURON_NUM_THREADS", "2")))
        _log(f"dynamo precompile enabled (BOLTZ_NEURON_PRECOMPILE=1), torch threads={torch.get_num_threads()}")
    _enable_aot_cache()
    import patches  # the shipped bf16 patch module
    patches.apply_load_patches()                       # torch.load weights_only=False + omegaconf
    patches.apply_patches(dtype=_ACTIVE_DTYPE, compile_safe=COMPILE_STRUCTURE)  # bf16 + dtype-safe
    _log(f"bf16 patches applied (dtype={DTYPE}, compile_safe={COMPILE_STRUCTURE})")
    _install_gen2_rng_shim()
    _install_pad_hook()
    _install_drop_disto_hook()
    _install_uid_wrap_hook()
    _install_writer_cpu_unpad()
    _install_trainer_hook()
    _install_load_hook()
    _log(f"ready: pad={PAD}(max_add={PAD_MAX_ADD}) compile_structure={COMPILE_STRUCTURE} steering={'on' if STEERING else 'off'} drop_disto={DROP_DISTO} bad_shapes={BAD_SHAPES or 'off'} target={TARGET or '(unset)'} instance={INSTANCE or '?'} lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG', 'default')}")
