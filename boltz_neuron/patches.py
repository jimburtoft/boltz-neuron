"""
Boltz-2 source patches for bf16 inference on AWS Neuron (used by boltz_neuron.preamble).

`apply_patches(dtype, compile_safe)` installs monkey-patches on Boltz-2 classes so that:

  1. the model runs in bf16 end to end: internal `.float()` casts and fp32 autocast islands
     are removed, and every boundary where Boltz builds fp32 tensors is cast to the active dtype;
  2. the diffusion score model traces cleanly under torch.compile (source-level patches,
     because Dynamo does not see a runtime `Tensor.float` override in every path);
  3. a few inference-only shortcuts are applied (eval-mode dropout as a scalar, empty-template
     skip), and the steering potentials stay fp32-safe when fk_steering is enabled.

Kept in fp32: the weighted_rigid_align SVD and the sigma schedule.

`apply_load_patches()` makes torch.load accept the Boltz checkpoint (weights_only=False plus an
omegaconf allowlist) and tolerates newer checkpoint kwargs.

Optional, off by default:
  BOLTZ_TRIMUL_KCHUNK=<k>          split the triangle-multiplication contraction into k-sized chunks
  BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1 treat --max_parallel_samples as a per-chunk cap
  BOLTZ_TRANSITION_Z_CHUNK=<n>     override the pair-transition chunk size (0 disables)
"""

import os

import torch
import torch.nn.functional as F  # noqa: F401  (kept: closures below resolve F the same way)

_ORIGINAL_TENSOR_FLOAT = torch.Tensor.float


_ACTIVE_DTYPE: "torch.dtype | None" = None


def _tri_mul_out_forward_clean(self, x: torch.Tensor, mask: torch.Tensor, use_kernels: bool = False) -> torch.Tensor:
    """Clean forward for TriangleMultiplicationOutgoing:
       - honors active dtype (no defensive .float() cast on the einsum)
       - use_kernels is ignored (we handle kernel selection at outer layer)
    """
    # Input norm + gating
    x = self.norm_in(x)
    x_in = x
    x = self.p_in(x) * self.g_in(x).sigmoid()
    x = x * mask.unsqueeze(-1)

    # Split -- KEEP in active dtype (no .float())
    a, b = torch.chunk(x, 2, dim=-1)

    # Triangular projection einsum
    x = torch.einsum("bikd,bjkd->bijd", a, b)

    # Output gating
    x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
    return x


def _tri_mul_in_forward_clean(self, x: torch.Tensor, mask: torch.Tensor, use_kernels: bool = False) -> torch.Tensor:
    """Clean forward for TriangleMultiplicationIncoming (same as outgoing but einsum order differs)."""
    x = self.norm_in(x)
    x_in = x
    x = self.p_in(x) * self.g_in(x).sigmoid()
    x = x * mask.unsqueeze(-1)

    a, b = torch.chunk(x, 2, dim=-1)

    x = torch.einsum("bkid,bkjd->bijd", a, b)

    x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
    return x


def _tri_mul_kchunked_einsum(a, b, direction, k_chunk):
    """Accumulate the tri_mul einsum over the contraction axis k in chunks.

    a, b: [B, N, N, D] (the two halves of the gate product, already split).
    direction: 'out' -> contract axis 2 (k in a[b,i,k,d], b[b,j,k,d]).
               'in'  -> contract axis 1 (k in a[b,k,i,d], b[b,k,j,d]).
    Returns [B, N, N, D].
    """
    import torch
    B, d0, d1, D = a.shape
    # k is the contraction axis
    if direction == "out":
        # "bikd,bjkd->bijd": k is axis 2
        N = a.shape[2]
        i_ax = 1
    else:
        # "bkid,bkjd->bijd": k is axis 1
        N = a.shape[1]
        i_ax = 2

    out = None
    for ks in range(0, N, k_chunk):
        ke = min(ks + k_chunk, N)
        if direction == "out":
            a_c = a[:, :, ks:ke, :]
            b_c = b[:, :, ks:ke, :]
            part = torch.einsum("bikd,bjkd->bijd", a_c, b_c)
        else:
            a_c = a[:, ks:ke, :, :]
            b_c = b[:, ks:ke, :, :]
            part = torch.einsum("bkid,bkjd->bijd", a_c, b_c)
        out = part if out is None else out + part
    return out


def _tri_mul_out_forward_kchunk(self, x, mask, use_kernels=False):
    import os, torch
    k_chunk = int(os.environ.get("BOLTZ_TRIMUL_KCHUNK", "0") or "0")
    x = self.norm_in(x)
    x_in = x
    x = self.p_in(x) * self.g_in(x).sigmoid()
    x = x * mask.unsqueeze(-1)
    a, b = torch.chunk(x, 2, dim=-1)
    if k_chunk and not self.training:
        x = _tri_mul_kchunked_einsum(a, b, "out", k_chunk)
    else:
        x = torch.einsum("bikd,bjkd->bijd", a, b)
    x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
    return x


def _tri_mul_in_forward_kchunk(self, x, mask, use_kernels=False):
    import os, torch
    k_chunk = int(os.environ.get("BOLTZ_TRIMUL_KCHUNK", "0") or "0")
    x = self.norm_in(x)
    x_in = x
    x = self.p_in(x) * self.g_in(x).sigmoid()
    x = x * mask.unsqueeze(-1)
    a, b = torch.chunk(x, 2, dim=-1)
    if k_chunk and not self.training:
        x = _tri_mul_kchunked_einsum(a, b, "in", k_chunk)
    else:
        x = torch.einsum("bkid,bkjd->bijd", a, b)
    x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
    return x


def _attention_pair_bias_v1_forward_clean(self, s, z, mask, multiplicity=1,
                                           to_keys=None, model_cache=None):
    """Clean forward for v1 AttentionPairBias: drops the fp32 autocast island.

    Signature matches boltz.model.layers.attention.AttentionPairBias.forward exactly.
    """
    B = s.shape[0]

    # Initial layer norm (v1-only)
    if self.initial_norm:
        s = self.norm_s(s)

    if to_keys is not None:
        k_in = to_keys(s)
        mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
    else:
        k_in = s

    q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
    k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
    v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

    # Cached z projection (during diffusion roll-out)
    if model_cache is None or "z" not in model_cache:
        z = self.proj_z(z)
        if model_cache is not None:
            model_cache["z"] = z
    else:
        z = model_cache["z"]
    z = z.repeat_interleave(multiplicity, 0)

    g = self.proj_g(s).sigmoid()

    # Attention -- keep in active dtype, no autocast, no .float()
    attn = torch.einsum("bihd,bjhd->bhij", q, k)
    attn = attn / (self.head_dim ** 0.5) + z
    attn = attn + (1 - mask[:, None, None]) * -self.inf
    # Softmax may upcast to fp32 internally under Dynamo. Cast back to v's
    # dtype to keep the downstream attn @ v matmul dtype-consistent.
    attn = attn.softmax(dim=-1).to(v.dtype)
    o = torch.einsum("bhij,bjhd->bihd", attn, v)

    o = o.reshape(B, -1, self.c_s)
    o = self.proj_o(g * o)
    return o


def _attention_pair_bias_v2_forward_clean(self, s, z, mask, k_in, multiplicity=1):
    """Clean forward for v2 AttentionPairBias: drops the fp32 autocast island."""
    B = s.shape[0]

    q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
    k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
    v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

    # v2 has a compute_pair_bias flag; proj_z is either LayerNorm+Linear+Rearrange or just Rearrange
    bias = self.proj_z(z)
    bias = bias.repeat_interleave(multiplicity, 0)

    g = self.proj_g(s).sigmoid()

    attn = torch.einsum("bihd,bjhd->bhij", q, k)
    attn = attn / (self.head_dim ** 0.5) + bias
    attn = attn + (1 - mask[:, None, None]) * -self.inf
    # Softmax may upcast to fp32 internally under Dynamo. Cast back to v's
    # dtype to keep the downstream attn @ v matmul dtype-consistent.
    attn = attn.softmax(dim=-1).to(v.dtype)
    o = torch.einsum("bhij,bjhd->bihd", attn, v)

    o = o.reshape(B, -1, self.c_s)
    o = self.proj_o(g * o)
    return o


def _transition_z_chunk_for(n_tokens: int):
    """Chunk size for the pair transition, or None to leave it unchunked.

    Mirrors upstream's own policy (64 when N > const.chunk_size_threshold), which
    upstream computes and then fails to apply. BOLTZ_TRANSITION_Z_CHUNK overrides:
    an int forces that chunk size at every N; "0" disables chunking entirely.
    """
    import os as _os
    from boltz.data import const

    _ov = _os.environ.get("BOLTZ_TRANSITION_Z_CHUNK", "")
    if _ov:
        v = int(_ov)
        return v if v > 0 else None
    return 64 if n_tokens > const.chunk_size_threshold else None


def _pairformer_layer_forward_clean(self, s, z, mask=None, pair_mask=None,
                                     chunk_size_tri_attn=None, use_kernels=False,
                                     use_cuequiv_mul=False, use_cuequiv_attn=False):
    """Clean PairformerLayer.forward: no dropout, no autocast, no .float() casts.

    Runs everything in whatever dtype the inputs arrive at. Signature matches upstream.
    """
    # Pairwise stack (patched tri_mul_out / tri_mul_in / tri_attn honor active dtype)
    z = z + self.tri_mul_out(z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels)
    z = z + self.tri_mul_in(z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels)
    z = z + self.tri_att_start(z, mask=pair_mask, chunk_size=chunk_size_tri_attn,
                                use_kernels=use_cuequiv_attn or use_kernels)
    z = z + self.tri_att_end(z, mask=pair_mask, chunk_size=chunk_size_tri_attn,
                              use_kernels=use_cuequiv_attn or use_kernels)
    # Honor a pair-transition chunk if one is set on the layer; None -> stock behavior.
    z = z + self.transition_z(z, getattr(self, "_m01_transition_z_chunk", None))

    # Sequence stack -- no autocast, no .float() casts.
    s_normed = self.pre_norm_s(s)
    s = s + self.attention(s=s_normed, z=z, mask=mask, k_in=s_normed)
    s = s + self.transition_s(s)
    s = self.s_post_norm(s)

    return s, z


def _outer_product_mean_forward_clean(self, m, mask, chunk_size=None):
    """Clean OuterProductMean.forward: no .float() cast on the outer-product einsum.

    Signature matches boltz.model.layers.outer_product_mean.OuterProductMean.forward.
    Only the non-chunked branch has the .float() cast; the chunked branch is
    already dtype-consistent.
    """
    # Expand mask
    mask = mask.unsqueeze(-1).to(m)

    # Compute projections
    m = self.norm(m)
    a = self.proj_a(m) * mask
    b = self.proj_b(m) * mask

    if chunk_size is not None and not self.training:
        # Chunked branch: original code (no defensive .float(), already dtype-consistent)
        for i in range(0, mask.shape[1], 64):
            if i == 0:
                num_mask = (
                    mask[:, i : i + 64, None, :] * mask[:, i : i + 64, :, None]
                ).sum(1)
            else:
                num_mask += (
                    mask[:, i : i + 64, None, :] * mask[:, i : i + 64, :, None]
                ).sum(1)
        num_mask = num_mask.clamp(min=1)

        for i in range(0, self.c_hidden, chunk_size):
            a_chunk = a[:, :, :, i : i + chunk_size]
            sliced_weight_proj_o = self.proj_o.weight[
                :, i * self.c_hidden : (i + chunk_size) * self.c_hidden
            ]
            z = torch.einsum("bsic,bsjd->bijcd", a_chunk, b)
            z = z.reshape(*z.shape[:3], -1)
            z = z / num_mask
            if i == 0:
                z_out = z.to(m) @ sliced_weight_proj_o.T
            else:
                z_out = z_out + z.to(m) @ sliced_weight_proj_o.T

        z_out = z_out + self.proj_o.bias
        return z_out
    else:
        # Non-chunked branch: drop the .float() cast on a and b
        mask2 = mask[:, :, None, :] * mask[:, :, :, None]
        num_mask = mask2.sum(1).clamp(min=1)
        # No .float() -- keep in active dtype
        z = torch.einsum("bsic,bsjd->bijcd", a, b)
        z = z.reshape(*z.shape[:3], -1)
        z = z / num_mask
        z = self.proj_o(z.to(m))
        return z


def _single_to_keys_dtype_safe(single, indexing_matrix, W, H):
    """Boltz `single_to_keys` calls torch.einsum(single, indexing_matrix) but
    `indexing_matrix` is forced to fp32 by `get_indexing_matrix().float()`. When
    `single` is bf16, the einsum fails on Neuron. Patch: cast
    indexing_matrix to match `single`'s dtype.
    """
    B, N, D = single.shape
    K = N // W
    single = single.view(B, 2 * K, W // 2, D)
    if indexing_matrix.dtype != single.dtype:
        indexing_matrix = indexing_matrix.to(single.dtype)
    return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
        B, K, H, D
    )


def _make_single_to_keys_dtype_safe(target_dtype):
    """Factory variant for compile_safe mode: force BOTH einsum operands to
    target_dtype so the output is unambiguously target_dtype. This is needed
    because in the compiled score model, `single` may arrive as fp32 (e.g.
    from a mask.unsqueeze(-1) path that wasn't caught by a forward patch), and
    casting indexing_matrix to single.dtype would then produce an fp32 einsum
    output that dtype-mismatches downstream bf16 matmuls.
    """
    def _single_to_keys(single, indexing_matrix, W, H):
        B, N, D = single.shape
        K = N // W
        single = single.view(B, 2 * K, W // 2, D)
        # Cast both operands to target_dtype. Use UNCONDITIONAL casts (no
        # `if dtype != target` guard): under --compile_structure such guards are
        # Python branches Dynamo can specialize and trace-out, silently dropping
        # the cast and producing an aten.bmm dtype mismatch. Keep only the
        # structural is_floating_point() guard on `single` so integer/index
        # tensors are never float-cast (that changes downstream code paths).
        if single.is_floating_point():
            single = single.to(target_dtype)
        # Unconditional cast for indexing_matrix: it is always a float selection
        # matrix. Under --compile_structure, get_indexing_matrix()'s .float() is
        # traced as fp32 (the global bypass monkeypatch is NOT seen by Dynamo),
        # and a guarded `if dtype != target` branch gets specialized/traced-out,
        # silently dropping the cast -> bmm dtype mismatch. Always emit the cast.
        indexing_matrix = indexing_matrix.to(target_dtype)
        return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
            B, K, H, D
        )
    return _single_to_keys


_INDEXING_MATRIX_CACHE = {}


_INDEXING_MATRIX_STATS = {"hit": 0, "miss": 0}


_P10_CACHE_DISABLED = os.environ.get("GRIDCP_P10_DISABLE_CACHE") == "1"


def _make_get_indexing_matrix_dtype_aware(target_dtype: torch.dtype):
    """Return a replacement for encodersv2.get_indexing_matrix that returns
    a tensor in `target_dtype` (bf16) instead of the hard-coded fp32.

    The result is a **pure function of (K, W, H, device, target_dtype)** --
    there is no data dependence -- so it is memoized. Boltz calls this once per
    `single_to_keys` invocation and rebuilds an identical `one_hot` matrix every
    time.

    The cached tensor is returned directly (not cloned). That is safe only because
    every consumer treats it as read-only: `_single_to_keys` above does
    `indexing_matrix.to(target_dtype)` (a no-op copy when already in target dtype)
    and then feeds it to `torch.einsum`, neither of which mutates in place. The
    tensor is also built under `no_grad` and detached so it can never accumulate
    graph state across calls.
    """
    def _get_indexing_matrix(K, W, H, device):
        from torch.nn.functional import one_hot
        assert W % 2 == 0
        assert H % (W // 2) == 0
        h = H // (W // 2)
        assert h % 2 == 0

        key = (int(K), int(W), int(H), str(device), str(target_dtype))
        # GRIDCP_P10_DISABLE_CACHE=1 bypasses the memo so the SAME binary can produce the
        # A/B baseline. Read once at module import (see _P10_CACHE_DISABLED) rather than
        # per call -- an os.environ read inside a traced region is a NKI/Dynamo hazard.
        if not _P10_CACHE_DISABLED:
            cached = _INDEXING_MATRIX_CACHE.get(key)
            if cached is not None:
                _INDEXING_MATRIX_STATS["hit"] += 1
                return cached
        _INDEXING_MATRIX_STATS["miss"] += 1

        with torch.no_grad():
            arange = torch.arange(2 * K, device=device)
            index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(min=0, max=h + 1)
            index = index.view(K, 2, 2 * K)[:, 0, :]
            onehot = one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
            out = onehot.reshape(2 * K, h * K).to(target_dtype).detach()

        if not _P10_CACHE_DISABLED:
            _INDEXING_MATRIX_CACHE[key] = out
        return out
    return _get_indexing_matrix


def _make_preconditioned_forward_dtype_safe(target_dtype: torch.dtype):
    """Return a patched preconditioned_network_forward that casts noised_atom_coords
    and sigma to target_dtype before calling score_model. The score_model has bf16
    weights but the diffusion loop generates fp32 atom_coords via torch.randn.

    Also casts all fp32 tensors in network_condition_kwargs (s_inputs, s_trunk,
    diffusion_conditioning contents) to target_dtype. This fixes a dtype mismatch
    where Boltz2.forward:555-559 explicitly `.float()` casts s_trunk, s_inputs,
    atom_mask before calling structure_module.sample, sending fp32 tensors into
    the compiled bf16 score_model.
    """
    from einops import rearrange

    def _cast_kwargs_to_target(kwargs):
        """Recursively cast fp32 floating-point tensors in the kwargs dict to
        target_dtype. Leaves bool, int, and already-target-dtype tensors alone.
        Preserves nested dicts (like diffusion_conditioning) but only casts
        their tensor values.
        """
        out = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point() and v.dtype != target_dtype:
                    out[k] = v.to(target_dtype)
                else:
                    out[k] = v
            elif isinstance(v, dict):
                out[k] = {kk: (vv.to(target_dtype) if isinstance(vv, torch.Tensor)
                                                    and vv.is_floating_point()
                                                    and vv.dtype != target_dtype
                                                    else vv)
                          for kk, vv in v.items()}
            else:
                # callables, lists, ints, None, etc. -- pass through
                out[k] = v
        return out

    def _pnw(self, noised_atom_coords, sigma, network_condition_kwargs):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device
        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        # Cast inputs to target_dtype at score_model boundary.
        # This includes r_noisy (atom coords, always generated as fp32 by torch.randn
        # even when weights are bf16) and times.
        r_in = (self.c_in(padded_sigma) * noised_atom_coords).to(target_dtype)
        t_in = self.c_noise(sigma).to(target_dtype)

        # ALSO cast network_condition_kwargs -- Boltz2.forward:555-559 explicitly
        # .float() casts s_trunk, s_inputs, atom_mask. Without this second cast,
        # fp32 tensors leak into the compiled score_model and dtype-mismatch at
        # torch.aten.mm against bf16 Linear weights.
        kwargs_bf16 = _cast_kwargs_to_target(network_condition_kwargs)

        r_update = self.score_model(
            r_noisy=r_in,
            times=t_in,
            **kwargs_bf16,
        )

        # Cast back to fp32 for downstream (SVD-based weighted_rigid_align needs fp32)
        r_update = r_update.to(noised_atom_coords.dtype)
        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * r_update
        )
        # guarantee the returned denoised coords have EXACTLY the same
        # dtype as the input noised coords. In the batched-diffusion sampler
        # (AtomDiffusion.sample) the output buffer is allocated as
        # `torch.zeros_like(atom_coords_noisy)` (fp32, since atom_coords come from
        # torch.randn), then filled via `atom_coords_denoised[chunk_ids] = chunk`.
        # That index-assignment lowers to scatter(), which asserts
        # self.dtype == src.dtype. When diffusion_samples>=2 in bf16 mode the
        # c_skip/c_out sigma coefficients or a compiled score_model return can
        # promote `denoised_coords` to a dtype different from the fp32 buffer,
        # tripping `scatter(): Expected self.dtype to be equal to src.dtype`.
        # This unconditional cast pins the return dtype to the buffer dtype and
        # closes the ds>=2 gap without touching any load-bearing fp32 upcasts
        # (SVD rigid-align, sigma schedule remain fp32).
        if denoised_coords.dtype != noised_atom_coords.dtype:
            denoised_coords = denoised_coords.to(noised_atom_coords.dtype)
        return denoised_coords

    return _pnw


def _fourier_embedding_forward_clean(self, times):
    """FourierEmbedding.forward: cast `times` to match self.proj.weight dtype.

    In bf16 mode, `times` (sigma schedule) is hard-coded fp32 in Boltz-2 but
    self.proj.weight is bf16 -> matmul dtype mismatch. Fix by casting `times`.
    """
    from einops import rearrange
    from math import pi
    times = rearrange(times, "b -> b 1")
    if times.dtype != self.proj.weight.dtype:
        times = times.to(self.proj.weight.dtype)
    rand_proj = self.proj(times)
    return torch.cos(2 * pi * rand_proj)


def _make_confidence_forward_dtype_safe(target_dtype: torch.dtype, original_forward):
    """Wrap ConfidenceModule.forward so that x_pred is cast to target_dtype at entry.
    This makes the entire confidence module run in bf16 (matching model weights),
    avoiding dtype mismatches at `torch.bmm(token_to_rep_atom.float(), x_pred)`.

    We keep pred_distogram_logits in whatever dtype it arrives at (torch.softmax
    handles mixed).
    """
    def _fwd(self, s_inputs, s, z, x_pred, feats, pred_distogram_logits,
             multiplicity=1, run_sequentially=False, use_kernels=False):
        # Cast x_pred to target dtype at boundary
        if x_pred.dtype != target_dtype:
            x_pred = x_pred.to(target_dtype)
        # s, s_inputs, z should already be target_dtype (from trunk in bf16)
        return original_forward(
            self, s_inputs, s, z, x_pred, feats, pred_distogram_logits,
            multiplicity=multiplicity, run_sequentially=run_sequentially,
            use_kernels=use_kernels,
        )
    return _fwd


def _make_diffusion_module_forward_dtype_safe(target_dtype: torch.dtype):
    """Return a Dynamo-safe replacement for DiffusionModule.forward.

    Replaces all internal .float() calls with .to(target_dtype). This makes the
    compiled score model (via --compile_structure) propagate bf16 dtype
    consistently instead of upcasting to fp32 at every conditioning boundary.

    Source: boltz/model/modules/diffusionv2.py:116-177 (DiffusionModule.forward).
    """
    def _fwd(self, s_inputs, s_trunk, r_noisy, times, feats,
             diffusion_conditioning, multiplicity=1):
        if self.activation_checkpointing and self.training:
            s, normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )
        else:
            s, normed_fourier = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens.
        # All 3 conditioning tensors were .float() in the original -- now cast
        # to target_dtype instead.
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].to(target_dtype),
            c=diffusion_conditioning["c"].to(target_dtype),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].to(target_dtype),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,  # Float['b m 3'],
            multiplicity=multiplicity,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        # mask.float() -> mask.to(target_dtype)
        # token_trans_bias.float() -> .to(target_dtype)
        a = self.token_transformer(
            a,
            mask=mask.to(target_dtype),
            s=s,
            bias=diffusion_conditioning["token_trans_bias"].to(target_dtype),
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        # atom_dec_bias.float() -> .to(target_dtype)
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].to(target_dtype),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return r_update
    return _fwd


def _make_atom_attention_encoder_forward_dtype_safe(target_dtype: torch.dtype):
    """Dynamo-safe AtomAttentionEncoder.forward (structure_prediction=True path).

    Original (encodersv2.py:452-494) uses autocast(enabled=False) + .float() at
    lines 484-485 for the atom_to_token bmm. We replace .float() with
    .to(target_dtype) so the bmm runs in bf16 consistently.
    """
    def _fwd(self, feats, q, c, atom_enc_bias, to_keys,
             r=None, multiplicity=1):
        B, N, _ = feats["ref_pos"].shape
        atom_mask = feats["atom_pad_mask"].bool()  # Bool['b m']

        if self.structure_prediction:
            q = q.repeat_interleave(multiplicity, 0)
            r_to_q = self.r_to_q_trans(r)
            q = q + r_to_q

        c = c.repeat_interleave(multiplicity, 0)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        q = self.atom_encoder(
            q=q,
            mask=atom_mask,
            c=c,
            bias=atom_enc_bias,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        # ORIGINAL: with autocast(q.device.type, enabled=False):
        #             q_to_a = self.atom_to_token_trans(q).float()
        #             atom_to_token = feats["atom_to_token"].float()
        #             ...
        # PATCHED: run without autocast override, cast to target_dtype.
        q_to_a = self.atom_to_token_trans(q).to(target_dtype)
        atom_to_token = feats["atom_to_token"].to(target_dtype)
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        # Original follows with a = a.to(q); no-op if both are bf16.
        a = a.to(q.dtype)
        return a, q, c, to_keys
    return _fwd


def _make_atom_attention_decoder_forward_dtype_safe(target_dtype: torch.dtype):
    """Dynamo-safe AtomAttentionDecoder.forward.

    Original (encodersv2.py:536-566) uses autocast(enabled=False) + .float() at
    lines 547, 550 for the atom_to_token bmm. Same fix as encoder.
    """
    def _fwd(self, a, q, c, atom_dec_bias, feats, to_keys, multiplicity=1):
        # ORIGINAL: with autocast(a.device.type, enabled=False):
        #             atom_to_token = feats["atom_to_token"].float()
        #             ...
        #             a_to_q = self.a_to_q_trans(a.float())
        # PATCHED: cast to target_dtype.
        atom_to_token = feats["atom_to_token"].to(target_dtype)
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

        a_to_q = self.a_to_q_trans(a.to(target_dtype))
        a_to_q = torch.bmm(atom_to_token, a_to_q)

        q = q + a_to_q.to(q.dtype)
        atom_mask = feats["atom_pad_mask"]  # Bool['b m']
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        q = self.atom_decoder(
            q=q,
            mask=atom_mask,
            c=c,
            bias=atom_dec_bias,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        # Final projection atom_s (e.g. 128) -> 3 (xyz coord update). This was
        # dropped in the earlier patch, causing r_update to be [B, N, 128]
        # instead of [B, N, 3] and breaking the c_skip*coords + c_out*r_update
        # broadcast in preconditioned_network_forward.
        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update
    return _fwd


def _make_atom_transformer_forward_dtype_safe(target_dtype: torch.dtype):
    """Dynamo-safe AtomTransformer.forward.

    Original (transformers.py:279-322) has `mask.float()` at line 313 which
    upcasts the bool mask to fp32 -- passed as `mask=` to DiffusionTransformer.
    Under torch.compile, this force-fp32 breaks the bf16 dtype chain.

    Fix: replace mask.float() with mask.to(target_dtype). Bool -> bf16 is a
    valid conversion; the DiffusionTransformer's downstream matmul expects
    bf16 to match the bf16 weights.

    AtomTransformer is called from AtomAttentionEncoder.atom_encoder and
    AtomAttentionDecoder.atom_decoder. Compile enters this method via those
    call paths.
    """
    def _fwd(self, q, c, p, to_keys=None, mask=None, multiplicity=1,
             model_cache=None):
        W = self.attn_window_queries
        H = self.attn_window_keys

        if W is not None:
            B, N, D = q.shape
            NW = N // W

            q = q.view((B * NW, W, -1))
            c = c.view((B * NW, W, -1))
            if mask is not None:
                mask = mask.view(B * NW, W)
            p = p.repeat_interleave(multiplicity, 0)
            p = p.view((p.shape[0] * NW, W, H, -1))

            to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)
        else:
            to_keys_new = None

        # ORIGINAL: mask=mask.float()
        # PATCHED: cast to target_dtype instead
        mask_cast = mask.to(target_dtype) if mask is not None else None

        q = self.diffusion_transformer(
            a=q,
            s=c,
            z=p,
            mask=mask_cast,
            multiplicity=1,
            to_keys=to_keys_new,
            model_cache=model_cache,
        )

        if W is not None:
            q = q.view((B, NW * W, D))

        return q
    return _fwd


def _apply_score_model_dtype_patches(target_dtype: torch.dtype):
    """Install Dynamo-safe patches on the score model's forward call tree.

    These replace .float() with .to(target_dtype) at SOURCE level inside:
      - DiffusionModule.forward (score model root, compiled by --compile_structure)
      - AtomAttentionEncoder.forward (structure_prediction=True path)
      - AtomAttentionDecoder.forward
      - AtomTransformer.forward (called transitively via atom_encoder/atom_decoder)
      - AtomDiffusion.preconditioned_network_forward (casts kwargs into score_model)
      - nn.Linear.forward (nuclear option -- casts input to weight.dtype at
        every Linear call; catches fp32 leaks anywhere in the model tree)

    Returns list of patch descriptions installed.
    """
    installed = []
    try:
        from boltz.model.modules.diffusionv2 import DiffusionModule
        DiffusionModule.forward = _make_diffusion_module_forward_dtype_safe(target_dtype)
        installed.append(f"DiffusionModule.forward -> Dynamo-safe (target={target_dtype})")
    except (ImportError, AttributeError) as e:
        installed.append(f"DiffusionModule.forward patch SKIPPED: {e}")

    try:
        from boltz.model.modules.encodersv2 import AtomAttentionEncoder
        AtomAttentionEncoder.forward = _make_atom_attention_encoder_forward_dtype_safe(target_dtype)
        installed.append(f"AtomAttentionEncoder.forward -> Dynamo-safe (target={target_dtype})")
    except (ImportError, AttributeError) as e:
        installed.append(f"AtomAttentionEncoder.forward patch SKIPPED: {e}")

    try:
        from boltz.model.modules.encodersv2 import AtomAttentionDecoder
        AtomAttentionDecoder.forward = _make_atom_attention_decoder_forward_dtype_safe(target_dtype)
        installed.append(f"AtomAttentionDecoder.forward -> Dynamo-safe (target={target_dtype})")
    except (ImportError, AttributeError) as e:
        installed.append(f"AtomAttentionDecoder.forward patch SKIPPED: {e}")

    try:
        from boltz.model.modules.transformers import AtomTransformer
        AtomTransformer.forward = _make_atom_transformer_forward_dtype_safe(target_dtype)
        installed.append(f"AtomTransformer.forward -> Dynamo-safe (target={target_dtype})")
    except (ImportError, AttributeError) as e:
        installed.append(f"AtomTransformer.forward patch SKIPPED: {e}")

    # Crucial fix: preconditioned_network_forward needs to cast the
    # full network_condition_kwargs (s_inputs, s_trunk, ...) to target_dtype
    # because Boltz2.forward:555 explicitly .float()-casts s_trunk and s_inputs
    # before calling structure_module.sample. Without this, fp32 tensors leak
    # into the compiled score_model.
    try:
        from boltz.model.modules.diffusionv2 import AtomDiffusion
        AtomDiffusion.preconditioned_network_forward = _make_preconditioned_forward_dtype_safe(target_dtype)
        installed.append(f"AtomDiffusion.preconditioned_network_forward -> Dynamo-safe kwargs cast (target={target_dtype})")
    except (ImportError, AttributeError) as e:
        installed.append(f"AtomDiffusion.preconditioned_network_forward patch SKIPPED: {e}")



    # Nuclear option for Dynamo: patch nn.Linear.forward to auto-cast
    # input dtype to match weight dtype. This handles the case where Dynamo
    # compiles individual Linear.forward frames as separate NEFFs (which
    # happens with torch_neuronx.neuron_dynamo_backend under fullgraph=False):
    # any Linear call with fp32 input + bf16 weight would produce a
    # torch.aten.mm dtype mismatch.
    #
    # Rationale: Boltz's original code relies on autocast(enabled=False) blocks
    # + .float() casts to force fp32 compute. Under Dynamo, those autocast
    # contexts don't propagate through the compiled graph. The .float() casts
    # are traced literally (fp32 output), breaking the bf16 weight assumption.
    # Auto-casting input to weight.dtype at Linear.forward is a safe universal
    # fix: works in eager, works in Dynamo, preserves bf16 weight throughput.
    _apply_linear_forward_dtype_safe(target_dtype)
    installed.append(f"nn.Linear.forward -> auto-cast input to weight.dtype (Dynamo-safe universal fix)")

    # Force single_to_keys to emit target_dtype (compile_safe variant) -- the
    # atom windowing einsum feeds into the compiled score model's attention.
    try:
        import boltz.model.modules.encodersv2 as _encv2
        _encv2.single_to_keys = _make_single_to_keys_dtype_safe(target_dtype)
        installed.append(f"encodersv2.single_to_keys -> force target_dtype (compile_safe variant)")
    except (ImportError, AttributeError) as e:
        installed.append(f"single_to_keys compile_safe patch SKIPPED: {e}")

    # RelativePositionEncoder.forward -- cast the final cat to the linear
    # weight dtype so --compile_pairformer/--compile_msa don't hit a bf16 x fp32
    # matmul at encodersv2.py:109 (the trunk-compile blocker).
    try:
        import boltz.model.modules.encodersv2 as _encv2
        _encv2.RelativePositionEncoder.forward = _make_rel_pos_encoder_forward_dtype_safe(target_dtype)
        installed.append("encodersv2.RelativePositionEncoder.forward -> dtype-safe cat (trunk-compile fix)")
    except (ImportError, AttributeError) as e:
        installed.append(f"RelativePositionEncoder compile_safe patch SKIPPED: {e}")

    # ------------------------------------------------------------------
    return installed


def _apply_linear_forward_dtype_safe(target_dtype: torch.dtype):
    """Monkey-patch nn.Linear.forward to auto-cast input to weight.dtype.

    This is a universal fix for the fp32-input × bf16-weight mismatch that
    fires under torch.compile / Dynamo when Boltz's .float() casts produce
    fp32 inputs to bf16 Linear layers.

    Original: `return F.linear(input, self.weight, self.bias)`
    Patched:  cast input to self.weight.dtype if they differ.
    """
    import torch.nn.functional as F

    def _linear_forward_dtype_safe(self, input: torch.Tensor) -> torch.Tensor:
        if input.dtype != self.weight.dtype and input.is_floating_point():
            input = input.to(self.weight.dtype)
        return F.linear(input, self.weight, self.bias)

    torch.nn.Linear.forward = _linear_forward_dtype_safe

    # Also patch torch.bmm to align dtypes at the boundary. bmm is a direct op
    # (not through a Module), so Dynamo captures whatever dtype is passed at
    # the call site. Boltz has many `torch.bmm(atom_to_token.float(), x)` sites
    # that produce fp32 × bf16 mismatches under compile. Auto-cast to the
    # higher-precision operand's dtype (or target_dtype if it differs).
    _original_bmm = torch.bmm

    def _bmm_dtype_safe(input: torch.Tensor, mat2: torch.Tensor, *args, **kwargs):
        if input.dtype != mat2.dtype and input.is_floating_point() and mat2.is_floating_point():
            # Prefer target_dtype (typically bf16). If neither operand is already
            # target_dtype, use the higher-precision one.
            if mat2.dtype == target_dtype:
                input = input.to(target_dtype)
            elif input.dtype == target_dtype:
                mat2 = mat2.to(target_dtype)
            else:
                # Both differ from target; align to input's dtype conservatively.
                mat2 = mat2.to(input.dtype)
        return _original_bmm(input, mat2, *args, **kwargs)

    torch.bmm = _bmm_dtype_safe


def _make_rel_pos_encoder_forward_dtype_safe(target_dtype: torch.dtype):
    """compile_safe: RelativePositionEncoder.forward (encodersv2.py:49-120) ends with
       p = linear_layer(cat([a_rel_pos.float(), a_rel_token.float(),
                             b_same_entity.float(), a_rel_chain.float()]))
    The .float() casts feed fp32 into a bf16 Linear. In eager the .float() bypass hides
    it; under torch.compile (trunk compilation) it reaches torch-MLIR as a
    bf16 x fp32 matmul and fails to legalize.

    This drop-in forward is byte-identical to upstream EXCEPT the final cat is cast to
    the linear weight dtype before the matmul. Bit-exact in bf16 (the cast is what the
    Linear would do anyway; one-hots are exact in bf16 for these small vocab sizes).
    """
    from torch.nn.functional import one_hot

    def _forward(self, feats):
        b_same_chain = torch.eq(feats["asym_id"][:, :, None], feats["asym_id"][:, None, :])
        b_same_residue = torch.eq(feats["residue_index"][:, :, None], feats["residue_index"][:, None, :])
        b_same_entity = torch.eq(feats["entity_id"][:, :, None], feats["entity_id"][:, None, :])

        d_residue = feats["residue_index"][:, :, None] - feats["residue_index"][:, None, :]
        if self.cyclic_pos_enc and torch.any(feats["cyclic_period"] > 0):
            period = torch.where(feats["cyclic_period"] > 0, feats["cyclic_period"],
                                 torch.zeros_like(feats["cyclic_period"]) + 10000)
            d_residue = (d_residue - period * torch.round(d_residue / period)).long()
        d_residue = torch.clip(d_residue + self.r_max, 0, 2 * self.r_max)
        d_residue = torch.where(b_same_chain, d_residue, torch.zeros_like(d_residue) + 2 * self.r_max + 1)
        a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)

        d_token = torch.clip(feats["token_index"][:, :, None] - feats["token_index"][:, None, :] + self.r_max,
                             0, 2 * self.r_max)
        d_token = torch.where(b_same_chain & b_same_residue, d_token,
                              torch.zeros_like(d_token) + 2 * self.r_max + 1)
        a_rel_token = one_hot(d_token, 2 * self.r_max + 2)

        d_chain = torch.clip(feats["sym_id"][:, :, None] - feats["sym_id"][:, None, :] + self.s_max,
                             0, 2 * self.s_max)
        d_chain = torch.where((~b_same_entity) if self.fix_sym_check else b_same_chain,
                              torch.zeros_like(d_chain) + 2 * self.s_max + 1, d_chain)
        a_rel_chain = one_hot(d_chain, 2 * self.s_max + 2)

        wdt = self.linear_layer.weight.dtype
        cat = torch.cat(
            [a_rel_pos.to(wdt), a_rel_token.to(wdt),
             b_same_entity.unsqueeze(-1).to(wdt), a_rel_chain.to(wdt)],
            dim=-1,
        )
        return self.linear_layer(cat)

    return _forward


def apply_patches(
    dtype: torch.dtype = torch.float32,
    apply_pairformer: bool = True,
    apply_attention: bool = True,
    apply_opm: bool = True,
    apply_global_float_bypass: bool = True,
    compile_safe: bool = False,
    skip_empty_templates: bool = True,
    steering_potentials_dtype_fix: bool = True,
    pair_pad_mask_gather_workaround: bool = True,
) -> dict:
    """Install monkey-patches on Boltz-2 classes.

    dtype: active dtype; in bf16 the patched forwards do not cast to fp32 internally.
    compile_safe: also install the Dynamo-safe score-model patches (needed with
        torch.compile on the score model).
    Returns a dict describing which patches were installed.
    """
    global _ACTIVE_DTYPE
    _ACTIVE_DTYPE = dtype

    installed = {"dtype": str(dtype), "patches": [], "compile_safe": compile_safe}

    # Global .float() bypass for the eager trunk; the compiled score model gets the
    # source-level patches below instead (the two are complementary).
    if apply_global_float_bypass and dtype == torch.bfloat16:
        _apply_global_float_bypass(dtype)
        installed["patches"].append("torch.Tensor.float -> bf16-safe (bool/int keep converting, bf16 stays bf16)")

    if compile_safe and dtype == torch.bfloat16:
        for p in _apply_score_model_dtype_patches(dtype):
            installed["patches"].append(p)

    # --- TriangleMultiplication ---
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
    )
    if os.environ.get("BOLTZ_TRIMUL_KCHUNK", "0") not in ("0", ""):
        # Split the contraction axis k into chunks and accumulate (same result; smaller
        # per-op graphs). Fallback for token counts whose single matmul does not compile.
        TriangleMultiplicationOutgoing.forward = _tri_mul_out_forward_kchunk
        TriangleMultiplicationIncoming.forward = _tri_mul_in_forward_kchunk
        installed["patches"].append(
            f"TriangleMultiplication{{Out,In}}going.forward -> K-CHUNKED contraction "
            f"(k_chunk={os.environ.get('BOLTZ_TRIMUL_KCHUNK')})")
    else:
        TriangleMultiplicationOutgoing.forward = _tri_mul_out_forward_clean
        installed["patches"].append("TriangleMultiplicationOutgoing.forward -> clean (no .float() cast)")
        TriangleMultiplicationIncoming.forward = _tri_mul_in_forward_clean
        installed["patches"].append("TriangleMultiplicationIncoming.forward -> clean (no .float() cast)")

    # --- AttentionPairBias v1 + v2 ---
    if apply_attention:
        try:
            from boltz.model.layers.attention import AttentionPairBias as _APB_V1
            _APB_V1.forward = _attention_pair_bias_v1_forward_clean
            installed["patches"].append("attention.AttentionPairBias.forward -> clean")
        except (ImportError, AttributeError):
            pass

        try:
            from boltz.model.layers.attentionv2 import AttentionPairBias as _APB_V2
            _APB_V2.forward = _attention_pair_bias_v2_forward_clean
            installed["patches"].append("attentionv2.AttentionPairBias.forward -> clean")
        except (ImportError, AttributeError):
            pass

    # --- PairformerLayer ---
    if apply_pairformer:
        from boltz.model.layers.pairformer import PairformerLayer
        PairformerLayer.forward = _pairformer_layer_forward_clean
        installed["patches"].append("PairformerLayer.forward -> clean (no autocast, no .float())")

    # --- OuterProductMean ---
    if apply_opm:
        try:
            from boltz.model.layers.outer_product_mean import OuterProductMean
            OuterProductMean.forward = _outer_product_mean_forward_clean
            installed["patches"].append("OuterProductMean.forward -> clean")
        except (ImportError, AttributeError):
            pass

    # --- encodersv2.single_to_keys + get_indexing_matrix (dtype safety fixes) ---
    if dtype == torch.bfloat16:
        try:
            import boltz.model.modules.encodersv2 as _encv2
            # In compile_safe mode, _apply_score_model_dtype_patches already installed the
            # force-target-dtype variant of single_to_keys; do not overwrite it.
            if not compile_safe:
                _encv2.single_to_keys = _single_to_keys_dtype_safe
                installed["patches"].append("encodersv2.single_to_keys -> dtype-safe (bf16 einsum fix)")
            _encv2.get_indexing_matrix = _make_get_indexing_matrix_dtype_aware(dtype)
            _encv2.FourierEmbedding.forward = _fourier_embedding_forward_clean
            installed["patches"].append(f"encodersv2.get_indexing_matrix -> dtype={dtype} (bf16 fix)")
            installed["patches"].append("encodersv2.FourierEmbedding.forward -> dtype-safe (bf16 fix)")
            # RelativePositionEncoder builds int one-hots then .float()s them into a full
            # [N,N,139] fp32 cat; the global bypass keeps genuine int->fp32 casts, so install
            # the dtype-safe forward here too (bit-exact: one-hots are exact in bf16).
            if not compile_safe:
                _encv2.RelativePositionEncoder.forward = _make_rel_pos_encoder_forward_dtype_safe(dtype)
                installed["patches"].append(
                    "encodersv2.RelativePositionEncoder.forward -> dtype-safe cat")
        except (ImportError, AttributeError):
            pass

        try:
            from boltz.model.modules.diffusionv2 import AtomDiffusion
            AtomDiffusion.preconditioned_network_forward = _make_preconditioned_forward_dtype_safe(dtype)
            installed["patches"].append(f"AtomDiffusion.preconditioned_network_forward -> cast to {dtype} at score_model boundary")
        except (ImportError, AttributeError):
            pass

        try:
            from boltz.model.modules.confidencev2 import ConfidenceModule
            _orig_conf_forward = ConfidenceModule.forward
            ConfidenceModule.forward = _make_confidence_forward_dtype_safe(dtype, _orig_conf_forward)
            installed["patches"].append(f"ConfidenceModule.forward -> cast x_pred to {dtype} at entry")
        except (ImportError, AttributeError):
            pass

    # --- Dropout mask (eval mode) ---
    _apply_dropout_patch()
    installed["patches"].append("get_dropout_mask -> eval-mode scalar")

    # --- Steering potentials (only used when fk_steering is on) ---
    # ORDER MATTERS: the VDW workaround REPLACES VDWOverlapPotential.compute_args, so it must
    # run BEFORE _apply_steering_potentials_dtype_patch, which WRAPS compute_args.
    if pair_pad_mask_gather_workaround:
        _vdw_patched = _apply_pair_pad_mask_gather_workaround()
        if _vdw_patched:
            installed["patches"].append(
                "VDWOverlapPotential.pair_pad_mask -> gather-free prefix compare "
                "(neuronx-cc remat_optimization hang workaround; bit-exact)")

    if steering_potentials_dtype_fix and dtype != torch.float32:
        _pot_patched = _apply_steering_potentials_dtype_patch()
        if _pot_patched:
            installed["patches"].append(
                f"steering potentials compute_args -> fp32-safe feats "
                f"({len(_pot_patched)} classes: {','.join(_pot_patched)})")

    # --- max_parallel_samples as a cap (opt-in) ---
    # Boltz 2.2.1 computes sample_ids.chunk(multiplicity % max_parallel_samples + 1), i.e. a chunk
    # COUNT from a modulus. With BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1 the count becomes
    # ceil(multiplicity / max_parallel_samples), so the value caps samples per chunk. Bit-exact
    # whenever ds <= max_parallel_samples.
    if os.environ.get("BOLTZ_FIX_MAX_PARALLEL_SAMPLES", "") == "1":
        try:
            import math as _math
            from boltz.model.modules.diffusionv2 import AtomDiffusion as _AD

            _orig_chunk = torch.Tensor.chunk

            def _make_fixed_sample(orig_sample):
                def _sample(self, *a, **kw):
                    mps = kw.get("max_parallel_samples", None)
                    mult = kw.get("multiplicity", None)
                    if mps is None or mult is None:
                        return orig_sample(self, *a, **kw)
                    _want = max(1, _math.ceil(mult / mps))
                    _shipped = mult % mps + 1

                    def _patched_chunk(t, n, dim=0):
                        if n == _shipped and t.numel() == mult:
                            return _orig_chunk(t, _want, dim)
                        return _orig_chunk(t, n, dim)

                    torch.Tensor.chunk = _patched_chunk
                    try:
                        return orig_sample(self, *a, **kw)
                    finally:
                        torch.Tensor.chunk = _orig_chunk

                return _sample

            _AD.sample = _make_fixed_sample(_AD.sample)
            installed["patches"].append(
                "AtomDiffusion.sample -> max_parallel_samples treated as a cap")
        except (ImportError, AttributeError) as e:
            installed["patches"].append(f"max_parallel_samples fix SKIPPED: {e}")

    # --- Skip the template module when the batch has no templates ---
    # Its output is exactly zero then (and it is a large all-zero cast); inputs with
    # templates are untouched.
    if skip_empty_templates:
        try:
            import boltz.model.modules.trunkv2 as _tv2
            for _cls in ("TemplateV2Module", "TemplateModule"):
                _c = getattr(_tv2, _cls, None)
                if _c is None:
                    continue
                _c.forward = _make_template_skip_when_empty(_c.forward)
                installed["patches"].append(
                    f"trunkv2.{_cls}.forward -> return zeros when template_mask.any() is False")
        except (ImportError, AttributeError) as e:
            installed["patches"].append(f"template skip SKIPPED: {e}")

    return installed


def _apply_global_float_bypass(target_dtype: torch.dtype):
    """In bf16 mode: patch torch.Tensor.float so that:
       - already-`target_dtype` tensors return self (no cast)
       - all other tensors convert to `target_dtype` (NOT fp32)

    This aggressively neutralizes ALL `.float()` sites in Boltz-2, routing them
    to bf16 in bf16 mode. Bool/int -> bf16 is still a valid conversion.
    The model weights are already bf16 (via model.to(bf16)), so this ensures
    inputs match weights in every Linear layer.
    """
    _original_float = torch.Tensor.float

    def _redirect_float(self):
        if self.dtype == target_dtype:
            return self
        return self.to(target_dtype)

    torch.Tensor.float = _redirect_float


def _apply_pair_pad_mask_gather_workaround():
    """Replace `atom_pad_mask[pair_index].all(dim=0)` with an equivalent compare.

    WHY: that indexing lowers to a `stablehlo.gather` fed by an
    `AwsNeuronCustomNativeKernel` custom_call, and on neuronx-cc 2.27.2878.0
    (Beta 5) the dataflow `custom_call -> gather` makes the backend pass
    `remat_optimization` **never terminate** -- it is a compiler defect, not resource exhaustion.

    Without this, `fk_steering=True` cannot compile at larger N: `VDWOverlapPotential`
    builds `triu_indices(A, A, 1)` over all atom pairs.

    THE REWRITE IS EXACT -- IT DOES NOT APPROXIMATE OR SKIP ANY PAIR.
    `atom_pad_mask` is a contiguous prefix of ones because
    `boltz/data/pad.py:pad_dim` pads right-only with `value=0`. Hence
    `mask[i] == (i < n_real)` and

        mask[pair_index].all(dim=0) == (idx0 < n_real) & (idx1 < n_real)

    All-pairs comparison is REQUIRED model behaviour: folding brings residues that
    are distant in sequence into physical contact, so a sequence-local or
    distance-cutoff approximation would change predictions. Every pair is still
    evaluated here; only the *gather* is removed.

    Verified bit-exact on 9 shape/padding combinations, plus an adversarial
    non-prefix mask that correctly differs (documenting the precondition).

    Effect on compile time for that subgraph: timeout -> 22 s.

    SAFETY: if the mask is ever NOT a prefix (should be impossible for a padding
    mask), we detect it and fall back to the original gather rather than silently
    computing something different.
    """
    import boltz.model.potentials.potentials as _pot
    from boltz.data import const as _const

    patched = []

    cls = getattr(_pot, "VDWOverlapPotential", None)
    if cls is None or "compute_args" not in cls.__dict__:
        return patched

    def compute_args(self, feats, parameters):
        # Re-implementation of boltz/model/potentials/potentials.py:438-495 with
        # TWO changes, both perf-only and numerically exact:
        #
        #  (1) The five 12.78M-element gathers off `pair_index` are computed on the
        #      HOST instead of on device, to dodge a neuronx-cc
        #      `remat_optimization` non-termination on `custom_call -> gather`.
        #      Without this, fk_steering cannot compile at all at N>=682.
        #
        #  (2) That host-side work is CACHED across sampling steps. `compute()` is
        #      called from inside the diffusion loop (diffusionv2.py:398-414) every
        #      `fk_resampling_interval`=3 steps -> ~18x per structure at s=50. But
        #      `pair_index` and all three masks depend ONLY on `feats` (topology),
        #      never on `atom_coords_denoised` and never on t. Uncached we rebuilt
        #      triu_indices(5056,5056,1) -- a 204 MB CPU allocation -- and redid
        #      four 12.78M-element gathers on every call (~3.7 GB of redundant
        #      allocation per structure).
        #
        # Only `parameters["buffer"]` is t-dependent (it can be a ParameterSchedule),
        # and it is applied AFTER the cached part, so caching the index/mask
        # computation is safe.
        #
        # ALL PAIRS ARE STILL ENUMERATED AND EVALUATED -- no cutoff, no sampling.

        # Cache key: identity + shape of the feats tensors that feed the
        # computation. A new batch gets new tensor objects -> cache miss.
        key = (id(feats["atom_pad_mask"]), tuple(feats["atom_pad_mask"].shape),
               id(feats["atom_to_token"]), tuple(feats["atom_to_token"].shape),
               id(feats["asym_id"]), id(feats["connected_chain_index"]))
        cached = getattr(self, "_oc_pair_cache", None)

        if cached is not None and cached[0] == key:
            pair_index, atom_vdw_radii = cached[1], cached[2]
        else:
            atom_chain_id = (
                torch.bmm(
                    feats["atom_to_token"].to(torch.float32),
                    feats["asym_id"].unsqueeze(-1).to(torch.float32),
                ).squeeze(-1).long()
            )[0]
            atom_pad_mask = feats["atom_pad_mask"][0].bool()
            chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
            single_ion_mask = (chain_sizes > 1)[atom_chain_id]

            vdw_radii = torch.zeros(_const.num_elements, dtype=torch.float32,
                                    device=atom_chain_id.device)
            vdw_radii[1:119] = torch.tensor(_const.vdw_radii, dtype=torch.float32,
                                            device=atom_chain_id.device)
            atom_vdw_radii = (
                feats["ref_element"].to(torch.float32) @ vdw_radii.unsqueeze(-1)
            ).squeeze(-1)[0]

            dev = atom_chain_id.device

            # All-pairs index on the HOST. 12.78M x 2 int64 = 204 MB on CPU; the
            # filtered result that reaches the device is ~1000x smaller.
            pi_c = torch.triu_indices(atom_chain_id.shape[0],
                                      atom_chain_id.shape[0], 1, device="cpu")
            pad_c = atom_pad_mask.cpu()
            ion_c = single_ion_mask.cpu()
            chain_c = atom_chain_id.cpu()

            # atom_pad_mask is a contiguous prefix of ones (boltz/data/pad.py
            # pad_dim pads right-only with value=0), so
            #   mask[pair_index].all(dim=0) == (idx0 < n_real) & (idx1 < n_real)
            # Fall back to the real gather if that precondition ever fails.
            n_total = int(pad_c.numel())
            n_real = int(pad_c.sum().item())
            if n_real != n_total and bool(pad_c[n_real:].any().item()):
                pair_pad_mask = pad_c[pi_c].all(dim=0)
            elif n_real == n_total:
                pair_pad_mask = torch.ones(pi_c.shape[1], dtype=torch.bool)
            else:
                pair_pad_mask = (pi_c[0] < n_real) & (pi_c[1] < n_real)

            pair_ion_mask = ion_c[pi_c[0]] * ion_c[pi_c[1]]

            num_chains = int(chain_c.max().item()) + 1
            cci = feats["connected_chain_index"][0].cpu()
            connected_chain_matrix = torch.eye(num_chains, dtype=torch.bool)
            connected_chain_matrix[cci[0], cci[1]] = True
            connected_chain_matrix[cci[1], cci[0]] = True
            connected_chain_mask = connected_chain_matrix[chain_c[pi_c[0]],
                                                          chain_c[pi_c[1]]]

            keep = pair_pad_mask * pair_ion_mask * ~connected_chain_mask
            pair_index = pi_c[:, keep].to(dev)

            self._oc_pair_cache = (key, pair_index, atom_vdw_radii)

        # t-dependent part -- recomputed every call, cheap (operates on the
        # FILTERED pair set, typically ~1000x smaller than all-pairs).
        lower_bounds = atom_vdw_radii[pair_index].sum(dim=0) * (
            1.0 - parameters["buffer"])
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None

    cls.compute_args = compute_args
    patched.append("VDWOverlapPotential.compute_args")
    return patched


def _apply_steering_potentials_dtype_patch():
    """Make the steering potentials dtype-safe under the bf16 .float() bypass.

    Required whenever ANY steering flag is on in bf16 mode (fk_steering,
    physical_guidance_update, contact_guidance_update).

    Root cause: `_apply_global_float_bypass` rewrites `torch.Tensor.float` to a
    no-op for the active dtype. Several potentials do

        feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)

    where `vdw_radii` is built with an EXPLICIT `dtype=torch.float32`
    (boltz/model/potentials/potentials.py:407-415 and 451-458). With the bypass
    installed the left operand stays bf16 while the right stays fp32, and the
    matmul fails to lower:

        Failed Torch-MLIR Lowering Torch Backend IR -> StableHLO IR:
        error: matmul: input datatypes mismatched
        error: failed to legalize operation 'torch.aten.mm'

    Same hazard at potentials.py:610-629 (`torch.bmm` of `token_to_rep_atom` /
    `atom_to_token` against explicitly-fp32 arange / `token_index`).

    Fix: wrap the two `compute_args` methods that own these matmuls and force both
    operands to a single dtype. We standardize on fp32 here because the constants
    (`const.vdw_radii`, index arithmetic) are physical/index quantities where fp32
    is both cheap (tiny tensors) and safer -- these are not on the perf-critical
    score-model path.

    Without this patch, --fk_steering cannot compile at all on Neuron.
    """
    import boltz.model.potentials.potentials as _pot

    patched = []

    # WHY a save/restore of torch.Tensor.float rather than a feats pre-cast:
    #
    #   vdw_radii = torch.zeros(..., dtype=torch.float32)           # explicit fp32
    #   atom_vdw_radii = feats["ref_element"].float() @ vdw_radii    # <- mismatch
    #
    # `vdw_radii` is built INSIDE compute_args, after any wrapper we install, so
    # we cannot pre-cast it. And pre-casting `feats["ref_element"]` to fp32 does
    # not help either: `_apply_global_float_bypass` defines
    #     .float() -> self if self.dtype == target else self.to(target)
    # so an fp32 tensor is actively converted fp32 -> bf16 by the very next
    # `.float()` call. The pre-cast is undone.
    #
    # The only robust fix is to give the potentials their ORIGINAL fp32 `.float()`
    # semantics for the duration of the call. These potentials operate on small
    # index / physical-constant tensors well off the perf-critical score-model
    # path, so fp32 there is effectively free.
    #
    # Verified by a direct lowering test: matched-dtype
    # operands lower fine; any mismatch fails with
    #   "matmul: input datatypes mismatched / failed to legalize torch.aten.mm".
    def _wrap_compute_args(cls):
        # Only wrap classes that define compute_args THEMSELVES; wrapping an
        # inherited attribute would double-wrap via the subclass.
        if "compute_args" not in cls.__dict__:
            return False
        orig = cls.__dict__["compute_args"]

        def compute_args(self, feats, parameters, *a, **kw):
            saved = torch.Tensor.float
            torch.Tensor.float = _ORIGINAL_TENSOR_FLOAT
            try:
                out = orig(self, feats, parameters, *a, **kw)
            finally:
                torch.Tensor.float = saved
            # Second upstream hazard, independent of dtype: Potential.compute
            # does `if index.shape[1] == 0` (potentials.py:29), but several
            # compute_args return a 1-D `index` when the corresponding feature
            # is absent (e.g. ContactPotentital returns
            # feats["contact_pair_index"][0], which is 1-D/empty for inputs with
            # no contact constraints). That raises
            #     IndexError: tuple index out of range
            # before any device work happens. Normalize to 2-D so compute() can
            # take its intended empty-index early return.
            try:
                idx = out[0]
                if isinstance(idx, torch.Tensor) and idx.dim() < 2:
                    out = (idx.reshape(1, -1),) + tuple(out[1:])
            except (TypeError, IndexError):
                pass
            return out

        cls.compute_args = compute_args
        return True

    for name in ("PoseBustersPotential", "ConnectionsPotential",
                 "VDWOverlapPotential", "SymmetricChainCOMPotential",
                 "StereoBondPotential", "ChiralAtomPotential",
                 "PlanarBondPotential", "TemplateReferencePotential",
                 "ContactPotentital"):
        cls = getattr(_pot, name, None)
        if cls is not None and _wrap_compute_args(cls):
            patched.append(name)

    return patched


def _apply_dropout_patch():
    """Mandatory monkey-patch: eval-mode dropout returns scalar 1.0."""
    import boltz.model.layers.dropout as _boltz_dropout

    def _eval_dropout_mask(dropout, z, training, columnwise=False):
        return torch.tensor(1.0, dtype=z.dtype, device=z.device)

    _boltz_dropout.get_dropout_mask = _eval_dropout_mask
    # Propagate through the modules that already imported the symbol
    import boltz.model.layers.pairformer as _p
    _p.get_dropout_mask = _eval_dropout_mask
    try:
        import boltz.model.modules.trunkv2 as _tv2
        _tv2.get_dropout_mask = _eval_dropout_mask
    except (ImportError, AttributeError):
        pass
    try:
        import boltz.model.modules.trunk as _tv1
        _tv1.get_dropout_mask = _eval_dropout_mask
    except (ImportError, AttributeError):
        pass


def _make_template_skip_when_empty(orig_forward):
    """Return zeros from TemplateV2Module.forward when the batch has no templates.

    With no templates the module spends most of its time on a large all-zero [B,T,N,N] cast
    and returns exactly zero: (v * template_mask).sum() is 0, relu(0) is 0, and u_proj has no
    bias, so the output is 0 and Boltz2.forward adds it to z. Returning zeros is therefore
    bit-exact. The check is per call, so inputs with templates take the normal path.
    Output shape/dtype match z: [B, N, N, token_z].
    """
    def _forward(self, z, feats, pair_mask, use_kernels: bool = False):
        tm = feats.get("template_mask", None)
        if tm is not None:
            try:
                # .any() on a device tensor costs one small sync; that is ~microseconds
                # against the 1.4-11.8 s this skips. Guard is cheap relative to payoff.
                if not bool(tm.any()):
                    return torch.zeros_like(z)
            except Exception:
                pass   # never let the guard break the model; fall through to stock
        return orig_forward(self, z, feats, pair_mask, use_kernels=use_kernels)
    return _forward


def apply_load_patches():
    """Force weights_only=False on torch.load and allowlist omegaconf types.

    Required because Boltz-2 checkpoints contain omegaconf.DictConfig objects
    that are not in the default weights_only=True allowlist in PyTorch >= 2.6.
    """
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load

    try:
        import omegaconf
        torch.serialization.add_safe_globals([
            omegaconf.dictconfig.DictConfig,
            omegaconf.listconfig.ListConfig,
        ])
    except ImportError:
        pass

    # -------------------------------------------------------------------------
    # The Boltz CDN serves a checkpoint NEWER than the pinned
    # boltz==2.2.1 package. Its `diffusion_process_args` carries exactly one key
    # that v2.2.1's AtomDiffusion.__init__ does not accept:
    #     mse_rotational_alignment=True
    # which makes load_from_checkpoint die with
    #     TypeError: AtomDiffusion.__init__() got an unexpected keyword argument
    # on a FRESH install -- i.e. the documented install path is broken as of today,
    # independent of Beta 5.
    #
    # It is a TRAINING-only setting (rotational alignment inside the MSE loss) with
    # no inference effect, and it is the ONLY unaccepted key (verified by diffing the
    # checkpoint dict against the AtomDiffusion __init__ signature). Drop unknown
    # kwargs rather than pinning to an older checkpoint, so the harness keeps working
    # as upstream adds further training-only knobs -- but PRINT what was dropped, so a
    # future genuinely-inference-relevant key cannot disappear silently.
    # -------------------------------------------------------------------------
    try:
        import inspect
        from boltz.model.modules.diffusionv2 import AtomDiffusion

        if not getattr(AtomDiffusion, "_fn_kwarg_filter", False):
            _accepted = set(inspect.signature(AtomDiffusion.__init__).parameters)
            _orig_init = AtomDiffusion.__init__

            def _filtered_init(self, *a, **kw):
                dropped = sorted(k for k in kw if k not in _accepted)
                for k in dropped:
                    kw.pop(k)
                if dropped:
                    print(f"[patches] dropped checkpoint kwargs not in boltz "
                          f"AtomDiffusion signature (training-only): {dropped}")
                return _orig_init(self, *a, **kw)

            AtomDiffusion.__init__ = _filtered_init
            AtomDiffusion._fn_kwarg_filter = True
    except ImportError:
        pass
