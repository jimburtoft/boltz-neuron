"""K-24: triangle attention (Boltz-2 Algorithms 13/14) with an NKI attention core, for one logical NeuronCore.

Enable with BOLTZ_NEURON_TRI_ATTN=k24 (see patches.install_k24_tri_attn). Inference only, batch 1.

Math (identical to boltz/model/layers/triangular_attention, starting node; the ending node runs the same on
the transposed pair representation):
    x = LayerNorm(z)                               [I, J, C]
    tri[h, q, k] = (x @ W_b)[q, k, h]              shared over rows i
    q, k, v, g = x @ W_{q,k,v,g}                   [I, H, J, D]
    o[i, h] = softmax(q[i,h] k[i,h]^T / sqrt(D) + tri[h] + 1e9 * (mask[i, :] - 1)) v[i, h]
    out = (sigmoid(g) * o) @ W_o

The softmax-attention core runs in a fork of nkilib attention_cte (k24nkilib, vendored) with two changes:
  * bias_heads=H: position_bias is the shared per-head [H, J, J] bias, kept resident in SBUF per head
    (batch order is head-major), instead of a dense per-batch [B, J, J] tensor.
  * the bias enters PSUM through an identity matmul (PSUM initialised by a matmul, then QK accumulates):
    stock nkilib preloads the bias with a Vector/Scalar copy and accumulates onto it, which the nc_matmul docs
    call undefined on NeuronCore-v3 and which gave state-dependent wrong first-Q-group results on trn2.
Mask handling: Boltz's pair mask is the outer product of the token mask, so for real rows the key mask is the
same for every row and folds into the per-head bias; padded rows reproduce stock's fp32 behaviour (all keys
at -1e9 -> uniform average of v) with a torch reduction.

The projections run in torch; the whole module is one torch.compile(backend="neuron") graph per (N, starting)
with the layer weights passed as inputs, so all 64 trunk layers share it.
"""
import math
import os

import torch

LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
I_CHUNK = int(os.environ.get("BOLTZ_NEURON_K24_I_CHUNK", "64"))  # rows i per kernel call
# Bias precision. "bf16" (default): one bf16 copy of the bias. "hilo": the bias is split into two bf16 parts (hi + lo)
# that both enter the fp32 PSUM, so it reaches the scores at ~fp32 precision. Measured on trn2 (2026-11-09): hilo cuts
# the attention-core error 4.5x (rel 0.0081 -> 0.0018 vs fp64), but the module error only 2-5% (0.0195 -> 0.0191:
# the bf16 LayerNorm/projections/inputs dominate), crystal RMSD is unchanged, and it costs 7-12% per module (7% E2E).
BIAS_PREC = os.environ.get("BOLTZ_NEURON_K24_BIAS", "bf16").lower()
SPLIT = os.environ.get("BOLTZ_NEURON_K24_SPLIT", "veltkamp").lower()

_cte = None
_compiled = {}
CALLS = {"k24": 0, "fallback": 0}


def _kernel():
    global _cte
    if _cte is None:
        from torch_neuronx import wrap_nki
        from boltz_neuron.k24nkilib.core.attention.attention_cte import attention_cte
        _cte = wrap_nki(attention_cte)
    return _cte


def _core(q, k, v, bias, scale):
    """q, k, v: [I, H, J, D] (q unscaled); bias [H, J, J] fp32 incl. key mask. -> o [I, H, J, D]."""
    I, H, J, D = q.shape
    cte = _kernel()[LNC]
    pb = (bias / scale).to(torch.float32)               # kernel computes scale * (QK + bias)
    kw = {}
    if BIAS_PREC == "hilo":
        # hi = exactly bf16-representable part of the fp32 bias; lo = the remainder (stored bf16 by the kernel).
        # NOT a bf16 round trip: neuronx-cc folds fp32->bf16->fp32 into a no-op inside a compiled graph, which
        # silently made lo == 0 (measured).
        pb = pb.contiguous()
        if SPLIT == "bitmask":
            hi = (pb.view(torch.int32) & -65536).view(torch.float32)
        else:   # Veltkamp split (C = 2^16 + 1): plain fp32 arithmetic, hi keeps the top 8 mantissa bits (bf16-exact)
            c = pb * 65537.0
            hi = c - (c - pb)
        kw["position_bias_lo"] = (pb - hi).contiguous()
        pb = hi
    pb = pb.contiguous()
    outs = []
    for i0 in range(0, I, I_CHUNK):
        i1 = min(I, i0 + I_CHUNK)
        qb = q[i0:i1].transpose(0, 1).reshape(-1, J, D).contiguous()                 # head-major batch
        kb = k[i0:i1].transpose(0, 1).reshape(-1, J, D).transpose(1, 2).contiguous()
        vb = v[i0:i1].transpose(0, 1).reshape(-1, J, D).contiguous()
        o = cte(qb, kb, vb, scale=scale, causal_mask=False, tp_q=True, tp_k=False, position_bias=pb, bias_heads=H, **kw)
        outs.append(o.reshape(H, i1 - i0, J, D).transpose(0, 1))
    return torch.cat(outs, 0) if len(outs) > 1 else outs[0]


def _forward_fn(x, mask, ln_w, ln_b, w_tri, w_q, w_k, w_v, w_g, w_o, starting: bool, inf: float, H: int, D: int):
    """Pure function of the inputs and weights (so one compiled graph serves every layer)."""
    if not starting:
        x = x.transpose(-2, -3)
        mask = mask.transpose(-1, -2)
    x = torch.nn.functional.layer_norm(x, (x.shape[-1],), ln_w.to(x.dtype), ln_b.to(x.dtype), 1e-5)
    x0 = x[0]
    I, J = x0.shape[0], x0.shape[1]
    tri = torch.nn.functional.linear(x0, w_tri).permute(2, 0, 1)        # [H, J, J] (bf16, as stock)
    m = mask[0]
    m_tok = m.amax(0)
    real_rows = m.amax(1) > 0
    bias = tri.float() + (inf * (m_tok.float() - 1.0))[None, None, :]
    q = torch.nn.functional.linear(x0, w_q).view(I, J, H, D).permute(0, 2, 1, 3)
    k = torch.nn.functional.linear(x0, w_k).view(I, J, H, D).permute(0, 2, 1, 3)
    v = torch.nn.functional.linear(x0, w_v).view(I, J, H, D).permute(0, 2, 1, 3)
    o = _core(q, k, v, bias, 1.0 / math.sqrt(D))
    vmean = v.float().mean(dim=2, keepdim=True).to(o.dtype).expand_as(v)
    o = torch.where(real_rows[:, None, None, None], o, vmean)
    o = o.permute(0, 2, 1, 3)
    g = torch.sigmoid(torch.nn.functional.linear(x0, w_g)).view(I, J, H, D)
    o = (o.to(g.dtype) * g).reshape(I, J, H * D)
    out = torch.nn.functional.linear(o, w_o)[None]
    if not starting:
        out = out.transpose(-2, -3)
    return out


_outer_cache = {}


def _is_outer_product(mask):
    """True if mask[0] == m_row[:, None] * m_col[None, :] for 0/1 vectors (Boltz's token pair mask). Cached per
    tensor (all trunk/MSA/template/confidence layers share one mask tensor), so the device->host check runs once."""
    # Keyed on the tensor object (kept alive by the cache, so its id cannot be reused). Masks are not modified in
    # place in Boltz inference, and inference tensors have no version counter.
    key = id(mask)
    r = _outer_cache.get(key)
    if r is None or r[1] is not mask:
        m = mask[0].float()
        r = bool(torch.equal(m, m.amax(1)[:, None] * m.amax(0)[None, :]))
        # keep the tensor alive with its entry so its storage (and data_ptr) cannot be reused for another mask
        r = (r, mask)
        if len(_outer_cache) > 64:
            _outer_cache.clear()
        _outer_cache[key] = r
    return r[0]


def forward(self, x, mask=None, chunk_size=None, use_kernels=False):
    """Drop-in for TriangleAttention.forward on the neuron device (batch 1)."""
    if x.device.type != "neuron" or x.shape[0] != 1 or self.mha.linear_g is None:
        CALLS["fallback"] += 1
        if CALLS["fallback"] == 1:
            print(f"[boltz-neuron] K-24 tri-attn: stock fallback (device={x.device.type}, batch={x.shape[0]})", flush=True)
        return self._k24_stock_forward(x, mask=mask, chunk_size=chunk_size, use_kernels=use_kernels)
    if mask is None:
        mask = x.new_ones(x.shape[:-1])
    elif not _is_outer_product(mask):
        # e.g. the affinity module's receptor/ligand cross mask: the key mask depends on the row, so it cannot be
        # folded into the shared per-head bias. Use stock attention for those modules.
        CALLS["fallback"] += 1
        if CALLS["fallback"] == 1:
            print("[boltz-neuron] K-24 tri-attn: stock fallback (pair mask is not an outer product)", flush=True)
        return self._k24_stock_forward(x, mask=mask, chunk_size=chunk_size, use_kernels=use_kernels)
    CALLS["k24"] += 1
    key = (bool(self.starting), tuple(x.shape), x.dtype)
    fn = _compiled.get(key)
    if fn is None:
        fn = torch.compile(_forward_fn, backend="neuron", fullgraph=True, dynamic=False)
        _compiled[key] = fn
        print(f"[boltz-neuron] K-24 tri-attn: compiling graph starting={key[0]} shape={key[1]}", flush=True)
    mha = self.mha
    return fn(x, mask.to(x.dtype), self.layer_norm.weight, self.layer_norm.bias, self.linear.weight,
              mha.linear_q.weight, mha.linear_k.weight, mha.linear_v.weight, mha.linear_g.weight, mha.linear_o.weight,
              bool(self.starting), float(self.inf), int(mha.no_heads), int(mha.c_hidden))
