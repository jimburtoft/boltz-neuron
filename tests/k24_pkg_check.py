"""Packaged K-24 path (boltz_neuron.k24_tri_attn) vs stock, real layer weights, one core, with/without padding.
  python pkg_check.py --n 384 544 --layers 0 31 63 --pad 0 20"""
import argparse, json, os, statistics as st, sys, time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[384]); ap.add_argument("--layers", type=int, nargs="+", default=[0])
    ap.add_argument("--pad", type=int, nargs="+", default=[0]); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    from boltz_neuron import preamble
    preamble.apply()
    from boltz.model.layers.triangular_attention.attention import TriangleAttentionStartingNode, TriangleAttentionEndingNode
    from boltz_neuron import k24_tri_attn
    ck = torch.load(os.path.expanduser("~/.boltz/boltz2_conf.ckpt"), map_location="cpu", weights_only=False)["state_dict"]
    dev = torch.device("neuron")
    for L in a.layers:
        for which, cls in [("start", TriangleAttentionStartingNode), ("end", TriangleAttentionEndingNode)]:
            pre = f"pairformer_module.layers.{L}.tri_att_{which}."
            sd = {k[len(pre):]: v for k, v in ck.items() if k.startswith(pre)}
            m = cls(c_in=128, c_hidden=32, no_heads=4, inf=1e9); m.load_state_dict(sd, strict=True); m.eval()
            md = cls(c_in=128, c_hidden=32, no_heads=4, inf=1e9); md.load_state_dict(sd, strict=True); md = md.eval().bfloat16().to(dev)
            for n in a.n:
                for npad in a.pad:
                    g = torch.Generator().manual_seed(L * 7 + n)
                    z = torch.randn(1, n, n, 128, generator=g)
                    mask = torch.ones(1, n, n)
                    if npad:
                        mask[:, n - npad:] = 0; mask[:, :, n - npad:] = 0
                    with torch.no_grad():
                        ref = m(z, mask=mask, chunk_size=None).double()
                        zd, mdk = z.bfloat16().to(dev), mask.bfloat16().to(dev)
                        stock = md._k24_stock_forward(zd, mask=mdk, chunk_size=128 if n > 384 else 512) if hasattr(md, "_k24_stock_forward") else md(zd, mask=mdk)
                        out = k24_tri_attn.forward(md, zd, mask=mdk)
                        torch.neuron.synchronize()
                        ts = []
                        for _ in range(a.reps):
                            torch.neuron.synchronize(); t = time.perf_counter(); k24_tri_attn.forward(md, zd, mask=mdk); torch.neuron.synchronize()
                            ts.append(time.perf_counter() - t)
                    out = out.cpu().double(); stock = stock.cpu().double()
                    nr = n - npad
                    def rel(x, sl=None):
                        x, r = (x, ref) if sl is None else (x[0, :sl, :sl], ref[0, :sl, :sl])
                        return round(((x - r).norm() / r.norm()).item(), 5)
                    print("PKG " + json.dumps({"layer": L, "which": which, "n": n, "pad": npad,
                                               "k24_rel": rel(out), "stock_bf16_rel": rel(stock),
                                               "k24_rel_real": rel(out, nr), "stock_rel_real": rel(stock, nr),
                                               "k24_ms": round(1000 * st.median(ts), 2)}), flush=True)


if __name__ == "__main__":
    main()
