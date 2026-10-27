"""Bit-exactness of the static-shape FlatBottomPotential.compute_function vs stock boltz 2.2.1 (CPU).

Run: python tests/test_flat_bottom_static.py   (needs torch + boltz importable; or BOLTZ_SRC=<boltz src dir>)
"""
import importlib, os, sys, types
import torch

if os.environ.get("BOLTZ_SRC"):
    sys.path.insert(0, os.environ["BOLTZ_SRC"])
P = importlib.import_module("boltz.model.potentials.potentials")
stock = P.FlatBottomPotential.compute_function
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boltz_neuron import patches  # noqa: E402

assert patches._apply_flat_bottom_static_shape_patch()
new = P.FlatBottomPotential.compute_function
assert new is not stock


class Dummy:  # compute_function does not use self
    pass


def case(shape, with_lb, with_ub, neg, k_scalar, seed):
    g = torch.Generator().manual_seed(seed)
    value = torch.rand(shape, generator=g) * 4
    lb = (torch.rand(shape[-1:], generator=g) * 2) if with_lb else None
    ub = (torch.rand(shape[-1:], generator=g) * 2 + 2) if with_ub else None
    k = torch.tensor(0.7) if k_scalar else torch.rand(shape[-1:], generator=g) + 0.5
    nm = None
    if neg:
        # stock asserts every entry is unbounded on one side or negated
        if lb is not None and ub is not None:
            lb = lb.clone(); ub = ub.clone()
            half = torch.rand(shape[-1:], generator=g) < 0.5
            lb[half] = float("-inf")
            ub[~half] = float("inf")
        nm = torch.rand(shape[-1:], generator=g) < 0.5
    out = []
    for fn in (stock, new):
        r = fn(Dummy(), value.clone(), k, None if lb is None else lb.clone(), None if ub is None else ub.clone(),
               negation_mask=None if nm is None else nm.clone(), compute_derivative=True)
        out.append(r)
    for a, b in zip(out[0], out[1]):
        assert a.shape == b.shape and torch.equal(a, b), (shape, with_lb, with_ub, neg, k_scalar, (a - b).abs().max())


n = 0
for shape in ((3, 1000), (1, 7), (2, 3, 50)):
    for with_lb in (True, False):
        for with_ub in (True, False):
            for neg in (False, True):
                if neg and not (with_lb and with_ub):
                    continue
                for k_scalar in (True, False):
                    for seed in range(3):
                        case(shape, with_lb, with_ub, neg, k_scalar, seed); n += 1
print(f"OK: {n} cases bit-exact (energy and dEnergy)")
