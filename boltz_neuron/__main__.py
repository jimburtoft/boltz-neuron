"""
boltz-neuron: drop-in `boltz predict` for AWS Inferentia2 via PyTorch Native.

Usage (identical to stock boltz, just the module name changes):
    python -m boltz_neuron predict input.yaml --out_dir out --recycling_steps 3 --sampling_steps 50 --num_workers 0

The validated configuration (bf16, compiled diffusion score model, adaptive padding, caches,
fk_steering off, Inferentia2 fixes) is applied automatically before boltz runs. See preamble.py.
"""
import sys


def main():
    from boltz_neuron import preamble
    preamble.apply()
    # Hand off to the stock Boltz CLI. sys.argv[0] is this module; drop it so click sees
    # `predict ...` as if invoked as `boltz`.
    from boltz.main import cli
    cli()


if __name__ == "__main__":
    main()
