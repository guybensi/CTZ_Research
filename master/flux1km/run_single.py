"""Run one method (or a few) through the full benchmark protocol from its own module.

Every method module ends with

    if __name__ == "__main__":
        from ..run_single import main_for
        main_for(["M3"])

so ``python3 -m flux1km.methods.m3_pixel_mlp --demo`` runs M3 with the reference baselines
(B-clim, B-const and M0-ridge) on the frozen protocol and writes the plain-text results file.
All other command-line options are those of flux1km.cli.
"""
from __future__ import annotations

import sys

REFERENCE_BASELINES = ("B-clim", "B-const", "M0-ridge")


def single_args(models: list[str], argv: list[str] | None = None, baselines: bool = True) -> list[str]:
    """Build the flux1km.cli argument list for the given models.

    The user's own --models wins. Unless --no-baselines is given, the reference baselines are added.

    Example:
        >>> single_args(["M3"], ["--demo"])
        ['--demo', '--models', 'M3', 'B-clim', 'B-const', 'M0-ridge']
        >>> single_args(["M3"], ["--demo", "--no-baselines"])
        ['--demo', '--models', 'M3']
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--no-baselines" in argv:
        argv.remove("--no-baselines")
        baselines = False
    if "--models" in argv:
        return argv
    chosen = list(models) + ([b for b in REFERENCE_BASELINES if b not in models] if baselines else [])
    return [*argv, "--models", *chosen]


def main_for(models: list[str], argv: list[str] | None = None) -> dict:
    """Run flux1km.cli.main for `models` (+ baselines) and return its result."""
    from . import cli

    return cli.main(single_args(models, argv))
