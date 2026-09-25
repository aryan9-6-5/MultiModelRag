"""CLI: python -m app.evaluate [--ablation]

Runs the retrieval benchmark (and optionally the full ablation study, which adds
a generation pass) against the COCO subset. Run
`python -m evaluation.datasets.download_coco_subset` first if the subset doesn't
exist yet."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true", help="Also run the generation ablation (Experiment C)")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    args = parser.parse_args()

    if args.ablation:
        from evaluation.ablation import run

        run()
    else:
        from evaluation.benchmarks.run_benchmark import run

        run(tuple(args.k))


if __name__ == "__main__":
    main()
