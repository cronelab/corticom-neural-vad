import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


logger = logging.getLogger("plot_context_analysis.py")


def main(context_dir: Path, out_dir: Optional[Path] = None):
    results = sorted(context_dir.glob("*.npy"))
    results = [np.load(r) for r in results]

    while len(results) < 7:
        results.append(np.array([]))

    fig = plt.figure(figsize=(500 / 96, 400 / 96))
    ax = fig.add_subplot(111)

    outlier_props = dict(markerfacecolor="gray", linestyle="none")
    bp = ax.boxplot(x=results, labels=["0", "-50", "-100", "-150", "-200", "-250", "-300"],
                    positions=np.arange(len(results)), widths=0.4,
                    patch_artist=True, flierprops=outlier_props)

    for medians in bp["medians"]:
        medians.set_color("black")
        medians.set_linewidth(2)

    for boxes in bp["boxes"]:
        boxes.set_facecolor("lightblue")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.set_ylim(0)
    ax.set_xlabel("Temporal Context [ms]")
    ax.set_ylabel("nVAD alignment error [s]")

    if out_dir is not None:
        plt.savefig(out_dir / "plot_context_analysis.png", dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render temporal context figure")
    parser.add_argument("con", help="Path to results from the temporal context analysis.")
    parser.add_argument("-o", "--out", help="Path to the output folder.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f"python plot_temporal_context.py {args.con} " + args.out if args.out else "")
    main(Path(args.con), out_dir=Path(args.out) if args.out else None)
