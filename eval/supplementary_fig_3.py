import argparse
import logging
import os
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from scipy.io import loadmat
from datetime import datetime as dt, timedelta
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict


logger = logging.getLogger("supplementary_fig_3.py")


def extract_data(folders: List[Path], data: Dict[str, List], is_syll: bool = False):
    """
    Populate the data dictionary with the dates and timestamps of the recording session.
    Syllable repetition will always be in the beginning.
    """
    for day in folders:
        runs = sorted(day.glob("*.mat"))
        for run in runs:
            if is_syll and run.parent.name not in data.keys():
                continue

            mat = loadmat(run.as_posix(), simplify_cells=True)
            start = dt.strptime(mat["parameters"]["StorageTime"]["Value"], "%Y-%m-%dT%H:%M:%S")
            stop = start + timedelta(0, len(mat["signal"]) / mat["parameters"]["SamplingRate"]["NumericValue"])

            if is_syll:
                data[start.date().strftime("%Y_%m_%d")].insert(0, (start, stop, run.stem))
            else:
                data[start.date().strftime("%Y_%m_%d")].append((start, stop, run.stem))


def render_plot(data: Dict[str, List], out_dir: Optional[Path] = None):
    """
    Render the plot for supplementary figure 3
    """
    days = sorted(data.keys())
    fig = plt.figure(figsize=(1100 / 96, 500 / 96))
    ax = fig.add_subplot(111)

    # Configure plot layout
    ys = np.arange(len(data.keys()))
    ax.set_yticks(ys)
    ax.set_yticklabels([f"$D_{i}$" for i in ys])
    ax.set_ylim(-1, len(ys))
    ax.invert_yaxis()

    xs = np.arange(0, 5)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Time [h]")
    ax.set_xlim(0, 4.0)

    for i, day in enumerate(days):
        ax.axhline(i, c="black", linewidth=0.85, linestyle="--", zorder=0)

        # Add syllable repetition block
        start, stop, _ = data[day][0]
        ax.add_patch(plt.Rectangle((0, i - 0.4), (stop - start).seconds / 3600, 0.8,
                                   facecolor="lemonchiffon", edgecolor="black", linewidth=0.85))

        session_start = start
        for sess_idx in range(1, len(data[day])):
            start, stop, _ = data[day][sess_idx]

            ax.add_patch(plt.Rectangle(((start - session_start).seconds / 3600, i - 0.4),
                                       (stop - start).seconds / 3600, 0.8,
                                       facecolor="plum", edgecolor="black", linewidth=0.85))

    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(out_dir / "supplementary_fig_3.png", dpi=150)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render supplementary figure 3")
    parser.add_argument("data", help="Path to the 50word experiment recordings.")
    parser.add_argument("syll", help="Path to the syllable repetition experiment recordings.")
    parser.add_argument("-o", "--out", help="Path to the output folder.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f"python supplementary_fig_3.py {args.data} {args.syll} " + args.out if args.out else "")
    data = defaultdict(list)

    data_dirs = [d for d in Path(args.data).iterdir() if d.is_dir()]
    syll_dirs = [d for d in Path(args.syll).iterdir() if d.is_dir()]

    # First insert info from syllable repetition data
    extract_data(data_dirs, data)

    # Second, append info from 50 Word recordings
    extract_data(syll_dirs, data, is_syll=True)

    render_plot(data, out_dir=Path(args.out) if args.out else None)
