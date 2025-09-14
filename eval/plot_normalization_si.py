import argparse
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from pathlib import Path
from typing import Optional


logger = logging.getLogger("plot_normalization_si.py")


def render_plot(syllable_root: Path, out_dir: Optional[Path] = None):
    days = [f.stem for f in syllable_root.glob("*.npy")]

    normalization_mean = []
    normalization_stdd = []
    normalization_days = []
    normalization_chan = []

    # Drop bad channels
    channel_mask = np.ones(128, dtype=bool)
    channel_mask[np.array([19, 38, 48, 52])-1] = False

    # Aggregate data
    for day in days:
        filename = syllable_root / f"{day}.npy"
        norm_stats = np.load(filename)
        normalization_days.extend([day, ] * (norm_stats.shape[1] - 4))
        normalization_mean.extend(norm_stats[0, channel_mask])
        normalization_stdd.extend(norm_stats[1, channel_mask])
        normalization_chan.extend((np.arange(norm_stats.shape[1] - 4) + 1).tolist())

    # Store data in dataframe
    df = pd.DataFrame.from_dict({"days": normalization_days, "mean": normalization_mean,
                                 "stdd": normalization_stdd, "chan": normalization_chan})
    df["days"] = df["days"].astype('category')


    fig, ax = plt.subplots(figsize=(1100 / 96, 440 / 96))
    df['day_codes'] = df['days'].cat.codes + np.tile(np.random.uniform(-0.2, 0.2, 124), len(days))
    g = sns.scatterplot(data=df, x='day_codes', y='mean', hue='chan', size='stdd', sizes=(1, 124), ax=ax, palette="flare")
    g.legend_.remove()
    ax.set_ylabel("Mean statistics per channel")
    ax.set_xlabel("")
    day_labels = [f"Day {d}" for d in range(len(days))]
    ax.set_xticks(np.arange(len(df['days'].cat.categories)), day_labels, rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 3)

    # Colorbar
    cmap = plt.get_cmap("flare")
    norm = plt.Normalize(1, 124)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Channels")

    # Legend
    n_bullets = 10
    ax.legend(ax.get_children()[0].legend_elements("sizes", num=n_bullets)[0], [""] * n_bullets, loc=1, ncol=n_bullets,
              frameon=False, framealpha=1, title="Size represents Std", columnspacing=0.01, labelspacing=0.1,
              handlelength=1, facecolor="white")

    plt.subplots_adjust(left=0.057, right=1, bottom=0.14)
    if out_dir is not None:
        plt.savefig(out_dir / "supplementary_figure_1.png", dpi=150)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render supplementary figure 1")
    parser.add_argument("data", help="Path to the normalization dir.")
    parser.add_argument("-o", "--out", help="Path to the output folder.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f"python plot_normalization_si.py {args.data}" + f" {args.out}" if args.out else "")
    render_plot(Path(args.data), out_dir=Path(args.out) if args.out else None)
