import argparse
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Optional


logger = logging.getLogger("plot_cluster_interpretation.py")


# region Electrode locations in brain plot
electrode_locations = [
    (1, 596.2444444444444, 400.5111111111111),
    (2, 599.7875, 421.65),
    (3, 602.123595505618, 443.59550561797755),
    (4, 604.56, 464.93),
    (5, 607.6987951807229, 485.7590361445783),
    (6, 610.8202247191011, 505.5056179775281),
    (7, 615.0617283950618, 528.3827160493827),
    (8, 618.3595505617977, 550.4494382022472),
    (9, 575.4642857142857, 402.45238095238096),
    (10, 579.4301075268817, 423.4623655913978),
    (11, 581.1851851851852, 446.5432098765432),
    (12, 583.109756097561, 468.3536585365854),
    (13, 586.445652173913, 489.5652173913044),
    (14, 590.1098901098901, 510.0769230769231),
    (15, 594.1590909090909, 531.3977272727273),
    (16, 597.9425287356322, 552.528735632184),
    (17, 553.9659090909091, 404.78409090909093),
    (18, 555.4903846153846, 427.11538461538464),
    (66, 400.4886363636364, 504.42045454545456),
    (20, 561.0430107526881, 469.98924731182797),
    (21, 566.7073170731708, 493.5487804878049),
    (22, 569.5888888888888, 513.3),
    (23, 573.1590909090909, 533.7045454545455),
    (24, 576.7848101265823, 555.5316455696203),
    (25, 531.6363636363636, 405.53409090909093),
    (26, 535.7875, 428.2375),
    (27, 539.0568181818181, 449.3863636363636),
    (28, 543.2048192771084, 471.6144578313253),
    (29, 545.4086021505376, 493.48387096774195),
    (30, 548.1647058823529, 514.5764705882353),
    (31, 552.0112359550562, 535.2022471910112),
    (32, 555.2528735632184, 556.4252873563219),
    (33, 511.02127659574467, 407.98936170212767),
    (34, 514.8705882352941, 429.45882352941175),
    (35, 519.1973684210526, 450.5263157894737),
    (36, 522.5851063829788, 473.531914893617),
    (37, 525.1609195402299, 495.41379310344826),
    (68, 406.8157894736842, 525.6973684210526),
    (39, 530.4444444444445, 536.4555555555555),
    (40, 534.0625, 557.65),
    (41, 490.4555555555556, 410.4),
    (42, 494.1034482758621, 431.4022988505747),
    (43, 497.04938271604937, 451.44444444444446),
    (44, 500.23809523809524, 472.4642857142857),
    (45, 503.32142857142856, 494.35714285714283),
    (46, 507.4421052631579, 515.5157894736842),
    (47, 510.2307692307692, 537.4358974358975),
    (69, 414.95180722891564, 545.6265060240963),
    (49, 469.96629213483146, 412.7303370786517),
    (50, 473.1162790697674, 433.5348837209302),
    (51, 476.28915662650604, 454.51807228915663),
    (82, 352.7906976744186, 503.5),
    (53, 483.52222222222224, 496.44444444444446),
    (54, 487.05, 517.6125),
    (55, 490.1309523809524, 538.6904761904761),
    (56, 491.77906976744185, 560.5116279069767),
    (57, 448.9512195121951, 414.3658536585366),
    (58, 452.2048192771084, 435.34939759036143),
    (59, 455.47, 455.92),
    (60, 458.6875, 476.65),
    (61, 463.9756097560976, 495.6341463414634),
    (62, 467.17045454545456, 519.4659090909091),
    (63, 470.8941176470588, 539.5294117647059),
    (64, 473.23809523809524, 563.3809523809524),
]

_, electrodes_y, electrodes_x = zip(*electrode_locations)
electrodes_x = np.array(electrodes_x)
electrodes_y = np.array(electrodes_y)
# endregion


def main(cluster_differences: Path, brain_plot_img: Path, out_dir: Optional[Path] = None, display_size: int = 128):
    # Data loading and re-formating
    data = np.load(cluster_differences.as_posix())
    electrode_contrib = np.diagonal(data)
    conn_contrib = data * (np.ones_like(data) - np.eye(len(data)))

    # Color definition
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=np.max(data))
    colormap = matplotlib.colormaps["Reds"]

    # Create the figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    brainplot = plt.imread(brain_plot_img.as_posix())

    # Plot the image and the scatters of the channels
    ax.imshow(brainplot)
    ax.scatter(electrodes_x, electrodes_y, c=colormap(normalize(electrode_contrib)),
               s=np.sum(conn_contrib, axis=0) * display_size, linewidths=0.5, edgecolors="black")

    ax.set_xticks([])
    ax.set_yticks([])

    # Hide black borders
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Insert the channel contribution gradient
    inset_ax = inset_axes(ax, width="25%", height=0.15, loc="upper right")
    inset_ax.set_title("Channel contribution", fontsize="medium")
    inset_ax.imshow(np.linspace(0, np.max(data), 1000).reshape((1, 1000)), aspect="auto", origin="lower",
                    cmap="Reds", vmin=0, vmax=np.max(data))
    inset_ax.set_xticks([0, 999])
    inset_ax.set_xticklabels(["0", "max"])
    inset_ax.set_yticks([])

    # Insert size guidelines
    minimum = np.min(np.sum(conn_contrib, axis=0) * display_size)
    maximum = np.max(np.sum(conn_contrib, axis=0) * display_size)
    ax.scatter(x=[950, 1050], y=[800, 800], s=[minimum, maximum], linewidths=0.75, edgecolors="black",
               facecolors="white")
    ax.text(x=875, y=765, s="Total interdependency", fontsize="medium")

    # Render the plot
    if out_dir is not None:
        plt.savefig(out_dir / "plot_cluster_interpretation.png", dpi=150)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot cluster interpretation plot")
    parser.add_argument("diff", help="Path to the results from the interpretation analysis.")
    parser.add_argument("img", help="Path to the brain plot image.")
    parser.add_argument("-o", "--out", help="Path to the output folder.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f"python plot_cluster_interpretation.py {args.diff} {args.img}" +
                (f" --out {args.out}" if args.out else ""))
    main(Path(args.diff), Path(args.img), out_dir=Path(args.out) if args.out else None)
