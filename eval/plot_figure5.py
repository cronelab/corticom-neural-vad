import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from local.utils import compute_trial_based_error, MulticolorPatch, MulticolorPatchHandler
from itertools import chain
from typing import Optional, List, Tuple


logger = logging.getLogger("plot_figure5.py")


def report_median_range(results: List[np.ndarray], descr: str):
    """
    Report the span of where the median error scores are distributed across days.
    """
    medians = [np.median(trial_scores) for trial_scores in results]
    logger.info(descr.format(np.min(medians), np.max(medians)))


def report_50_percent_range(results: List[np.ndarray], descr: str):
    """
    Report the range about where 50% of the alignment errors are.
    """
    trials = np.concatenate(results)
    q75, q25 = np.percentile(trials, [75, 25])
    logger.info(descr.format(q25, q75))


def load_baseline(files: List[Path]) -> List[np.ndarray]:
    """
    Load the baseline results to compute the statistics for the paper.
    """
    result = []

    for f in files:
        data = np.load(f.as_posix())
        errors = compute_trial_based_error(data[0, :], data[1, :], data[2, :])
        result.append(errors)

    return result


def main(baseline_dir: Path, nvad_dir: Path, out_dir: Optional[Path] = None):
    cnn_baseline = sorted((baseline_dir / "cnn").glob("*.npy"))
    rnn_baseline = sorted((baseline_dir / "rnn").glob("**/*.npy"))
    lr_baseline = sorted((baseline_dir / "lr").glob("*.npy"))
    nvad = sorted(nvad_dir.rglob("rnn/*/result.npy"))
    lr_nvad = sorted((nvad_dir / "lr").glob("*.npy"))
    cnn_nvad = sorted((nvad_dir / "cnn").glob("*.npy"))

    # Load fold specific results
    cnn_results = load_baseline(cnn_baseline)
    rnn_results = load_baseline(rnn_baseline)
    lr_results = load_baseline(lr_baseline)
    nvad_results = []
    lr_nvad_results = load_baseline(lr_nvad)
    cnn_nvad_results = load_baseline(cnn_nvad)

    # Report outliers
    counter, outlier_calc_rnn = calculate_outliers(nvad, nvad_results)
    _, outlier_calc_cnn = calculate_outliers(cnn_nvad, cnn_nvad_results)
    _, outlier_calc_lr = calculate_outliers(lr_nvad, lr_nvad_results)

    outlier_calc = np.mean([outlier_calc_rnn, outlier_calc_cnn, outlier_calc_lr])
    if counter > 0:
        logger.info(f"Outliers (> 1.2 s for avg speech length) across all days: {round(outlier_calc / counter, 3)} "
                    f"({outlier_calc} out of {counter} trials).")

    # Compute statistics for results section
    report_median_range(
        nvad_results,
        descr="RNN trained on TICC: Median range across days: {0:.03f} - {1:.03f} sec."
    )

    report_50_percent_range(
        nvad_results,
        descr="RNN trained on TICC: 50% of the trials are in the error range of {0:.03f} and {1:.03f} sec."
    )

    report_median_range(
        cnn_nvad_results,
        descr="CNN trained on TICC: Median range across days: {0:.03f} - {1:.03f} sec."
    )

    report_50_percent_range(
        cnn_nvad_results,
        descr="CNN trained on TICC: 50% of the trials are in the error range of {0:.03f} and {1:.03f} sec."
    )

    report_median_range(
        lr_nvad_results,
        descr="LR trained on TICC: Median range across days: {0:.03f} - {1:.03f} sec."
    )

    report_50_percent_range(
        lr_nvad_results,
        descr="LR trained on TICC: 50% of the trials are in the error range of {0:.03f} and {1:.03f} sec."
    )

    # Baseline statistics
    report_median_range(
        rnn_results,
        descr="RNN trained on ground truth: Median range across days: {0:.03f} - {1:.03f} sec."
    )

    report_50_percent_range(
        rnn_results,
        descr="RNN trained on ground truth: 50% of the trials are in the error range of {0:.03f} and {1:.03f} sec."
    )

    report_median_range(
        cnn_results,
        descr="CNN trained on ground truth: Median range across days: {0:.03f} - {1:.03f} sec."
    )

    report_50_percent_range(
        cnn_results,
        descr="CNN trained on ground truth: 50% of the trials are in the error range of {0:.03f} and {1:.03f} sec."
    )

    report_median_range(
        lr_results,
        descr="LR trained on ground truth: Median range across days: {0:.03f} - {1:.03f} sec."
    )
    report_50_percent_range(
        lr_results,
        descr="LR trained on ground truth: 50% of the trials are in the error range of {0:.03f} and {1:.03f} sec."
    )

    # Print sample sizes for boxplot
    for d, result in enumerate(nvad_results, start=1):
        logger.info(f"n samples for boxplot on day {d}: {len(result)}")

    # region Plot figure
    fig = plt.figure(figsize=(1100 / 96, 440 / 96))
    outlier_props = dict(linestyle="none", marker="o", alpha=0.5, linewidth=0.0)
    ax = fig.add_subplot(111)
    widths = 0.1
    spacing = 0.125

    # Proposed approach with LR, CNN and RNN
    ap_lr = ax.boxplot(x=lr_nvad_results, widths=widths, labels=[""] * len(lr_nvad_results),
                       positions=np.arange(len(lr_nvad_results)) - spacing / 2 - spacing - spacing,
                       patch_artist=True, flierprops=outlier_props, showfliers=False)

    ap_cnn = ax.boxplot(x=cnn_nvad_results, widths=widths, labels=[""] * len(cnn_nvad_results),
                        positions=np.arange(len(cnn_nvad_results)) - spacing / 2 - spacing,
                        patch_artist=True, flierprops=outlier_props, showfliers=False)

    ap_rnn = ax.boxplot(x=nvad_results, widths=widths, labels=[""] * len(nvad_results),
                        positions=np.arange(len(nvad_results)) - spacing / 2,
                        patch_artist=True, flierprops=outlier_props, showfliers=False)

    # Baseline box plots
    ref_lr = ax.boxplot(x=lr_results, labels=[f"Day {i + 1}" for i in range(len(cnn_baseline))],
                        positions=np.arange(len(cnn_baseline)) + spacing / 2, widths=widths,
                        patch_artist=True, flierprops=outlier_props, showfliers=False)

    ref_cnn = ax.boxplot(x=cnn_results, widths=widths, labels=[""] * len(cnn_baseline),
                         positions=np.arange(len(lr_baseline)) + spacing / 2 + spacing,
                         patch_artist=True, flierprops=outlier_props, showfliers=False)

    ref_rnn = ax.boxplot(x=rnn_results, widths=widths, labels=[""] * len(cnn_baseline),
                         positions=np.arange(len(lr_baseline)) + spacing / 2 + spacing + spacing,
                         patch_artist=True, flierprops=outlier_props, showfliers=False)

    ax.set_xticks(np.arange(len(cnn_baseline)))
    ax.set_xticklabels([f"Day {i + 1}" for i in range(len(cnn_baseline))])

    # Apply coloring to the box plots
    for medians in chain(ap_lr["medians"], ap_cnn["medians"], ap_rnn["medians"],
                         ref_lr["medians"], ref_cnn["medians"], ref_rnn["medians"]):
        medians.set_color("black")
        medians.set_linewidth(2)

    for boxes in ap_lr["boxes"]:
        boxes.set_facecolor("#6baed6")

    for boxes in ap_cnn["boxes"]:
        boxes.set_facecolor("#9ecae1")

    for boxes in ap_rnn["boxes"]:
        boxes.set_facecolor("#c6dbef")

    for boxes in ref_lr["boxes"]:
        boxes.set_facecolor("#fd8d3c")

    for boxes in ref_cnn["boxes"]:
        boxes.set_facecolor("#fdae6b")

    for boxes in ref_rnn["boxes"]:
        boxes.set_facecolor("#fdd0a2")

    ax.set_ylabel("nVAD alignment error [s]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.set_ylim(0, 2.0)
    ax.set_xlim(-0.5, 8.5)

    ax.axhline(y=1.2, linestyle="--", color="red", alpha=0.5)

    # Figure legend
    h, l = ax.get_legend_handles_labels()
    h.append(MulticolorPatch(["#6baed6", "#9ecae1", "#c6dbef"]))
    h.append(MulticolorPatch(["#fd8d3c", "#fdae6b", "#fdd0a2"]))
    l.append("Trained on estimated labels (LR, CNN & RNN)")
    l.append("Baseline (LR, CNN & RNN)")

    fig.legend(h, l, handler_map={MulticolorPatch: MulticolorPatchHandler()},
               ncols=2, frameon=False, loc='upper right', bbox_to_anchor=(0.975, 0.95))

    # Finalize plot
    # plt.xticks(rotation=45)
    plt.subplots_adjust(left=0.064, bottom=0.1, right=0.964, top=0.886)

    if out_dir is not None:
        plt.savefig(out_dir / "plot_figure3.png", dpi=300)
    else:
        plt.show()
    # endregion


def calculate_outliers(nvad, nvad_results) -> Tuple[int, int]:
    outlier_calc = 0
    counter = 0
    for nvad_fold in nvad:
        data = np.load(nvad_fold.as_posix())
        errors = compute_trial_based_error(data[0, :], data[1, :], data[2, :])
        nvad_results.append(errors)

        outlier_calc += np.sum(errors > 1.2)
        counter += len(errors)
    return counter, outlier_calc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render Figure 3")
    parser.add_argument("baseline", help="Path to the baseline directory.")
    parser.add_argument("nvad", help="Path to the baseline directory.")
    parser.add_argument("-o", "--out", help="Path to the output folder.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f"python plot_figure3.py {args.baseline} {args.nvad}" + (f" --out {args.out}" if args.out else ""))
    main(Path(args.baseline), Path(args.nvad), out_dir=Path(args.out) if args.out else None)
