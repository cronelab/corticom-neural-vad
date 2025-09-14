import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import h5py
from pathlib import Path
from scipy.io.wavfile import read as wavread
from scipy.signal import spectrogram
from local.utils import SelectElectrodesOverSpeechAreas, load_data_from_single_day, compute_trial_based_error
from local.utils import speech_detection_probability, false_alarm_probability, speech_detected
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Optional
matplotlib.use("TkAgg")


logger = logging.getLogger("plot_figure2.py")


def main(wav_filename: Path, hdf_filename: Path, csv_filename: Path, dev_day: Path,
         plot_start: int, plot_end: int, out_dir: Optional[Path] = None):
    # Read .wav file
    fs, audio = wavread(wav_filename)

    # Read normalized ecog features
    with h5py.File(hdf_filename, "r") as f:
        ecog = f["hga_activity"][...]
        speech_channels = SelectElectrodesOverSpeechAreas()
        ecog = speech_channels(ecog)

        acoustic_vad = f["acoustic_labels"][...]
        ticc_vad = f["ticc_labels"][...]
        trial_ids = f["trial_ids"][...]

    # Read trials to extract experiment run start and stop timings to truncate
    trials = pd.read_csv(csv_filename, sep="\t", names=["Start", "Stop", "Label"])
    start, _, _ = trials.iloc[0]
    _, stop, _ = trials.iloc[-1]
    start = int(start * fs)
    stop = int(stop * fs)

    # Truncate audio to only the trial segment
    audio = audio[start:stop]

    # Compute spectrogram
    f, t, spec = spectrogram(audio, fs, mode="magnitude", window="blackman",
                             nperseg=int(0.05 * fs), noverlap=int(0.04 * fs))

    # Fix edge case mismatch
    vmax = min([spec.shape[1], ecog.shape[0]])
    spec = spec[:, :vmax]
    ecog = ecog[:vmax, :]
    acoustic_vad = acoustic_vad[:vmax]
    ticc_vad = ticc_vad[:vmax]
    trial_ids = trial_ids[:vmax]

    # Visualize data
    fig, ((ax_bp1, ax_hga), (ax_bp2, ax_spec)) = plt.subplots(2, 2, figsize=(1100 / 96, 500 / 96), sharex=True,
                                                              width_ratios=(0.065, 0.935))

    # Neural data
    extent = [trials["Start"].iloc[0], trials["Stop"].iloc[-1], 0, len(speech_channels)]
    hga_im = ax_hga.imshow(ecog.T, aspect="auto", origin="lower", vmin=-4, vmax=4, cmap="PiYG", extent=extent,
                           interpolation="none")
    ax_hga.set_ylabel("Channels")
    ax_hga.set_yticks([0, len(speech_channels)])
    ax_hga.set_yticklabels([1, len(speech_channels)])

    ax_hga.spines["top"].set_visible(False)
    ax_hga.spines["bottom"].set_visible(False)
    ax_hga.spines["left"].set_visible(False)
    ax_hga.spines["right"].set_visible(False)

    # Audio data
    spec = 10 * np.log10(spec / np.max(spec))
    spec[spec < -50] = -50
    extent = [trials["Start"].iloc[0], trials["Stop"].iloc[-1], 0, 5000]
    spec_im = ax_spec.imshow(spec, aspect="auto", origin="lower", cmap="Greys", extent=extent, vmin=-40, vmax=0)

    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["bottom"].set_visible(False)
    ax_spec.spines["left"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)
    ax_spec.set_xlabel("Time [s]")
    ax_spec.set_ylim([0, 5000])

    ax_spec.set_yticks([0, 5000])
    ax_spec.set_yticklabels([0, 5])
    ax_spec.set_ylabel("Frequency [kHz]")

    # Plot VADs
    xs = np.linspace(trials["Start"].iloc[0], trials["Stop"].iloc[-1], len(acoustic_vad))
    nvad = ax_hga.plot(xs, ticc_vad * 64, c="black", linestyle="--", label="Neural clustering", zorder=1)
    avad = ax_spec.plot(xs, acoustic_vad * 5000, c="r", label="Reference acoustic VAD", zorder=1)
    fig.legend(ncols=2, frameon=False, loc='upper right', bbox_to_anchor=(0.93, 0.95))

    # Limit plot
    if np.isinf(plot_end):
        plot_end = int(len(ecog) * 0.01)

    if plot_start < trials["Start"].iloc[0]:
        plot_start = trials["Start"].iloc[0]
    ax_hga.set_xlim([plot_start, plot_end])
    ax_spec.set_xlim([plot_start, plot_end])

    # Left side of the plot
    gs = ax_bp1.get_gridspec()
    for ax in [ax_bp1, ax_bp2]:
        ax.remove()
    ax_bp = fig.add_subplot(gs[:, 0])

    dev_nvad, dev_vad, dev_tids = load_data_from_single_day(dev_day)
    err = compute_trial_based_error(dev_nvad, dev_vad, dev_tids)

    # Plot panel a
    outlier_props = dict(markerfacecolor="gray", linestyle="none")
    bp = ax_bp.boxplot(x=[err, ], widths=0.5, labels=[""], positions=np.arange(1),
                       patch_artist=True, flierprops=outlier_props)
    ax_bp.set_ylabel("nVAD alignment error [s]")
    bp["medians"][0].set_color("black")
    bp["medians"][0].set_linewidth(2)
    bp["boxes"][0].set_facecolor("plum")
    ax_bp.set_xticklabels(["Dev day"])
    ax_bp.spines["top"].set_visible(False)
    ax_bp.spines["right"].set_visible(False)
    ax_bp.spines["left"].set_linewidth(1.5)
    ax_bp.spines["bottom"].set_linewidth(1.5)

    # Add color bars
    axins_hga = inset_axes(ax_hga, width="100%", height="100%", borderpad=0,
                           bbox_to_anchor=(1.01, 0, 0.02, 1), bbox_transform=ax_hga.transAxes)

    fig.colorbar(hga_im, cax=axins_hga, orientation="vertical")
    axins_hga.set_xlabel("dB")

    axins_spec = inset_axes(ax_spec, width="100%", height="100%", borderpad=0,
                            bbox_to_anchor=(1.01, 0, 0.02, 1), bbox_transform=ax_spec.transAxes)

    fig.colorbar(spec_im, cax=axins_spec, orientation="vertical")
    axins_spec.set_xlabel("dB")

    # Add panel letters
    fig.text(0.015, 0.95, "a", fontsize=12, weight="bold")
    fig.text(0.150, 0.95, "b", fontsize=12, weight="bold")

    # Log statistics:
    logger.info(f"Dev set median alignment error: {np.median(err):.02} s")
    logger.info(f"Interquatile range (75% and 25%): {np.percentile(err, [75 ,25])}")
    logger.info(f"Speech detection probability {speech_detection_probability(dev_nvad, dev_vad, dev_tids):.02f} %")
    logger.info(f"False alarm probability {false_alarm_probability(dev_nvad, dev_vad, dev_tids):.02f} %")

    not_detected_count, trials_count = speech_detected(dev_nvad, dev_vad, dev_tids)
    logger.info(f"For {not_detected_count} out of {trials_count} no speech was detected.")
    top_k_errors = np.sum(err > 1.2)
    logger.info(f"{top_k_errors} trials resulted in greater alignment error than the average speaking duration.")
    logger.info(f"Largest 8 indices (starting with 1): {(np.argpartition(err, -top_k_errors)[-top_k_errors:]) + 1}")

    """
    # Plot trials with high alignment error
    from local.utils import get_trial
    tmp = np.argpartition(err, -top_k_errors)[-top_k_errors:]
    print(tmp)
    for i, (start, stop) in enumerate(get_trial(trial_ids=dev_tids)):
        if i not in tmp:
            continue

        p = dev_nvad[start:stop]
        t = dev_vad[start:stop]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t, c="black")
        ax.plot(p, c="r")
        ax.set_title(f"Trial: {i}")

        plt.show()
    """

    # Add example errors between panel b plots
    x_positions = [0.207, 0.355, 0.529, 0.666, 0.840]
    for error, x_pos in zip(compute_trial_based_error(ticc_vad, acoustic_vad, trial_ids)[:5], x_positions):
        fig.text(x_pos, 0.485, f"{round(error * 1000)} ms", fontsize=8)

    plt.subplots_adjust(wspace=0.136, left=0.064, right=0.924)
    if out_dir is not None:
        plt.savefig(out_dir / "figure_2.png", dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render Figure 2")
    parser.add_argument("wav", help="Path to the wavfile with the high-quality audio.")
    parser.add_argument("hdf", help="Path to the HDF5 container with the generated labels.")
    parser.add_argument("csv", help="Path to the .lab file with the stimuli timings.")
    parser.add_argument("dev", help="Which day to use as the development day.")
    parser.add_argument("-s", "--start", required=True, help="Path to the output folder.")
    parser.add_argument("-e", "--end", required=True, help="Path to the output folder.")
    parser.add_argument("-o", "--out", help="Path to the output folder.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f'python plot_figure2.py {args.wav} {args.hdf} {args.csv} {args.dev} {args.out}')
    main(Path(args.wav), Path(args.hdf), Path(args.csv), dev_day=Path(args.dev),
         plot_start=int(args.start), plot_end=int(args.end), out_dir=Path(args.out) if args.out else None)
