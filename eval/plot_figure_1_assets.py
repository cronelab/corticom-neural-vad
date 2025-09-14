import argparse
import logging
import h5py
import numpy as np
import pandas as pd
import tempfile
import matplotlib
from local.utils import SelectElectrodesOverSpeechAreas
from matplotlib import pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from local.clustering import DilatedTICC
from typing import Optional
from contextlib import redirect_stdout
from scipy.io.wavfile import read as wavread
from itertools import pairwise
matplotlib.use("TkAgg")

logger = logging.getLogger("plot_figure_1_assets.py")


def main(recording: Path, wav_file: Path, lab_file: Path, start: int, end: int, out_dir: Optional[Path] = None):
    # Compute cluster assignments
    with h5py.File(recording.as_posix(), "r") as f, tempfile.NamedTemporaryFile() as temp:
        ecog = f["hga_activity"][...]
        ecog = SelectElectrodesOverSpeechAreas()(ecog)
        vad = f["acoustic_labels"][...]

        # Save data to temporary file so that it can get read by the TICC framework
        np.savetxt(temp.name, ecog, fmt="%.4e", delimiter=",")
        temp.flush()

        ticc = DilatedTICC(window_size=1, number_of_clusters=2, lambda_parameter=11e-4, beta=100, maxIters=100,
                           threshold=2e-5, num_proc=1)

        # Perform clustering on validation data (without the output on stdout)
        with redirect_stdout(None):
            cluster_assignment, _ = ticc.fit(input_file=temp.name)

    # Extract frames of interest
    data = np.zeros((len(ticc.iteration_log), end - start), dtype=np.int16)
    for i in range(len(ticc.iteration_log)):
        data[i, :] = ticc.iteration_log[i][start:end]

    if np.sum(cluster_assignment) > 0.5 * len(cluster_assignment):
        data = data - 1
        data = np.abs(data)

    # Extract waveform
    df = pd.read_csv(lab_file.as_posix(), sep='\t', names=["start", "stop", "word"])
    global_start = df.iloc[0]["start"]

    fs, wav = wavread(wav_file)
    wav = wav[int(global_start * fs) + int((start / 100) * fs):int(global_start * fs) + int((end / 100) * fs)]
    wav = wav / np.max(wav)
    wav *= 2

    fig, (ax_wav, ax_hga) = plt.subplots(figsize=(10, 5), nrows=2, ncols=1, height_ratios=(0.15, 0.85))
    ax_wav.plot(wav, c="gray")
    ax_wav.spines["top"].set_visible(False)
    ax_wav.spines["bottom"].set_visible(False)
    ax_wav.spines["left"].set_visible(False)
    ax_wav.spines["right"].set_visible(False)
    ax_wav.set_xlim(0, len(wav))
    ax_wav.set_xticks([])
    ax_wav.set_yticks([])
    vad = vad[start:end]
    diff = np.where(vad[:-1] != vad[1:])[0]
    for pos in diff:
        ax_wav.axvline(pos * 160, c="black", linewidth=2)
    diff = np.concatenate([np.array([0]), diff, np.array([len(vad)])])
    for f, s in list(pairwise(diff)):
        ax_wav.text((f + (s - f) * 0.5) * 160, 1.2, s=vad[f+1], fontsize=14, c="black")
    ax_wav.text(0, 1.2, s="Cluster:", fontsize=14, c="black")

    ax_hga.imshow(ecog[start:end, :].T, aspect="auto", origin="lower", cmap="PiYG")
    ax_hga.set_yticks([0, 63])
    ax_hga.set_yticklabels([1, 64])
    ax_hga.set_xticks(np.arange(0, end - start, 200))
    ax_hga.set_xticklabels([v // 100 for v in np.arange(0, end - start, 200)], fontsize=14)
    ax_hga.spines["top"].set_visible(False)
    ax_hga.spines["bottom"].set_visible(False)
    ax_hga.spines["left"].set_visible(False)
    ax_hga.spines["right"].set_visible(False)
    ax_hga.set_ylabel("Channels", labelpad=-10, fontsize=16)
    ax_hga.set_xlabel("Time [s]", fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.97, top=0.97, hspace=0.05)

    if out_dir is not None:
        plt.savefig(out_dir / "figure_1_hga_asset.png", dpi=150)
    else:
        plt.show()

    # Plot figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d', aspect='auto')

    x = np.arange(data.shape[1])  # timepoints
    y = np.arange(data.shape[0])  # iterations
    x, y = np.meshgrid(x, y)

    reds = plt.colormaps['Reds']

    # Render waveform
    ax.plot(xs=np.linspace(0, data.shape[1], len(wav)), ys=wav, color='black')

    # Render all iterations
    for i in range(len(data)):
        ax.plot(x[i], data[i], zs=y[len(data) - 1 - i], zdir='y', color=reds((i + 2) / len(data)), linewidth=1.0)

    # Configure plot layout
    ax.grid(False)
    ax.yaxis.pane.fill = False
    ax.xaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')

    ax.set_zlabel("Speech", labelpad=0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Iterations")

    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(0, data.shape[0] - 1)
    ax.set_xticks(np.linspace(0, data.shape[1], 5, endpoint=True))
    ax.set_xticklabels([0, 1, "", 3, 4, ])
    ax.set_yticks([0, data.shape[0] - 1])
    ax.set_yticklabels([data.shape[0] - 1, 0])
    ax.set_zticks([1, ])

    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([2.0, 1.0, 0.5, 1.75]))
    ax.xaxis.line.set_lw(0.)

    if out_dir is not None:
        plt.savefig(out_dir / "figure_1_convergence_asset.png", dpi=150)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render Figure 1 assets")
    parser.add_argument("recording", help="Path to a specific recording used to render the plot.")
    parser.add_argument("wav", help="Path to the corresponding wav file for that recording.")
    parser.add_argument("lab", help="Path to the corresponding lab file for that recording.")
    parser.add_argument("start", help="Starting frame to be plotted.")
    parser.add_argument("end", help="Ending frame to be plotted.")
    parser.add_argument("-o", "--out", help="Path to the output folder.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f"python plot_figure1_assets.py {args.recording} {args.wav} {args.lab} {args.start} {args.end}" +
                (f" --out {args.out}" if args.out else ""))
    main(Path(args.recording), Path(args.wav), Path(args.lab), int(args.start), int(args.end),
         out_dir=Path(args.out) if args.out else None)
