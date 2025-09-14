import argparse
import logging
import sys
sys.path.insert(0, "TICC")
import h5py
import tempfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from warnings import simplefilter
from local.utils import SelectElectrodesOverSpeechAreas, compute_trial_based_error
from local.clustering import DilatedTICC as TICC
from contextlib import redirect_stdout
matplotlib.use("TkAgg")

logger = logging.getLogger("compute_temporal_context.py")


def main(dev_day: Path, out_dir: Path, beta: int = 100, lambda_parameter: float = 0.11, max_iterations: int = 100,
         step_size: int = 5, start_ws: int = 1, end_ws: int = 8):
    simplefilter(action='ignore', category=FutureWarning)
    sessions = sorted(dev_day.glob("*.hdf"))

    # The channel selector is only necessary for the data from CC01 patient
    channel_selector = SelectElectrodesOverSpeechAreas()

    # Initialize the TICC clustering approach
    for ws in range(start_ws, end_ws):
        rs = []
        wrote_mrfs = False
        for sess in sessions:
            print(f"Processing ws={ws}, sess: {sess.as_posix()}")
            with h5py.File(sess.as_posix(), "r") as f, tempfile.NamedTemporaryFile() as temp:
                ecog = f["hga_activity"][...]
                targ = f["acoustic_labels"][...]
                tids = f["trial_ids"][...]

                if sess.parent.name != "PY17N009":
                    ecog = channel_selector(ecog)

                # Save data to temporary file so that it can get read by the TICC framework
                np.savetxt(temp.name, ecog, fmt="%.4e", delimiter=",")
                temp.flush()

                ticc = TICC(window_size=ws, number_of_clusters=2, lambda_parameter=lambda_parameter,
                            beta=beta, maxIters=max_iterations, threshold=2e-5, num_proc=4, dilation=step_size)

                # Perform clustering on validation data (without the output on stdout)
                with redirect_stdout(None):
                    cluster_assignment, mrf = ticc.fit(input_file=temp.name)
                    cluster_assignment = cluster_assignment.astype(np.int16)
                    speech_mrfs = 1

                    if np.sum(cluster_assignment) > 0.5 * len(cluster_assignment):
                        cluster_assignment = cluster_assignment - 1
                        cluster_assignment = np.abs(cluster_assignment)
                        speech_mrfs = 0

                    # Render cluster plots for figure 1 at window_size 5
                    if sess.parent != "PY17N009" and ws == 5 and not wrote_mrfs:
                        # Define file indicators for -200 ms, -150 ms, ...
                        temporal_context_file_indicators = list(range(-200, 50, 50))

                        # Iterate over both clusters
                        for cluster_id in mrf.keys():
                            minimum = np.min(np.abs(mrf[cluster_id]))
                            maximum = np.max(np.abs(mrf[cluster_id]))

                            # Iterate over the individual Markov Random Field windows
                            for win, i in enumerate(range(0, mrf[cluster_id].shape[0], len(channel_selector))):
                                fig = plt.figure(figsize=(1, 1))
                                ax = fig.add_subplot(111)
                                # Limit visualization only to 16 channels (otherwise not much will be visible)
                                ax.imshow(np.abs(mrf[cluster_id][i:(i + 16), i:(i + 16)]), cmap="Reds", aspect="equal",
                                          vmin=minimum, vmax=maximum)
                                ax.set_xticks([])
                                ax.set_yticks([])

                                filename = (f'{"speech" if cluster_id == speech_mrfs else "non_speech"}'
                                            f'{str(temporal_context_file_indicators[win])}ms.png')
                                plt.savefig(out_dir / filename)
                                plt.close(fig)

                        wrote_mrfs = True

                rs.append(compute_trial_based_error(cluster_assignment, targ, tids))

        rs = np.hstack(rs)

        # Save results
        filename = f"ws={ws}" if sessions[0].parent != "PY17N009" else f"ws={ws}_PY17N009"
        logger.info(f"{filename}: {np.median(rs)}")
        np.save(out_dir / filename, rs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute temporal context errors")
    parser.add_argument("dev", help="Path to the development day folder (or the folder to PY17N009).")
    parser.add_argument("out", help="Path to the output folder.")
    parser.add_argument("-s", "--start", help="Starting window size.", default="1")
    parser.add_argument("-e", "--end", help="End window size.", default="7")
    parser.add_argument("-b", "--beta", help="Beta hyperparameter.", default="400")
    parser.add_argument("-l", "--lamb", help="Lambda hyperparameter.", default="0.0011")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f'python compute_temporal_context.py {args.dev} {args.out} --start {args.start} --end {args.end} '
                f'--beta {args.beta} --lamb {args.lamb}')
    main(Path(args.dev), Path(args.out), start_ws=int(args.start), end_ws=int(args.end) + 1,
         beta=int(args.beta), lambda_parameter=float(args.lamb))
