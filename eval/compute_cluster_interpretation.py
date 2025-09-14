import argparse
import logging
import sys
sys.path.insert(0, "TICC")
import h5py
import tempfile
import numpy as np
from pathlib import Path
from warnings import simplefilter
from local.utils import SelectElectrodesOverSpeechAreas
from contextlib import redirect_stdout
from TICC.TICC_solver import TICC


logger = logging.getLogger("compute_cluster_interpretation.py")


def main(dev_day: Path, out_dir: Path = None, beta: int = 400, lambda_parameter: float = 0.0011):
    simplefilter(action='ignore', category=FutureWarning)
    sessions = sorted(dev_day.glob("*.hdf"))

    channel_selector = SelectElectrodesOverSpeechAreas()

    # Initialize the TICC clustering approach
    sess = sessions[0]
    with h5py.File(sess.as_posix(), "r") as f, tempfile.NamedTemporaryFile() as temp:
        ecog = channel_selector(f["hga_activity"][...])

        # Save data to temporary file so that it can get read by the TICC framework
        np.savetxt(temp.name, ecog, fmt="%.4e", delimiter=",")
        temp.flush()

        ticc = TICC(window_size=1, number_of_clusters=2, lambda_parameter=lambda_parameter,
                    beta=beta, maxIters=100, threshold=2e-1, num_proc=1)

        # Perform clustering on validation data (without the output on stdout)
        with redirect_stdout(None):
            cluster_assignment, mrf = ticc.fit(input_file=temp.name)
            cluster_assignment = cluster_assignment.astype(np.int16)
            if np.sum(cluster_assignment) < 0.5 * len(cluster_assignment):
                # Swap  markov random fields to have the speech cluster parameters in the first position
                mrf[0], mrf[1] = mrf[1], mrf[0]

        np.save(out_dir / "parameter_differences.npy", np.abs(mrf[0]) - np.abs(mrf[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute differences for speech and non speech cluster parameters.")
    parser.add_argument("dev_day", help="Path to the .HDF data for te development dir.")
    parser.add_argument("out", help="Path to the output folder.")
    parser.add_argument("--beta", "-b", default="400", help="The beta parameter to be used for TICC.")
    parser.add_argument("--lamb", "-l", default="0.0011", help="The lambda parameter to be used for TICC.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f"python compute_cluster_interpretation.py {args.dev_day} {args.out} "
                f"--beta {args.beta} --lamb {args.lamb}")
    main(Path(args.dev_day), out_dir=Path(args.out), beta=int(args.beta), lambda_parameter=float(args.lamb))
