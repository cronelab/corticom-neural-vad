import argparse
import logging
import os
import numpy as np
import h5py
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
from warnings import simplefilter
from local.clustering import DilatedTICC as TICC
from local.utils import SelectElectrodesOverSpeechAreas


logger = logging.getLogger("estimate_vad_labels.py")


# region Hyperparameter settings and training configuration
window_size = 1
max_iterations = 100
dev_days = []
# endregion


def main(target_dir: Path, corpus_dir: Path, beta: int, lambda_parameter: float):
    # region Prepare data from the experiment runs
    simplefilter(action='ignore', category=FutureWarning)
    sessions = sorted(corpus_dir.rglob("*.hdf"))

    # Skip all sessions from the development set
    sessions = [sess for sess in sessions if sess.parent.name not in dev_days]

    for sess in sessions:
        print(f"Processing session {sess.as_posix()}")

        # Perform clustering on validation data of current fold
        with h5py.File(sess.as_posix(), "r") as f, tempfile.NamedTemporaryFile() as temp:
            ecog = f["hga_activity"][...]
            tids = f["trial_ids"][...]
            ecog = SelectElectrodesOverSpeechAreas()(ecog)

            # Save data to temporary file so that it can get read by the TICC framework
            np.savetxt(temp.name, ecog, fmt="%.4e", delimiter=",")
            temp.flush()

            # Initialize the TICC clustering approach
            ticc = TICC(window_size=window_size, number_of_clusters=2, lambda_parameter=lambda_parameter,
                        beta=beta, maxIters=max_iterations, threshold=2e-5, num_proc=1)

            # Perform clustering on validation data (without the output on stdout)
            with redirect_stdout(None):
                cluster_assignment, mrf = ticc.fit(input_file=temp.name)

            # Write labels and markov random fields to target directory
            # We assume here that the non-speech cluster has the majority of frames
            cluster_assignment = cluster_assignment.astype(np.int16)
            if np.sum(cluster_assignment) > 0.5 * len(cluster_assignment):
                cluster_assignment = cluster_assignment - 1
                cluster_assignment = np.abs(cluster_assignment)

                # Swap also the markov random fields
                mrf[0], mrf[1] = mrf[1], mrf[0]

            filename = target_dir / sess.parent.name / sess.name
            os.makedirs(filename.parent.as_posix(), exist_ok=True)
            with h5py.File(filename.as_posix(), "w") as out:
                out.create_dataset("ticc_labels", data=cluster_assignment)
                out.create_dataset("speech_mrf", data=mrf[1])
                out.create_dataset("non_speech_mrf", data=mrf[0])
                out.create_dataset("hga_activity", data=f["hga_activity"][...])
                out.create_dataset("trial_ids", data=f["trial_ids"][...])
                out.create_dataset("acoustic_labels", data=f["acoustic_labels"][...])
    # endregion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Use the TICC algorithm to estimate the vad labels for a session")
    parser.add_argument("out_dir", help="Path to the output folder.")
    parser.add_argument("corpus", help="Path where the old HDF5 containers are stored.")
    parser.add_argument("--beta", "-b", default="400", help="The beta parameter to be used for TICC.")
    parser.add_argument("--lamb", "-l", default="0.0011", help="The lambda parameter to be used for TICC.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f'python estimate_vad_labels.py {args.out_dir} {args.corpus} --beta {args.beta} --lamb {args.lamb}')
    main(target_dir=Path(args.out_dir), corpus_dir=Path(args.corpus), beta=int(args.beta),
         lambda_parameter=float(args.lamb))
