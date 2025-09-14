import argparse
import logging
import numpy as np
import h5py
import json
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
from warnings import simplefilter
from local.clustering import DilatedTICC as TICC
from local.utils import compute_trial_based_error
from itertools import product
from typing import List


logger = logging.getLogger("hyperparam_optim.py")


def main(patient_file: Path, betas: List[int], lambda_parameters: List[float]):
    # region Prepare data from the experiment runs
    simplefilter(action='ignore', category=FutureWarning)

    result = []
    for beta, lambda_parameter in product(betas, lambda_parameters):
        # Load patient features
        with h5py.File(patient_file.as_posix(), "r") as f:
            ecog = f["hga_activity"][...]
            tids = f["trial_ids"][...]
            targ = f["acoustic_labels"][...]

        # Run TICC on patient features
        with tempfile.NamedTemporaryFile() as temp:
            # Save data to temporary file so that it can get read by the TICC framework
            np.savetxt(temp.name, ecog, fmt="%.4e", delimiter=",")
            temp.flush()

            # Initialize the TICC clustering approach
            ticc = TICC(window_size=1, number_of_clusters=2, lambda_parameter=lambda_parameter, beta=beta,
                        maxIters=100, threshold=2e-5, num_proc=1)

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

            # Skip last second due to the differences in task designs (early_stop=100)
            err = compute_trial_based_error(nvad=cluster_assignment, vad=targ, trial_ids=tids, early_stop=100)
            logger.info(f"Average error with beta={beta} and lambda={lambda_parameter}: {np.mean(err):.05f} sec "
                        f"(±{np.std(err):.02f})")

            result.append((np.mean(err), beta, lambda_parameter))

    # Obtain best value
    best_score = np.inf
    best_index = None
    for i, score in enumerate(result):
        if score[0] < best_score:
            best_score = score[0]
            best_index = i

    # Log found hyperparameters
    best_err, best_beta, best_lambda = result[best_index]
    logger.info(f"Best results leading to an average score of {best_err}: beta={best_beta}, lambda={best_lambda}")

    # Store hyperparameters to file
    with open(patient_file.parent / "result.json", 'w') as f:
        json.dump(dict(beta=best_beta, lam=best_lambda), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Infer hyperparameters from PY21N004 for the TICC algorithm.")
    parser.add_argument("patient", help="Path to HDF5 container containing the patient data.")
    parser.add_argument("-b", "--betas", nargs='+', help="Range of beta values for the hyperparameter estimation.",
                        required=True)
    parser.add_argument("-l", "--lambdas", nargs='+', help="Range of lambda values for the hyperparameter estimation.",
                        required=True)
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    betas = [int(v) for v in args.betas]
    lambdas = [float(v) for v in args.lambdas]
    logger.info(f"python hyperparam_optim.py {args.patient} --betas " + " ".join(args.betas) + " --lambdas " +
                " ".join(args.lambdas))

    main(patient_file=Path(args.patient), betas=betas, lambda_parameters=lambdas)
