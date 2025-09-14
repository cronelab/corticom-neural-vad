import argparse
import numpy as np
import logging
from pathlib import Path
from local.corticom import NGSLSWordReading, FiftyWordReading
from local.utils import get_trial, compute_trial_based_error


logger = logging.getLogger("eval_generalization.py")


def compute_statistics(results_dir: Path, prefix: str):
    filenames = sorted(results_dir.glob(f"{prefix}*.npy"))

    # Focus only on stimuli that have not been seen in the training data
    unseen_stimuli = {i: s for i, s in enumerate(NGSLSWordReading.get_stimuli_values())
                      if s not in FiftyWordReading.get_stimuli_values()}
    logger.info(f"Number of unseen words for {prefix}: {len(unseen_stimuli)}")

    median_alignment_errors = []
    errors = []
    for filename in filenames:
        data = np.load(filename.as_posix())

        pred, orig, tids = [], [], []
        for start, stop in get_trial(data[2, :]):
            trial_indices = data[2, start:stop]
            if len(np.unique(trial_indices)) != 1:
                print("Something went wrong")

            if trial_indices[0] not in unseen_stimuli.keys():
                continue
            else:
                pred.append(data[0, start:stop])
                orig.append(data[1, start:stop])
                tids.append(data[2, start:stop])

        # Concatenate
        pred = np.concatenate(pred)
        orig = np.concatenate(orig)
        tids = np.concatenate(tids)

        data = np.vstack([pred, orig, tids])

        # Compute median error
        err = compute_trial_based_error(data[0, :], data[1, :], data[2, :])
        median_alignment_errors.append(np.median(err))
        errors.extend(err.tolist())

    logger.info(f"{prefix}: Alignment errors min/max: "
                f"{np.min(median_alignment_errors):.02f} / {np.max(median_alignment_errors):.02f} s")
    q75, q25 = np.percentile(errors, [75, 25])
    logger.info(f"{prefix}: 50% of errors in the range of {q25:.02f}, {q75:.02f} s")


def main(results_dir: Path):
    compute_statistics(results_dir, "rnn")
    compute_statistics(results_dir, "cnn")
    compute_statistics(results_dir, "lr")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Computer the reported numbers in the paper on the unseen data.")
    parser.add_argument("dir", help="Path to the unseen_results directory.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f'python eval_generalization.py {args.dir}')
    main(results_dir=Path(args.dir))
