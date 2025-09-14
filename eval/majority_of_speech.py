import argparse
import numpy as np
import tqdm
import logging
from pathlib import Path
from local.utils import get_trial
from sklearn.metrics import recall_score


logger = logging.getLogger("majority_of_speech.py")


def main(unseen_results: Path):
    """
    Determine for each day how many trials the approach identified more than 50% odf the speech for
    the discussion section.
    """
    for results_filename in sorted(unseen_results.rglob("Day*.npy")):
        logger.info(f"Processing {results_filename.as_posix()}")

        counter = 0
        data = np.load(results_filename.as_posix())
        for start, stop in tqdm.tqdm(get_trial(trial_ids=data[2, :])):
            if recall_score(data[1, start:stop], data[0, start:stop], average="binary", pos_label=1) >= 0.5:
                counter += 1

        logger.info(f"Score: {counter / 721:.02f}")


if __name__ == "__main__":
    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    # read command line arguments
    parser = argparse.ArgumentParser("Report how many trials the approach identified more than 50% of the speech.")
    parser.add_argument("unseen_results_dir", help="Path to unseen_results folder.")
    args = parser.parse_args()

    logger.info(f'python majority_of_speech.py {args.unseen_results_dir}')
    main(Path(args.unseen_results_dir))
