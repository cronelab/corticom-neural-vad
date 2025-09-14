import argparse
import logging
import h5py
import numpy as np
from pathlib import Path
from local.utils import get_trial


logger = logging.getLogger("avg_speech_duration.py")


def main(corpus_dir: Path):
    """
    Compute average speech duration.
    """
    result = []

    feature_files = list(corpus_dir.rglob('*.hdf'))
    for feat_file in feature_files:
        with h5py.File(feat_file, 'r') as f:
            vad = f["acoustic_labels"][...]
            tid = f["trial_ids"][...]

        for start, stop in get_trial(tid):
            if np.all(vad[start:stop] == 0):
                # Skip silence trials
                continue

            result.append(np.count_nonzero(vad[start:stop]) * 0.01)

    logger.info(f"Average duration from participants saying the words: {np.mean(result):.02} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Measure average speech duration for a dataset.")
    parser.add_argument("corpus", help="Path where the feature .hdf5 files are stored.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f'python avg_speech_duration.py {args.corpus}')
    main(corpus_dir=Path(args.corpus))
