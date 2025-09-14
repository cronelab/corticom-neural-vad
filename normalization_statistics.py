import argparse
import configparser
import logging
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional
from corticom.auxilary.bci2000 import BCI2000MatFile
from prepare_corpus import get_feature_extractor


logger = logging.getLogger("normalization_statistics.py")


def get_default_session_name() -> str:
    return datetime.now().strftime("%Y_%m_%d")


def get_paths(settings_filename: str) -> Tuple[str, str]:
    settings_config = configparser.ConfigParser()
    settings_config.read(settings_filename)

    # Compile path to session dir
    base_path = settings_config.get("Normalization", "base_path")
    session = get_default_session_name() if settings_config.get("Normalization", "session") == "" \
        else settings_config.get("Normalization", "session")
    session = os.path.join(base_path, session)

    # Get normalization file if provided
    norm_file = None if settings_config.get("Normalization", "normalization_file") == "" \
        else settings_config.get("Normalization", "normalization_file")

    return session, norm_file


def main(norm_files: List[Path], out: Optional[Path], overwrite: bool):
    logger.info("Aggregating trails on which normalization statistics will be computed.")
    trials = []
    for norm_file in norm_files:
        logger.info(f"Processing {norm_file}")
        mat_file = BCI2000MatFile(mat_filename=norm_file.as_posix())
        ecog = mat_file.signals()

        if mat_file.bad_channels() is not None:
            logger.warning(f"Found the following bad channels in the normalization data: {mat_file.bad_channels()}")

        for _, start, stop in mat_file.trial_indices():
            extractor = get_feature_extractor(mat_file)
            feats = extractor.extract_features(ecog[start:int(stop + (0.04 * mat_file.fs)), :])
            trials.append(feats)

    logger.info("Compute normalization statistics.")
    normalization_data = np.concatenate(trials)
    mean = np.mean(normalization_data, axis=0)
    std = np.std(normalization_data, axis=0)

    if out:
        if out.name.endswith(".npy"):
            os.makedirs(out.parent, exist_ok=True)
            out_filename = out.as_posix()
        else:
            os.makedirs(out, exist_ok=True)
            out_filename = out / "normalization.npy"
    else:
        out_filename = "normalization.npy"

    logger.info(f"Normalization statistics will be stored in {out_filename}")
    normalization_statistics = np.vstack([mean, std])
    np.save(out_filename, normalization_statistics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute z-score statistics from .mat files")
    parser.add_argument('norm', nargs='+',
                        help='List of mat files used to infer the normalization statistics.')
    parser.add_argument('--out', required=False, default=None,
                        help='Output path on were to store the resulting normalization statistics.')
    parser.add_argument('--overwrite', required=False, default=False, action='store_true',
                        help='Specify if existing statistics shall be overwritten.')
    args = parser.parse_args()

    # Initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    # Get session dir
    cmd = f"python normalization_statistics.py {args.norm}"
    if args.out:
        cmd += f" --out {args.out}"
    if args.overwrite:
        cmd += f" --overwrite"

    logger.info(cmd)
    norm_files = [Path(nf) for nf in args.norm]
    main(norm_files, out=Path(args.out) if args.out else None, overwrite=args.overwrite)
