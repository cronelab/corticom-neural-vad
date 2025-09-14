import argparse
import numpy as np
import os
import logging
import pandas as pd
from prepare_corpus import FeatureExtractionPipeline
from typing import Optional, Tuple
from local.corticom import HighGammaExtractor
from local.utils import CommonAverageReferencing, SelectElectrodesFromPY17N009
from pathlib import Path
from local.utils import BCI2000MatFile
from local.utils import save_data_to_hdf
from itertools import pairwise
from local.corticom import ExperimentMapping


logger = logging.getLogger("prepare_PY17N009.py")


class PY17N009Pipeline(FeatureExtractionPipeline):
    """
    Feature computation pipeline for our patient PY21N004 to infer initial values for the clustering approach.
    """
    def __init__(self, mat_filename: Path, min_trial_length: Optional[float] = None):
        super(PY17N009Pipeline, self).__init__(mat_filename, mat_filename.with_suffix(".wav"), min_trial_length)
        self.vad_filename = mat_filename.with_suffix(".vad")

    def get_features(self) -> np.ndarray:
        """
        Compute high-gamma features for PY21N004 from the implanted ECoG grid.
        """
        ecog = self.mat.signals()
        trials = self.mat.trial_indices(self.min_trial_length)

        # Extract experiment period
        start = trials[0][1]
        stop = trials[-1][2]

        extractor = get_feature_extractor(self.mat)
        features = extractor.extract_features(ecog[start:int(stop + (0.04 * self.mat.fs)), :])
        return features

    def get_vad_labels(self) -> np.ndarray:
        """
        Obtain the VAD information from file.
        """
        df_vad = pd.read_csv(self.vad_filename, sep="\t", names=["start", "stop", "speech"])
        df_trial = pd.read_csv(self.vad_filename.parent / (self.vad_filename.stem + "_trials.lab"),
                               sep="\t", names=["start", "stop", "speech"])
        start = df_trial.iloc[0].start

        df_vad.start -= start
        df_vad.stop -= start
        df_vad.loc[df_vad.start < 0, "start"] = 0
        df_vad.loc[df_vad.stop < 0, "stop"] = 0

        # Compute number of frames
        trials = self.mat.trial_indices(self.min_trial_length)
        start = trials[0][1]
        stop = trials[-1][2]
        nb_frames = (stop - start) // 20

        # Fill VAD information
        frames = np.zeros(nb_frames, dtype=np.int16)

        for _, (start, stop, label) in df_vad.iterrows():
            if label == 1:
                start = int(round(start, 2) * 100)
                stop = int(round(stop, 2) * 100)

                frames[start:stop] = 1

        return frames

    def get_trial_ids(self) -> np.ndarray:
        stimuli = ExperimentMapping.extract_stimuli_values(self.mat.mat)

        trials = self.mat.trial_indices(self.min_trial_length)
        labels, start, stop = zip(*trials)
        indices = list(start) + [stop[-1]]
        segments = list(pairwise(indices))

        trial_ids = list()
        last_stimuli_code = None
        for label, (start, stop), trial_stop in zip(labels, segments, stop):
            interval = int(stop + (0.04 * self.mat.fs)) - start
            overlap = 0.04 * self.mat.fs
            window_shift = 0.01 * self.mat.fs
            num_windows = int(np.floor((interval - overlap) / window_shift))

            stimuli_code = stimuli.index(label) + 1
            if last_stimuli_code is None or last_stimuli_code != stimuli_code:
                trial_ids.append(np.ones(num_windows) * stimuli_code)
                last_stimuli_code = stimuli_code
            else:
                trial_ids.append(np.ones(num_windows) * stimuli_code * -1)
                last_stimuli_code = stimuli_code * -1

        return np.hstack(trial_ids).astype(np.int16)


def get_feature_extractor(cleaned_mat_file: BCI2000MatFile) -> HighGammaExtractor:
    """
    Return the configuration for computing the high-gamma features.
    """
    fs = cleaned_mat_file.fs

    feature_selection = SelectElectrodesFromPY17N009()
    pre_transforms = [feature_selection]

    ecog_grid = np.arange(len(feature_selection), dtype=np.int16).reshape(feature_selection.shape()) + 1
    layout = np.arange(len(feature_selection)) + 1
    car = CommonAverageReferencing(exclude_channels=[], grids=[ecog_grid, ], layout=layout)
    pre_transforms.append(car)
    post_transforms = None

    # Initialize HighGammaExtraction module
    nb_electrodes = len(feature_selection)
    ex = HighGammaExtractor(fs=fs, nb_electrodes=nb_electrodes, pre_transforms=pre_transforms,
                            post_transforms=post_transforms)

    return ex


def get_baseline_statistics(mat_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute z-score features from the data of PY17N009.
    """
    mat = BCI2000MatFile(mat_filename=mat_file.as_posix())
    ecog = mat.signals()

    normalization_data = []
    for _, start, stop in mat.trial_indices():
        # Extract 1 s of data before each trial (before played sound appears) as baseline periods
        stop = start - mat.fs  # Stop 1 second before trial start
        start = start - 2 * mat.fs  # Start 2 seconds before trial start

        # Compute features on those windows
        extractor = get_feature_extractor(mat)
        feats = extractor.extract_features(ecog[start:stop, :])

        normalization_data.append(feats)

    # Return baseline statistics
    normalization_data = np.concatenate(normalization_data)
    return np.mean(normalization_data, axis=0), np.std(normalization_data, axis=0)


def main(out_base_path: Path, mat_file: Path):
    """
    Prepare features for PY17N009 using a similar pipeline as with CC01.
    """
    norm_means, norm_stds = get_baseline_statistics(mat_file)

    accumulative_audio_sum = 0.0
    pipeline = PY17N009Pipeline(mat_filename=mat_file)
    ecog = pipeline.get_features()
    targ = pipeline.get_vad_labels()
    tids = pipeline.get_trial_ids()
    accumulative_audio_sum += len(targ) * 0.01

    if not (len(ecog) == len(targ) == len(tids)):
        logger.error(f"Length mismatch: {len(ecog)}, {len(targ)}, {len(tids)}.")

    # Normalization for ecog data
    ecog = (ecog - norm_means) / norm_stds

    # Store parameters in HDF container
    out_filename = out_base_path / f"{mat_file.stem}.hdf"
    os.makedirs(out_filename.parent, exist_ok=True)
    parameters = dict(hga_activity=ecog, acoustic_labels=targ, trial_ids=tids)
    save_data_to_hdf(out_filename.as_posix(), parameters=parameters, overwrite=True)

    logger.info(f"Finished. Total of {accumulative_audio_sum / 60 / 60:.02f}h of speech data.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare the data for the epilepsy patient to infer hyperparameters not from CC01 data.")
    parser.add_argument("out_dir", help="Path to the output directory.")
    parser.add_argument("mat", help="Path to the mat file from BCI2000.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f'python prepare_PY17N009.py {args.out_dir} {args.mat}')
    main(out_base_path=Path(args.out_dir), mat_file=Path(args.mat))
