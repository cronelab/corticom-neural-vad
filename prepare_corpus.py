import argparse
import numpy as np
import os
import logging
from typing import Optional, List, Dict
from local.corticom import ExperimentMapping
from local.corticom import HighGammaExtractor
from local.utils import SelectElectrodesFromBothGrids, BadChannelCorrection, CommonAverageReferencing
from local.vad import EnergyBasedVad
from pathlib import Path
from tqdm import tqdm
from scipy.io.wavfile import read as wavread
from local.utils import BCI2000MatFile
from local.utils import save_data_to_hdf
from itertools import pairwise
from corticom.auxilary.vad import MelFilterBank as mel
from scipy.signal.windows import hann


logger = logging.getLogger("prepare_corpus.py")


class FeatureExtractionPipeline:
    """
    Extract features and targets, such as LPC coefficients and voice activity labels, from all provided mat files.
    """
    def __init__(self, mat_filename: Path, wav_filename: Optional[Path] = None,
                 min_trial_length: Optional[float] = None):
        self.mat_filename = mat_filename
        self.wav_filename = wav_filename
        self.min_trial_length = min_trial_length
        self.mat = BCI2000MatFile(mat_filename=self.mat_filename.as_posix())

        if wav_filename is not None:
            self.fs_audio, self.wav = wavread(wav_filename)

    def get_features(self) -> np.ndarray:
        ecog = self.mat.signals()
        trials = self.mat.trial_indices(self.min_trial_length)

        # Extract experiment period
        start = trials[0][1]
        stop = trials[-1][2]

        extractor = get_feature_extractor(self.mat)
        features = extractor.extract_features(ecog[start:int(stop + (0.04 * self.mat.fs)), :])
        return features

    def get_vad_labels(self) -> np.ndarray:
        if self.wav_filename is None:
            raise ValueError("No wav filename provided.")

        trials = self.mat.trial_indices(self.min_trial_length)
        labels, start, stop = zip(*trials)
        indices = list(start) + [stop[-1]]
        segments = list(pairwise(indices))

        vad_labels = list()
        for label, (start, stop), trial_stop in zip(labels, segments, stop):
            interval = int(stop + (0.04 * self.mat.fs)) - start
            overlap = 0.04 * self.mat.fs
            window_shift = 0.01 * self.mat.fs
            num_windows = int(np.floor((interval - overlap) / window_shift))

            if label == "SILENCE":
                vad_labels.append(np.zeros(num_windows, dtype=np.int16))
                continue

            # Convert start and stop indices to high-fidelity audio sampling rate
            start = int(start * self.fs_audio / self.mat.fs)
            stop = int(stop * self.fs_audio / self.mat.fs) + int(0.04 * self.fs_audio)

            trial_audio = self.wav[start:stop]

            # Shift audio by 16 ms to account for filter delay
            filter_delay_pad = np.zeros(int(0.016 * self.fs_audio), dtype=np.int16)
            trial_audio = np.hstack([filter_delay_pad, trial_audio[:-len(filter_delay_pad)]])

            # Compute LPC features
            vad = EnergyBasedVad()
            labels = vad.from_wav(trial_audio, sampling_rate=self.fs_audio)
            vad_labels.append(labels)

        return np.hstack(vad_labels).astype(np.int16)

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

    def accumulative_audio_duration(self) -> float:
        accumulative_sum = 0.0
        for _, start, stop in self.mat.trial_indices(self.min_trial_length):
            accumulative_sum += stop - start

        return accumulative_sum / self.mat.fs


def compute_spectrogram(audio, sr=16000, window_length=0.05, window_shift=0.01, mel_bins=40) -> np.ndarray:
    window_length = int(sr * window_length)
    window_shift = int(sr * window_shift)
    overlap = window_length - window_shift

    num_windows = int(np.floor((len(audio) - overlap) / window_shift))

    segmentation = np.zeros((num_windows, window_length))
    for i in range(num_windows):
        segmentation[i] = audio[i * window_shift: i * window_shift + window_length]

    spectrogram = np.zeros((segmentation.shape[0], window_length // 2 + 1), dtype='complex')
    win = hann(window_length)
    for i in range(num_windows):
        spectrogram[i] = np.fft.rfft(segmentation[i] * win)

    mfb = mel(spectrogram.shape[1], mel_bins, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram


def get_feature_extractor(cleaned_mat_file: BCI2000MatFile) -> HighGammaExtractor:
    fs = cleaned_mat_file.fs
    bad_channels = cleaned_mat_file.bad_channels()
    contaminated_channels = cleaned_mat_file.contaminated_channels()

    # Reorder and select only channels from both grids
    feature_selection = SelectElectrodesFromBothGrids()
    pre_transforms = [feature_selection]

    speech_grid = np.flip(np.arange(64, dtype=np.int16).reshape((8, 8)) + 1, axis=0)
    motor_grid = np.flip(np.arange(64, dtype=np.int16).reshape((8, 8)) + 65, axis=0)
    layout = np.arange(128) + 1
    car = CommonAverageReferencing(exclude_channels=[19, 38, 48, 52], grids=[speech_grid, motor_grid], layout=layout)
    pre_transforms.append(car)
    post_transforms = None

    # Initialize channel correction
    if contaminated_channels is not None:
        logger.debug(f"Found contaminated channels in {cleaned_mat_file.mat_filename}: {contaminated_channels}.")
        corrected_channels = bad_channels + contaminated_channels
        ch_correction = BadChannelCorrection(bad_channels=corrected_channels, grids=[speech_grid, motor_grid],
                                             layout=layout)
        post_transforms = [ch_correction, ]

    # Initialize HighGammaExtraction module
    nb_electrodes = len(feature_selection)
    ex = HighGammaExtractor(fs=fs, nb_electrodes=nb_electrodes, pre_transforms=pre_transforms,
                            post_transforms=post_transforms)

    return ex


class ZScoresFromSyllableRepetitions(dict):
    """
    Creates a dictionary that stores z-score normalization statistics computed from the syllable repetition recordings.
    """
    def __init__(self, syllable_recordings: Dict[str, Path], show_pbar: bool = False):
        super(ZScoresFromSyllableRepetitions, self).__init__()

        desc = "Computing z-score statistics per day on SyllableRepetition data"
        pbar = not show_pbar
        for day, syllable_recording_path in tqdm(syllable_recordings.items(), desc=desc, disable=pbar):
            syllable_recording = BCI2000MatFile(mat_filename=syllable_recording_path.as_posix())

            ecog = syllable_recording.signals()
            data = list()
            for _, start, stop in syllable_recording.trial_indices():
                extractor = get_feature_extractor(syllable_recording)
                feats = extractor.extract_features(ecog[start:int(stop + (0.04 * syllable_recording.fs)), :])
                data.append(feats)

            normalization_data = np.concatenate(data)
            self[day] = (np.mean(normalization_data, axis=0), np.std(normalization_data, axis=0))


def main(out_base_path: Path, norm_dir: Path, folders: List[Path]):
    normalization_recordings = norm_dir.glob("**/*.mat")

    syllable_repetitions = {path.parent.name: path for path in normalization_recordings}
    z_score_mapping = ZScoresFromSyllableRepetitions(syllable_recordings=syllable_repetitions, show_pbar=True)

    accumulative_audio_sum = 0.0
    for folder in folders:
        mat_files = sorted(list(folder.glob("**/*.mat")))
        wav_files = [mat_file.with_suffix(".wav") for mat_file in mat_files]

        for mat_file, wav_file in zip(mat_files, wav_files):
            if mat_file.parent.name not in z_score_mapping:
                logger.warning(f"No normalization data for {mat_file.parent.name}. Skipping it!")
                continue

            pipeline = FeatureExtractionPipeline(mat_filename=mat_file, wav_filename=wav_file, min_trial_length=4.0)
            ecog = pipeline.get_features()
            targ = pipeline.get_vad_labels()
            tids = pipeline.get_trial_ids()
            accumulative_audio_sum += len(targ) * 0.01

            if not (len(ecog) == len(targ) == len(tids)):
                logger.error(f"Length mismatch: {len(ecog)}, {len(targ)}, {len(tids)}.")

            # Normalization for ecog data
            norm_means, norm_stds = z_score_mapping[mat_file.parent.name]
            ecog = (ecog - norm_means) / norm_stds

            # Store parameters in HDF container
            out_filename = Path(os.path.join(out_base_path.as_posix(), mat_file.parent.name,
                                             mat_file.with_suffix('.hdf').name))
            os.makedirs(out_filename.parent, exist_ok=True)
            parameters = dict(hga_activity=ecog, acoustic_labels=targ, trial_ids=tids)
            save_data_to_hdf(out_filename.as_posix(), parameters=parameters, overwrite=True)

            # Write normalization statistics to file
            norm_path = out_base_path.parent / "normalization"
            os.makedirs(norm_path, exist_ok=True)
            if not (norm_path / f"{mat_file.parent.name}.npy").exists():
                np.save(norm_path / f"{mat_file.parent.name}.npy", np.vstack([norm_means, norm_stds]))

    logger.info(f"Finished. Total of {accumulative_audio_sum / 60 / 60:.02f}h of speech data.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare the data corpus of the speech data into features and labels for training the neural "
                    "network architectures.")
    parser.add_argument("out_dir", help="Path to the parent directory in which the feature/label HDF "
                                        "files will be stored.")
    parser.add_argument("norm_dir", help="Path to parent directory in which the recording mat files from BCI2000 are stored that will be used to compute the normalization statistics.")
    parser.add_argument("folders", nargs='+', help="List of folders containing the recording mat files from BCI2000.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f'python prepare_corpus.py {args.out_dir} {args.norm_dir} {args.folders}')
    folders = [Path(folder) for folder in args.folders]
    main(out_base_path=Path(args.out_dir), norm_dir=Path(args.norm_dir), folders=folders)
