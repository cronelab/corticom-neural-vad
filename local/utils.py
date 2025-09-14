import os
import math
import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Levenshtein import distance
from itertools import pairwise
from typing import Callable, List, Tuple, Optional, Dict, TypeAlias
from pathlib import Path
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MatplotlibPath
from scipy.io import loadmat
from local.corticom import ExperimentMapping
from collections import defaultdict
from torch.utils.data import Dataset
from operator import itemgetter
from scipy.ndimage import binary_dilation


# region Alignment error based on DTW distance
def get_trial(trial_ids: np.ndarray) -> List[Tuple[int, int]]:
    """
    Get the trials based on the stimuli codes in the .mat file.
    """
    split_indices = np.where(trial_ids[:-1] != trial_ids[1:])[0] + 1
    split_indices = [0] + split_indices.tolist() + [len(trial_ids)]
    return list(pairwise(split_indices))


def compute_trial_based_error(nvad: np.ndarray, vad: np.ndarray, trial_ids: np.ndarray,
                              early_stop: int = 0) -> np.ndarray:
    """
    Compute for each trial the alignment error using the Levenshtein distance. Early stopping criteria is used to
    exclude the period of 1 s of the beep sound in the syllable repetition task.
    """
    result = []
    for start, stop in get_trial(trial_ids=trial_ids):
        p = nvad[start:stop-early_stop]
        t = vad[start:stop-early_stop]
        d = distance(p, t)
        result.append(d * 0.01)

    return np.array(result, dtype=np.float32)


def load_data_from_single_day(day: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from a single day, such as the development day
    """
    data = defaultdict(list)
    blocks = sorted(day.glob("*.hdf"))
    for block in blocks:
        with h5py.File(block.as_posix(), "r") as hf:
            data["orig"].append(hf["acoustic_labels"][...])
            data["pred"].append(hf["ticc_labels"][...])
            data["tids"].append(hf["trial_ids"][...])

    return np.concatenate(data["pred"]), np.concatenate(data["orig"]), np.concatenate(data["tids"])


def speech_detection_probability(nvad: np.ndarray, vad: np.ndarray, trial_ids: np.ndarray,
                                 early_stop: int = 0) -> float:
    """
    Compute the average probability across all trials how likely it is that a single frame, that actually a speech
    frame, gets classified as one.
    """
    prob = []
    for start, stop in get_trial(trial_ids=trial_ids):
        p = nvad[start:stop - early_stop]
        t = vad[start:stop - early_stop]
        if np.sum(t) == 0:
            # Skip SILENCE trials
            continue

        prob.append(np.sum(p & t) / np.sum(t))

    return np.mean(prob).item()


def false_alarm_probability(nvad: np.ndarray, vad: np.ndarray, trial_ids: np.ndarray,
                            early_stop: int = 0) -> float:
    """
    Compute the average probability across all trials how likely it is that a single non-speech frame gets falsely
    classified as a speech frame.
    """
    prob = []
    for start, stop in get_trial(trial_ids=trial_ids):
        p = nvad[start:stop - early_stop]
        t = vad[start:stop - early_stop]
        if np.sum(t) == 0:
            # Skip SILENCE trials
            continue

        len_silence = len(t) - np.sum(t)
        fa = np.sum((t - p) < 0)
        prob.append(fa / len_silence)

    return np.mean(prob).item()


def speech_detected(nvad: np.ndarray, vad: np.ndarray, trial_ids: np.ndarray) -> Tuple[int, int]:
    """
    Count the number of trials where not a single frame from the ground truth speech gets correctly classified.
    """
    not_detected_count = 0
    trials_count = 0
    for start, stop in get_trial(trial_ids=trial_ids):
        p = nvad[start:stop]
        m = vad[start:stop] > 0
        if np.sum(p[m]) == 0:
            not_detected_count += 1

        trials_count += 1

    return not_detected_count, trials_count
# endregion


# region Preprocessing
class SelectElectrodesOverSpeechAreas:
    """
    Feature selection covering only electrodes that I have been identified as carrying speech information. This
    selection method tries to keep much of the original electrode alignment by dropping the 4 bad channels and replace
    them with the 4 channels from the upper limb grid. Keeping the spatial alignment helps the convolutional models in
    extracting spatial information
    """
    def __init__(self):
        self.speech_grid_mapping = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,
                                              8,  9, 10, 11, 12, 13, 14, 15,
                                             16, 17, 66, 19, 20, 21, 22, 23,
                                             24, 25, 26, 27, 28, 29, 30, 31,
                                             32, 33, 34, 35, 36, 67, 38, 39,
                                             40, 41, 42, 43, 44, 45, 46, 68,
                                             48, 49, 50, 81, 52, 53, 54, 55,
                                             56, 57, 58, 59, 60, 61, 62, 63, ])

    def __len__(self):
        return len(self.speech_grid_mapping)

    def __call__(self, data):
        return data[:, self.speech_grid_mapping]

    def __repr__(self):
        return f"Channels: {', '.join(map(str, self.speech_grid_mapping + 1))}"


class SelectElectrodesFromBothGrids:
    """
    Feature selection covering the electrodes from both grids (ordered chan1, chan2, ..., chan64, chan65, chan66, ...)
    """
    def __init__(self):
        self.grid_mapping = [125, 123, 121, 119, 122, 111, 118, 124, 120, 126, 127, 116, 114, 113, 115, 117, 98, 97, 96,
                             104, 100, 102, 101, 99, 105, 112, 107, 106, 108, 103, 109, 110, 17, 21, 9, 28, 26, 31, 13,
                             27, 25, 22, 30, 11, 29, 23, 19, 15, 1, 2, 4, 0, 24, 12, 14, 7, 5, 18, 6, 10, 3, 8, 20, 16,
                             50, 33, 44, 51, 63, 40, 38, 46, 42, 48, 56, 37, 35, 41, 47, 58, 61, 60, 59, 43, 49, 45, 54,
                             62, 32, 53, 55, 52, 57, 39, 34, 36, 85, 84, 83, 87, 80, 86, 90, 78, 75, 92, 76, 88, 82, 94,
                             70, 74, 69, 66, 79, 71, 73, 77, 68, 67, 64, 65, 95, 93, 81, 72, 91, 89]

    def __len__(self):
        return len(self.grid_mapping)

    def __call__(self, data):
        return data[:, self.grid_mapping]


class SelectElectrodesFromPY17N009:
    """
    Select the electrode contacts from the lower part of the ECoG grid excluding the two channels located in
    the auditory cortex (index 39 and 55).
    """
    def __init__(self):
        self.grid_mapping = [32, 48, 64, 80, 96, 112, 128, 144,
                             33, 49, 65, 81, 97, 113, 129, 145,
                             34, 50, 66, 82, 98, 114, 130, 146,
                             35, 51, 67, 83, 99, 115, 131, 147,
                             36, 52, 68, 84, 100, 116, 132, 148,
                             37, 53, 69, 85, 101, 117, 133, 149,
                             38, 54, 70, 86, 102, 118, 134, 150,
                             71, 87, 103, 119, 135, 151]

    def __len__(self):
        return len(self.grid_mapping)

    def __call__(self, data):
        return data[:, self.grid_mapping]

    def shape(self) -> Tuple[int, int]:
        return 1, 62


def get_trial_ids_for_day(corpus_day: Path) -> np.ndarray:
    """
    Accumulate all trial ids for a specific day in sorted order (sorted based on the run)
    """
    ids = []
    recordings = sorted(corpus_day.glob("*.hdf"))
    for recording in recordings:
        with h5py.File(recording, "r") as hf:
            ids.append(hf["trial_ids"][...])

    return np.concatenate(ids)


def count_trials_by_indices_list(indices_list: List[int]) -> int:
    """
    Return the number of trials by counting how often the stimuli values change.
    """
    counter = 0
    if len(indices_list) == 0:
        return counter

    counter += 1
    for i in range(1, len(indices_list)):
        if indices_list[i-1] != indices_list[i]:
            counter += 1

    return counter


class ZScoreNormalization:
    """
    Normalizes each channel according to predefined means and standard deviations.
    """
    def __init__(self, channel_means: np.ndarray, channel_stds: np.ndarray):
        self.channel_means = channel_means
        self.channel_stds = channel_stds

    def __call__(self, data):
        return (data - self.channel_means) / self.channel_stds


class GaussianNoise:
    """
    Add gaussian noise to each sample.
    """
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        return sample + np.random.normal(loc=self.mean, scale=self.std, size=sample.shape)
# endregion


# region CAR Filtering
class CommonAverageReferencing:
    """
    Subtract the global average within each electrode grid at each point in time from each electrode channel in that
    particular grid. Specified bad channels will not be included in computing the mean.
    Expects data to be in the form: T x E.
    """
    def __init__(self, exclude_channels: List[int], grids: List[np.ndarray], layout: np.ndarray):
        """
        :param exclude_channels: List of integer values (1 to 128) indicating which channels have been marked to be
        excluded from being used to be included in computing the global mean (e.g. bad channels)
        :param grids: List of 2D numpy arrays that specify the electrode alignment in each grid
        :param layout: List of integers that map each column in the data array to the channel in the grids list.
                       Example: [52, ...] if data has the 52nd channel in the first position.
        """
        self.grids = grids
        self.layout = layout

        # Construct the selection masks on where in the data to apply the CAR filtering.
        self.selection_masks_application = [np.isin(layout, grid) for grid in grids]

        # Construct the selection masks on where in the data to compute the global mean.
        self.selection_masks_computation = []
        for grid, mask_appl in zip(self.grids, self.selection_masks_application):
            mask_comp = mask_appl.copy()
            for excluded_channel in exclude_channels:
                if excluded_channel in grid:
                    mask_comp[np.argmax(layout == excluded_channel)] = False

            self.selection_masks_computation.append(mask_comp)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        result = data.copy()
        for mask_comp, mask_appl in zip(self.selection_masks_computation, self.selection_masks_application):
            means_per_time_point = np.mean(data[:, mask_comp], axis=1).reshape((-1, 1))
            means_per_time_point = np.tile(means_per_time_point, reps=(1, np.count_nonzero(mask_appl)))
            result[:, mask_appl] = result[:, mask_appl] - means_per_time_point

        return result

    def __repr__(self):
        """
        CommonAverageReferencing: 2 grids
        Grid 0:
            mask_appl: [...]
            mask_comp: [...]

        Grid 1:
            mask_appl: [...]
            mask_comp: [...]
        """
        string_info = f"CommonAverageReferencing ({len(self.grids)} grids):\n"
        for i, (m_appl, m_comp) in enumerate(zip(self.selection_masks_application, self.selection_masks_computation)):
            string_info += f"Grid {i}\n"
            string_info += f"\tmask_appl: [{', '.join(map(str, self.layout[m_appl]))}]\n"
            string_info += f"\tmask_comp: [{', '.join(map(str, self.layout[m_comp]))}]\n"

        return string_info
# endregion


# region Bad Channel Correction
class BadChannelCorrection:
    """
    Replace content of predefined bad channels with the mean from neighboring (non-bad) channels.
    """
    def __init__(self, bad_channels: List[int], grids: List[np.ndarray], layout: np.ndarray):
        """
        :param bad_channels: List of integer values (1 to 128) indicating which channels have been marked as
        bad channels
        :param grids: List of 2D numpy arrays that specify the numbering in each grid
        :param layout: List of integers that map each column in the data array to the channel in the grids list.
                       Example: [chan52, ...]
        """
        self.grids = grids
        self.layout = layout
        self.masks = [np.ones(grid.shape, dtype=bool) for grid in grids]
        self._construct_masks(bad_channels=bad_channels)
        self.footprint = self._get_footprint()
        self.patches = [(bad_channel, self._identify_neighbors(bad_channel)) for bad_channel in bad_channels]

        # Patches will store a tuple of the index where the bad channel is located in the data (according to the layout
        # information), and a list of indices from where in the data the mean will be computed.
        self.patches = [(np.where(self.layout == bad_channel)[0], self._find_neighbors_idx(neighbors))
                        for bad_channel, neighbors in self.patches]

    def _get_footprint(self):
        """
        Uses by default an 8-neighbor footprint. Override this function in case for a different neighboring set.
        """
        footprint = np.ones(9, dtype=bool).reshape((3, 3))
        footprint[1, 1] = False
        return footprint

    def _find_neighbors_idx(self, neighbors: List[int]) -> np.ndarray:
        return np.concatenate([np.where(self.layout == neighbor)[0] for neighbor in neighbors])

    def _identify_grid_index(self, channel: int) -> int:
        """
        Returns the index in which grid the channel was found.
        """
        for i, grid in enumerate(self.grids):
            if channel in grid:
                return i

        raise IndexError('Channel could not be found in given grids.')

    def _construct_masks(self, bad_channels: List[int]) -> None:
        """
        Construct binary masks for each grid that would reject all bad channels in that grid.
        """
        for bad_channel in bad_channels:
            grid_index = self._identify_grid_index(bad_channel)
            grid = self.grids[grid_index]
            row, col = np.where(grid == bad_channel)
            self.masks[grid_index][row, col] = False

    def _identify_neighbors(self, channel: int) -> List[int]:
        """
        Returns a list of all neighboring channels which should be considered for calculating the mean.
        """
        grid_index = self._identify_grid_index(channel)
        grid = self.grids[grid_index]
        row, col = np.where(grid == channel)
        mask = np.zeros(grid.shape, dtype=bool)
        mask[row, col] = True
        mask = binary_dilation(mask, structure=self.footprint)
        mask = mask & self.masks[grid_index]
        return grid[mask]

    def __call__(self, data: np.ndarray) -> np.ndarray:
        result = data.copy()
        for bad_channel_location, neighbors in self.patches:
            result[:, bad_channel_location] = np.mean(data[:, neighbors], axis=1).reshape((len(data), -1))

        return result

    def __len__(self):
        return len(self.patches)

    def __repr__(self):
        # Correcting 4 bad channels: 38 -> [37, 39, ...], ...
        items = []
        for bc_index, neighbor_indices in self.patches:
            bc = self.layout[bc_index].item()
            neighbors = [self.layout[neighbor_index] for neighbor_index in neighbor_indices]
            items.append(f"{bc} -> {str(neighbors)}")
        return f"Correcting {len(self.patches)} bad channels: {', '.join(items)}"
# endregion


# region Data loading
TrialIndices: TypeAlias = Tuple[str, int, int]


def load_data_for_days(days: List[str], corpus: Path, targets: str, x_transform: Optional[Callable] = None,
                       y_transform: Optional[Callable] = None,
                       id_transform: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ECoG activity, voice activity detection labels and trial ids from *.hdf archives across multiple days
    """
    x = []
    y = []
    ids = []
    for day in days:
        recordings = sorted((corpus / day).glob("*.hdf"))
        for recording in recordings:
            with h5py.File(recording, "r") as hf:
                ecog = hf["hga_activity"][...]
                targ = hf[targets][...]
                tids = hf["trial_ids"][...]

                # Prepare ECoG features
                if x_transform is not None:
                    ecog = x_transform(ecog)

                if y_transform is not None:
                    targ = y_transform(targ)

                if id_transform is not None:
                    tids = id_transform(tids)

                # Accumulate dataset
                x.append(ecog)
                y.append(targ)
                ids.append(tids)

    return np.concatenate(x), np.concatenate(y), np.concatenate(ids)


def save_data_to_hdf(filename: str, parameters: Dict[str, np.ndarray], overwrite: bool = False) -> bool:
    """
    Store timed aligned neural and acoustic data to .hdf container.
    """
    if os.path.exists(filename) and not overwrite:
        print(f'File {filename} already exists and overwrite is set to False. Training data is not stored.')
        return False

    with h5py.File(filename, 'w') as hf:
        for container_name, data in parameters.items():
            hf.create_dataset(container_name, data=data)

    return True


class BCI2000MatFile:
    """
    Wrapper class which makes the contents from the BCI2000 mat files more accessible.
    """
    def __init__(self, mat_filename: str):
        self.mat_filename = mat_filename
        self.mat = loadmat(self.mat_filename, simplify_cells=True)
        self.fs = self.mat['parameters']['SamplingRate']['NumericValue']

    def bad_channels(self) -> Optional[List[int]]:
        if 'bad_channels' in self.mat.keys():
            bad_channels = self.mat['bad_channels']
            if type(bad_channels) is np.ndarray:
                bad_channels = bad_channels.tolist()

            bad_channels = [bad_channels] if type(bad_channels) is not list else bad_channels
            bad_channels = [int(bad_channel[4:]) for bad_channel in bad_channels]
        else:
            bad_channels = None

        return bad_channels

    def contaminated_channels(self) -> Optional[List[int]]:
        if "contaminated_electrodes" in self.mat.keys():
            contaminated_electrodes = self.mat['contaminated_electrodes']
            if type(contaminated_electrodes) is int:
                contaminated_electrodes = [contaminated_electrodes, ]
            else:
                contaminated_electrodes = contaminated_electrodes.tolist()
            return contaminated_electrodes
        else:
            return None

    def trial_indices(self, min_trial_length: Optional[float] = None) -> List[TrialIndices]:
        stimuli = ExperimentMapping.extract_stimuli_values(self.mat)

        # Read stimulus code
        stimulus_code = self.mat['states']['StimulusCode']
        experiment_class = ExperimentMapping.get_experiment_class(mat_filename=self.mat_filename)
        experiment = experiment_class(stimulus_code, stimuli)
        trial_indices = experiment.get_trial_indices()

        # Round to 10ms
        trial_indices = [(label, round(start, -1), round(stop, -1)) for label, start, stop in trial_indices]

        if min_trial_length is not None:
            nb_min_samples = min_trial_length * self.fs
            trial_indices = [(label, start, max(stop, start + nb_min_samples)) for label, start, stop in trial_indices]

        return trial_indices

    def stimuli_indices(self) -> List[TrialIndices]:
        stimuli = ExperimentMapping.extract_stimuli_values(self.mat)

        # Read stimulus code
        stimulus_code = self.mat['states']['StimulusCode']
        experiment_class = ExperimentMapping.get_experiment_class(mat_filename=self.mat_filename)
        experiment = experiment_class(stimulus_code, stimuli)
        stimuli_indices = experiment.get_stimuli_indices()

        return stimuli_indices

    def signals(self) -> np.ndarray:
        signals = self.mat['signal']
        gain = self.mat['parameters']['SourceChGain']['NumericValue']
        return signals * gain

    def ordered_stimulus_codes(self) -> List[int]:
        stimulus_code = self.mat['states']['StimulusCode']

        # Get stimulus codes
        stimulus_codes = np.unique(stimulus_code).tolist()
        stimulus_codes = sorted(stimulus_codes)[1:]

        return stimulus_codes


class TensorDataset(Dataset):
    """
    Dataset class similar to pytorch's TensorDataset but allows transform operations.
    """
    def __init__(self, *tensors: torch.Tensor, transforms: Optional[List[Callable]] = None) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

        if transforms is None:
            self.transforms = [lambda x: x for _ in range(len(tensors))]
        else:
            self.transforms = transforms + [lambda x: x for _ in range(len(tensors) - len(transforms))]

    def __getitem__(self, index):
        return tuple(func(tensor[index]) for tensor, func in zip(self.tensors, self.transforms))

    def __len__(self):
        return self.tensors[0].size(0)
# endregion


# region Plotting utils
def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)


class MulticolorPatch(object):
    """
    Based on https://stackoverflow.com/a/67870930
    """
    def __init__(self, colors):
        self.colors = colors


class MulticolorPatchHandler(object):
    """
    Based on https://stackoverflow.com/a/67870930
    """
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle((width / len(orig_handle.colors) * i - handlebox.xdescent,
                                          -handlebox.ydescent),
                                         width / len(orig_handle.colors),
                                         height,
                                         facecolor=c,
                                         edgecolor='black'))

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


class BezierCurveLine(PathPatch):
    """Bezier Curve Line PathPatch"""

    def __init__(
        self,
        rad1: float,
        rad2: float,
        height_ratio: float = 0.5,
        **kwargs,
    ):
        """
        Parameters
        ----------
        rad1 : float
            Radian position1
        r1 : float
            Radius position1
        rad2 : float
            Radian position2
        r2 : float
            Radius position2
        height_ratio : float, optional
            Bezier curve height ratio parameter
        direction : int, optional
            `0`: No edge shape (Default)
            `1`: Directional(1 -> 2) arrow edge shape
            `-1`: Directional(1 <- 2) arrow edge shape
            `2`: Bidirectional arrow edge shape
        arrow_height : float, optional
            Arrow height size (Radius unit)
        arrow_width : float, optional
            Arrow width size (Degree unit)
        **kwargs : dict, optional
            Patch properties (e.g. `lw=1.0, hatch="//", ...`)
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html>
        """
        kwargs.update(fill=False)
        kwargs.setdefault("lw", 0.5)

        r1 = 1.0
        r2 = 1.0

        def bezier_paths(
            rad1: float,
            rad2: float,
            r1: float,
            r2: float,
            height_ratio: float = 0.5,
        ) -> list[tuple[MatplotlibPath.code_type, tuple[float, float]]]:
            if height_ratio >= 0.5:
                # Example1: height_ratio: 0.50 => r_ctl_pos: 0
                # Example2: height_ratio: 0.75 => r_ctl_pos: 25
                # Example3: height_ratio: 1.00 => r_ctl_pos: 50
                r_ctl_pos = 1.0 * (height_ratio - 0.5)
                rad_ctl_pos = (rad1 + rad2) / 2 + math.pi
            else:
                # Example1: height_ratio: 0.25 => r_ctl_pos: 25
                # Example2: height_ratio: 0.00 => r_ctl_pos: 50
                r_ctl_pos = 1.0 * (0.5 - height_ratio)
                rad_ctl_pos = (rad1 + rad2) / 2
            return [
                (MatplotlibPath.LINETO, (rad1, r1)),
                (MatplotlibPath.CURVE3, (rad_ctl_pos, r_ctl_pos)),
                (MatplotlibPath.LINETO, (rad2, r2)),
            ]

        path_data: list[tuple[MatplotlibPath.code_type, tuple[float, float]]] = []

        path_data.append((MatplotlibPath.MOVETO, (rad1, r1)))
        path_data.extend(bezier_paths(rad1, rad2, r1, r2, height_ratio))
        path_data.append((MatplotlibPath.LINETO, (rad2, r2)))

        verts, codes = [p[1] for p in path_data], [p[0] for p in path_data]
        bezier_arrow_line_path = MatplotlibPath(verts, codes, closed=True)  # type: ignore
        super().__init__(bezier_arrow_line_path, **kwargs)
# endregion


# region Network training
class StoreBestModel:
    """
    Store the best model (according to the validation loss or the average cost) at a dedicated location.
    The model will not be updated if the validation score is worse than any seen before.
    """
    def __init__(self, filename: str, info: Optional[dict] = None):
        self.current_best_score = np.inf
        self.filename = filename
        self.optional_info = info

    def update(self, model: nn.Module, score: float, info: Optional[dict] = None):
        if score < self.current_best_score:
            torch.save(model.state_dict(), self.filename)
            self.current_best_score = score
            self.optional_info = info
            print(f"Updated best model weights for a score of {score:.04f}.", flush=True)


class ContinuousSampling(Dataset):
    """
    Dataset class for sampling same size (time dimension) segments from a continuous stream of high-gamma activity.
    """
    def __init__(self, feature_files: List[str], n_timesteps: int, target_label: str = "ticc_labels",
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        """
        Initialize a dataset by randomly drawing a fixed number of segments from all trials
        :param feature_files: List of feature files that contain time-aligned neural and acoustic features
        :param n_timesteps: Number of frames in the temporal dimension (number of time steps)
        :param transform: Callable to apply on the features
        :param target_transform: Callable to apply on the targets
        """
        self.feature_files = feature_files
        self.n_total_trials = 0
        self.n_timesteps = n_timesteps
        self.target_label = target_label
        self.transform = transform
        self.target_transform = target_transform

        # Open each feature file
        self.fhs = [h5py.File(hdf_file, 'r') for hdf_file in feature_files]

        self.feature_dict = {}  # Will hold data in form (start_index, stop_index): fh
        for fh in self.fhs:
            n_trials = self._count_trials_in_recordings(fh)
            self.feature_dict[(self.n_total_trials, self.n_total_trials + n_trials)] = fh
            self.n_total_trials += n_trials

    @staticmethod
    def _count_trials_in_recordings(fh: h5py.File) -> int:
        trial_ids = fh["trial_ids"][...]
        split_indices = np.where(trial_ids[:-1] != trial_ids[1:])[0] + 1
        subsequences = np.split(trial_ids, split_indices)
        return len(subsequences)

    @staticmethod
    def _count_frames_in_recording(fh: h5py.File) -> int:
        return len(fh["trial_ids"][...])

    def get_total_num_of_frames(self) -> int:
        return sum(self._count_frames_in_recording(fh) for fh in self.fhs)

    def __del__(self):
        for fh in self.fhs:
            fh.close()

    def __len__(self):
        return self.n_total_trials

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        for (start, stop) in self.feature_dict.keys():
            if start <= index <= stop:
                fh = self.feature_dict[(start, stop)]
                hi = self._count_frames_in_recording(fh) - self.n_timesteps
                rn = np.random.randint(low=0, high=hi, size=1).item()

                hga = fh["hga_activity"][rn:rn + self.n_timesteps]
                lpc = fh[self.target_label][rn:rn + self.n_timesteps]

                if self.transform:
                    hga = self.transform(hga)
                if self.target_transform:
                    lpc = self.target_transform(lpc)

                return hga, lpc


class SequentialSpeechTrials(Dataset):
    """
    Dataset class that returns each trial individually, independent of the number of time steps (such as 15 seconds for
    sentences, or 2 seconds for single words, etc...).
    """
    def __init__(self, feature_files: List[str], transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, target_specifier: str = "ticc_labels"):
        self.feature_files = feature_files
        self.transform = transform
        self.target_transform = target_transform
        self.target_specifier = target_specifier

        # Open each feature file
        self.fhs = [h5py.File(hdf_file, 'r') for hdf_file in feature_files]
        self.nb_trials = [self._count_trials(fh['trial_ids'][...]) for fh in self.fhs]
        self.trial_labels = []
        self.trial_filename = []
        self.frame_counter = 0
        for fh, fname in zip(self.fhs, feature_files):
            self.frame_counter += len(fh['trial_ids'][...])
            trial_stimuli = self._squeeze_trial_ids(fh['trial_ids'][...])
            self.trial_labels.extend(trial_stimuli)
            self.trial_filename.extend([fname] * len(trial_stimuli))

        self.cumulative_length = 0
        self.feature_dict = {}  # Will hold data in form (start_index, stop_index): fh
        for nb_trials, fh in zip(self.nb_trials, self.fhs):
            self.feature_dict[(self.cumulative_length, self.cumulative_length + nb_trials)] = fh
            self.cumulative_length += nb_trials

    def __del__(self):
        for fh in self.fhs:
            fh.close()

    def __len__(self):
        return sum(self.nb_trials)

    @staticmethod
    def _count_trials(trial_ids: List[int]) -> int:
        return len(np.where(trial_ids[:-1] != trial_ids[1:])[0]) + 1

    @staticmethod
    def _find_indices_of_nth_subsequence(n: int, seq: np.ndarray) -> Tuple[int, int]:
        """
        Given a sequence of integer values, find the boundaries of the nth subsequence within that sequence. Example:
        seq: [4, 4, 4, 3, 3, 3, -3, -3, -3, 5, 5, 5, -5, -5, -5, 5, 5, 5], n=4 would result in (9, 12)

        Subsequences are zero-indexed.
        """
        take_nth_element = itemgetter(n)
        borders = (np.where(seq[:-1] != seq[1:])[0] + 1).tolist()
        borders = [0] + borders + [len(seq)]
        borders = tuple(pairwise(borders))
        start, stop = take_nth_element(borders)
        return start, stop

    @staticmethod
    def _squeeze_trial_ids(trial_ids: List[int]) -> List[int]:
        last_entry = trial_ids[0]
        result = [last_entry]
        for i in range(1, len(trial_ids)):
            if trial_ids[i] != last_entry:
                result.append(abs(trial_ids[i]))
                last_entry = trial_ids[i]

        return result

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        for (start, stop) in self.feature_dict.keys():
            if start <= index < stop:
                trial_ids = self.feature_dict[(start, stop)]['trial_ids'][...]
                trial_start, trial_stop = self._find_indices_of_nth_subsequence(index - start, trial_ids)

                hga = self.feature_dict[(start, stop)]['hga_activity'][trial_start:trial_stop]
                lpc = self.feature_dict[(start, stop)][self.target_specifier][trial_start:trial_stop]

                if self.transform:
                    hga = self.transform(hga)
                if self.target_transform:
                    lpc = self.target_transform(lpc)
                return hga, lpc

    def __repr__(self):
        covered_days = sorted(set([Path(feature_file).parent.name for feature_file in self.feature_files]))
        return f"SequentialSpeechTrials: {sum(self.nb_trials)} trials with {self.frame_counter} frames " \
               f"(total: {(self.frame_counter * 0.01) / 3600:.02f}h). Days covered: {', '.join(covered_days)}"

    def plot_trial(self, index: int, stimuli_map: Optional[Dict[int, str]] = None):
        hga, lpc = self[index]

        label = stimuli_map[self.trial_labels[index]] if stimuli_map is not None else str(self.trial_labels[index])
        ig, (ax_hga, ax_lpc) = plt.subplots(2, 1, figsize=(14, 8), num=1, clear=True)

        ax_hga.set_title(f"Label: {label}, Filename: {self.trial_filename[index]}", loc="left")
        hga_im = ax_hga.imshow(hga.T, aspect='auto', origin='lower', cmap='bwr', vmin=-4, vmax=4)
        ax_hga.set_xlim(0, len(hga))
        # ax_hga.set_xticks([])
        ax_hga.set_ylabel('Channel', labelpad=-18)
        ax_hga.set_yticks(np.linspace(0, hga.shape[1], 2, endpoint=True))
        ax_hga.set_yticklabels([1, hga.shape[1]])

        lpc_im = ax_lpc.imshow(lpc.T, aspect='auto', origin='lower', cmap='viridis')
        ax_lpc.set_xlim(0, len(lpc))
        # ax_lpc.set_xticks([])
        ax_lpc.set_ylabel('LPC coefficients', labelpad=-18)
        ax_lpc.set_yticks(np.linspace(0, lpc.shape[1] - 1, 2, endpoint=True))
        ax_lpc.set_yticklabels([1, 20])

        plt.show()


def visualize_vad_predictions(pred: np.ndarray, orig: np.ndarray, filename: Path):
    """
    Plot the original and the predicted curves of the VAD. Title indicates how many frames have been correctly
    classified.
    """
    fig, ax = plt.subplots(1, 1, num=1, clear=True)
    ax.plot(orig, c="black", linestyle="--")
    ax.plot(pred, c="orange")
    ax.set_xlim(0, len(orig))
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel("VAD")
    ax.set_xticks([0, 100])
    ax.set_xticklabels([0, 1])
    ax.set_title(f"Trial accuracy: {list(pred == orig).count(True) / len(pred) * 100:.2f}")
    plt.savefig(filename.as_posix(), dpi=72)
# endregion
