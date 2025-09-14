import os
import mne
import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import LeaveOneOut
from typing import Optional, Union, Tuple, List, Dict, Callable
from functools import partial, reduce
from scipy.signal import sosfilt_zi, sosfilt
from hga_optimized import WarmStartFrameBuffer, compute_log_power_features


logger = logging.getLogger("corticom.py")


class LeaveOneDayOut(LeaveOneOut):
    """
    Perform cross-validation on held-out recording sessions of a whole day.
    """
    def split(self, X, y=None, groups=None, start_with_day: Optional[str] = None):
        """
        Generate pairs of dates to distinguish days that will be used for training and the one that will be used for
        testing
        :param X: List of strings in the form "year_month_day" that indicate the recording days
        :param y: Not used, just here for compatability
        :param groups: Not used, just here for compatability
        :param start_with_day: If not None will rotate the list of days so that start_with_day will be first test day
        :return: Tuple containing one list of the days used for training and one string that indicates the testing day.
        """
        ordered_days = sorted(X)
        if start_with_day is not None:
            if start_with_day not in ordered_days:
                raise ValueError(f"The day {start_with_day} is not in the list of provided days {ordered_days}.")

            # Rotate sorted list to re-arrange days that test day will be the first day in the list
            while ordered_days[0] != start_with_day:
                ordered_days.append(ordered_days.pop(0))

        indices = np.arange(len(ordered_days))
        for test_index in self._iter_test_masks(indices, None, None):
            train_index = indices[np.logical_not(test_index)]
            train_day = [ordered_days[i] for i in train_index]
            test_day = ordered_days[np.argmax(test_index)]
            yield train_day, test_day


# region Trial indices
class Experiment(ABC):
    """
    Abstract class defining the interface for extracting trial segments from different experiment tasks.
    """
    def __init__(self, stimulus_code: np.ndarray, stimuli: Union[Dict[int, str], List[str]]):
        self.stimulus_code = stimulus_code
        self.stimuli = stimuli

        # Infer stimuli dict from the positions in the list, starting at index 1 to reserve 0 for not being a stimuli.
        if isinstance(self.stimuli, list):
            self.stimuli = {(index + 1): item for index, item in enumerate(self.stimuli)}

    def __repr__(self):
        return f'{self.__class__.__name__}(len: {len(self.stimulus_code)} samples, with {len(self.stimuli)} stimuli ' \
               f'across {len(self.get_trial_indices())} trials)'

    def _determine_trial_boundaries(self) -> List[Tuple[int, int]]:
        diff = np.where(self.stimulus_code[:-1] != self.stimulus_code[1:])[0] + 1
        return list(zip(diff[::], diff[1::]))

    @staticmethod
    def trial_indices_to_lab(filename: str, trial_indices: List[Tuple[str, int, int]], fs: int):
        with open(filename, 'w') as f:
            for label, start, stop in trial_indices:
                f.write(f'{start / fs:.03f}\t{stop / fs:.03f}\t{label}\n')

    @abstractmethod
    def get_trial_indices(self) -> List[Tuple[str, int, int]]:
        """
        Abstract method which return a list of tuples containing the label name, start and stop indices of each trial.
        """
        ...

    @abstractmethod
    def get_stimuli_indices(self) -> List[Tuple[str, int, int]]:
        """
        Abstract method which return a list of tuples containing the label name, start and stop indices of each
        stimulus presentation phase before a trial.
        """
        ...

    def get_webfm_baseline_windows(self, fs: int, length: float = 0.8) -> List[Tuple[str, int, int]]:
        """
        Method which return a list of tuples that contain 0.8 seconds pre-stimulus cue start to compute
        high-gamma baseline on.
        """
        trials = self.get_stimuli_indices()
        baseline_windows = [('BL', int(start - length * fs), start) for _, start, _ in trials]
        return baseline_windows

    def get_experiment_run_indices(self) -> Tuple[str, int, int]:
        """
        Returns a tuple of start and stop indices of the complete experiment run. This gets determined by the stimulus
        codes, since each recording could have data before and after the actual run.
        """
        trial_boundaries = self._determine_trial_boundaries()
        start = trial_boundaries[0][0]

        trials = self.get_trial_indices()
        stop = trials[-1][2]

        return 'Experiment run', start, stop

    @staticmethod
    def get_stimuli_values() -> Optional[list]:
        return None


class SyllableRepetition(Experiment):
    """
    Task in which syllables are audibly presented and after an acoustic hint the patient repeats the presented syllable.
    """
    @staticmethod
    def _swap_auditory_stimuli_codes(stimuli_codes: np.ndarray, trials: List[Tuple[int, int]]) -> np.ndarray:
        stimuli_presentation_segments = trials[::2]
        patient_speaking_segments = trials[1::2]

        new_stimuli_codes = stimuli_codes.copy()

        # Transfer stimuli codes from presentation to actual speaking
        for k, (start, stop) in enumerate(patient_speaking_segments):
            code = stimuli_codes[stimuli_presentation_segments[k][0]]
            new_stimuli_codes[start:stop] = code

        # Zero out stimuli codes from presentation
        for start, stop in stimuli_presentation_segments:
            new_stimuli_codes[start:stop] = 0

        return new_stimuli_codes

    @staticmethod
    def _determine_trial_length(trials: List[Tuple[int, int]]) -> int:
        start, stop = trials[1]
        return stop - start

    def get_trial_indices(self) -> List[Tuple[str, int, int]]:
        """
        Stimulus codes != 0 indicate the segments in which the syllable is acoustically presented. Afterwards, the
        stimulus code switches to 0 while the patient repeats the syllable.

        :return: List of tuples in which the first component represents the label of the stimuli that was used during
        acoustic presentation, followed by start and end indices of that segment.
        """
        trials = self._determine_trial_boundaries()

        # Append last trial which cannot identified through the difference method.
        trial_length = self._determine_trial_length(trials)
        trial_length = min(trial_length, len(self.stimulus_code))
        trials.append((trials[-1][1], trials[-1][1] + trial_length))

        stim_codes = self._swap_auditory_stimuli_codes(self.stimulus_code, trials)
        trials = [(self.stimuli[stim_codes[start]], start, stop) for (start, stop) in trials if stim_codes[start] != 0]
        return trials

    def _stimuli_extraction(self, entry_condition: Callable, exit_condition: Callable) -> List[Tuple[str, int, int]]:
        start = None
        label = None
        result = []
        for i in range(len(self.stimulus_code)):
            if entry_condition(self.stimulus_code[i]) and start is None:
                start = i
                label = self.stimuli[self.stimulus_code[i]]

            if exit_condition(self.stimulus_code[i]) and start is not None:
                result.append((label, start, i))
                start = None
                label = None

        return result

    def get_stimuli_indices(self) -> List[Tuple[str, int, int]]:
        entry_condition = partial(lambda stimulus_code: stimulus_code != 0)
        exit_condition = partial(lambda stimulus_code: stimulus_code == 0)
        return self._stimuli_extraction(entry_condition=entry_condition, exit_condition=exit_condition)


class NGSLSWordReading(Experiment):
    """
    Patient reading words from the NGSLS corpus. Each word is shown for a few seconds on screen, followed by a short
    inter-trial phase. Both phases have a slight temporal randomness added.
    """
    def _stimuli_extraction(self, entry_condition: Callable, exit_condition: Callable) -> List[Tuple[str, int, int]]:
        start = None
        label = None
        result = []
        for i in range(len(self.stimulus_code)):
            if entry_condition(self.stimulus_code[i]) and start is None:
                start = i
                label = self.stimuli[self.stimulus_code[i]]

            if exit_condition(self.stimulus_code[i]) and start is not None:
                result.append((label, start, i))
                start = None
                label = None

        return result

    def get_trial_indices(self) -> List[Tuple[str, int, int]]:
        return self.get_stimuli_indices()

    def get_stimuli_indices(self) -> List[Tuple[str, int, int]]:
        entry_condition = partial(lambda stimulus_code: stimulus_code != 0)
        exit_condition = partial(lambda stimulus_code: stimulus_code == 0)
        return self._stimuli_extraction(entry_condition=entry_condition, exit_condition=exit_condition)

    @staticmethod
    def get_stimuli_values() -> List[str]:
        stimuli_value = [
            'A', 'Able', 'About', 'Absolutely', 'Accept', 'Account',
            'Across', 'Action', 'Actually', 'Add', 'Address', 'Affect',
            'After', 'Afternoon', 'Again', 'Against', 'Age', 'Ago',
            'Agree', 'Ahead', 'Air', 'All', 'Allow', 'Almost',
            'Along', 'Already', 'Alright', 'Also', 'Although', 'Always',
            'Amount', 'And', 'Another', 'Answer', 'Any', 'Anybody',
            'Anyone', 'Anything', 'Anyway', 'Apply', 'Area', 'Around',
            'As', 'Ask', 'At', 'Available', 'Aware', 'Away',
            'Awful', 'Baby', 'Back', 'Bad', 'Bag', 'Base',
            'Basically', 'Be', 'Bear', 'Because', 'Become', 'Bed',
            'Before', 'Begin', 'Behind', 'Believe', 'Benefit', 'Bet',
            'Between', 'Big', 'Bill', 'Bit', 'Black', 'Bloody',
            'Board', 'Body', 'Book', 'Both', 'Bother', 'Bottom',
            'Box', 'Boy', 'Break', 'Breath', 'Brief', 'Bring',
            'Brother', 'Budget', 'Build', 'Bus', 'Business', 'But',
            'Buy', 'By', 'Call', 'Can', 'Car', 'Card',
            'Care', 'Carry', 'Case', 'Catch', 'Cause', 'Center',
            'Certain', 'Certainly', 'Chance', 'Change', 'Charge', 'Cheap',
            'Check', 'Child', 'Choice', 'Choose', 'Church', 'City',
            'Class', 'Clean', 'Clear', 'Clock', 'Close', 'Cold',
            'College', 'Color', 'Come', 'Comment', 'Committee', 'Community',
            'Company', 'Completely', 'Computer', 'Concern', 'Consider', 'Contact',
            'Continue', 'Control', 'Conversation', 'Cool', 'Copy', 'Correct',
            'Cost', 'Cough', 'Could', 'Council', 'Country', 'Couple',
            'Course', 'Cover', 'Create', 'Cut', 'Dad', 'Date',
            'Day', 'Deal', 'Dear', 'Decide', 'Decision', 'Definitely',
            'Degree', 'Depend', 'Detail', 'Develop', 'Development', 'Die',
            'Difference', 'Different', 'Difficult', 'Dinner', 'Discuss', 'Discussion',
            'Do', 'Doctor', 'Document', 'Dog', 'Dollar', 'Door',
            'Down', 'Draw', 'Drink', 'Drive', 'Drop', 'During',
            'Each', 'Early', 'Easy', 'Eat', 'Education', 'Effect',
            'Effort', 'Eight', 'Eighty', 'Either', 'Eleven', 'Else',
            'End', 'Enjoy', 'Enough', 'Especially', 'Even', 'Evening',
            'Ever', 'Every', 'Everybody', 'Everyone', 'Everything', 'Exactly',
            'Example', 'Expect', 'Experience', 'Explain', 'Extra', 'Eye',
            'Face', 'Fact', 'Fair', 'Fairly', 'Fall', 'Family',
            'Far', 'Father', 'Feel', 'Few', 'Fifteen', 'Fifty',
            'Figure', 'Fill', 'Find', 'Fine', 'Finish', 'Fire',
            'First', 'Fit', 'Five', 'Floor', 'Focus', 'Follow',
            'Food', 'Foot', 'For', 'Force', 'Forget', 'Form',
            'Forty', 'Forward', 'Four', 'Free', 'Friend', 'From',
            'Front', 'Full', 'Fun', 'Fund', 'Funny', 'Further',
            'Future', 'Game', 'General', 'Get', 'Girl', 'Give',
            'Go', 'Good', 'Government', 'Grade', 'Great', 'Group',
            'Grow', 'Guess', 'Guy', 'Hair', 'Half', 'Hand',
            'Hang', 'Happen', 'Happy', 'Hard', 'Hate', 'Have',
            'He', 'Head', 'Health', 'Hear', 'Help', 'Here',
            'High', 'Hit', 'Hold', 'Holiday', 'Home', 'Hope',
            'Hospital', 'Hot', 'Hour', 'House', 'How', 'Hundred',
            'Husband', 'I', 'Idea', 'If', 'Imagine', 'Important',
            'In', 'Include', 'Increase', 'Individual', 'Information', 'Instead',
            'Interest', 'Into', 'Involve', 'Issue', 'It', 'Item',
            'Itself', 'Job', 'Just', 'Keep', 'Kid', 'Kill',
            'Kind', 'Know', 'Lady', 'Land', 'Language', 'Large',
            'Last', 'Late', 'Laugh', 'Laughter', 'Law', 'Lead',
            'Learn', 'Least', 'Leave', 'Less', 'Let', 'Letter',
            'Level', 'Life', 'Light', 'Like', 'Line', 'List',
            'Listen', 'Little', 'Live', 'Load', 'Local', 'Long',
            'Look', 'Lose', 'Lot', 'Love', 'Lovely', 'Low',
            'Machine', 'Main', 'Major', 'Make', 'Man', 'Many',
            'Market', 'Marry', 'Mathematics', 'Matter', 'May', 'Maybe',
            'Mean', 'Meet', 'Member', 'Mention', 'Middle', 'Might',
            'Mile', 'Million', 'Mind', 'Mine', 'Minute', 'Miss',
            'Moment', 'Money', 'Month', 'More', 'Morning', 'Most',
            'Mother', 'Move', 'Movie', 'Much', 'Music', 'Must',
            'Myself', 'Name', 'National', 'Near', 'Need', 'Never',
            'New', 'News', 'Next', 'Nice', 'Night', 'Nine',
            'Nineteen', 'Ninety', 'No', 'Nobody', 'Noise', 'Normally',
            'Not', 'Note', 'Nothing', 'Notice', 'Now', 'Number',
            'Obviously', 'Of', 'Off', 'Offer', 'Office', 'Often',
            'Okay', 'Old', 'On', 'Once', 'One', 'Only',
            'Open', 'Opportunity', 'Or', 'Order', 'Other', 'Ought',
            'Out', 'Outside', 'Over', 'Own', 'Page', 'Paper',
            'Parent', 'Park', 'Part', 'Particular', 'Particularly', 'Party',
            'Pass', 'Past', 'Pause', 'Pay', 'People', 'Per',
            'Percent', 'Perhaps', 'Period', 'Person', 'Phone', 'Pick',
            'Picture', 'Piece', 'Place', 'Plan', 'Play', 'Please',
            'Plus', 'Point', 'Police', 'Policy', 'Position', 'Possible',
            'Pound', 'Power', 'Present', 'Press', 'Pretty', 'Price',
            'Probably', 'Problem', 'Process', 'Produce', 'Product', 'Program',
            'Project', 'Provide', 'Public', 'Pull', 'Purpose', 'Push',
            'Put', 'Quality', 'Question', 'Quick', 'Quite', 'Raise',
            'Rate', 'Rather', 'Read', 'Ready', 'Real', 'Realize',
            'Really', 'Reason', 'Receive', 'Record', 'Red', 'Relate',
            'Relationship', 'Remember', 'Report', 'Require', 'Research', 'Response',
            'Rest', 'Result', 'Review', 'Right', 'Ring', 'Road',
            'Room', 'Round', 'Rule', 'Run', 'Sale', 'Same',
            'Save', 'Say', 'School', 'Score', 'Second', 'See',
            'Seem', 'Sell', 'Send', 'Sense', 'Service', 'Set',
            'Seven', 'Seventy', 'Several', 'Shall', 'Share', 'She',
            'Shop', 'Short', 'Should', 'Show', 'Side', 'Sign',
            'Since', 'Sing', 'Single', 'Sister', 'Sit', 'Site',
            'Situation', 'Six', 'Sixty', 'Size', 'Sleep', 'Small',
            'So', 'Some', 'Somebody', 'Someone', 'Something', 'Sometimes',
            'Somewhere', 'Soon', 'Sorry', 'Sort', 'Sound', 'Speak',
            'Special', 'Specific', 'Spend', 'Staff', 'Stage', 'Stand',
            'Standard', 'Start', 'State', 'Statement', 'Stay', 'Step',
            'Stick', 'Still', 'Stop', 'Story', 'Straight', 'Street',
            'Strong', 'Student', 'Study', 'Stuff', 'Subject', 'Such',
            'Suggest', 'Summer', 'Support', 'Suppose', 'Sure', 'Surprise',
            'System', 'Table', 'Take', 'Talk', 'Tape', 'Tax',
            'Tea', 'Teach', 'Teacher', 'Team', 'Tell', 'Ten',
            'Tend', 'Term', 'Test', 'Text', 'Than', 'Thank',
            'That', 'The', 'Themselves', 'Then', 'There', 'Therefore',
            'They', 'Thing', 'Think', 'Thirty', 'This', 'Though',
            'Thousand', 'Three', 'Through', 'Throw', 'Till', 'Time',
            'To', 'Today', 'Together', 'Tomorrow', 'Tonight', 'Too',
            'Top', 'Totally', 'Toward', 'Town', 'Trade', 'Train',
            'Travel', 'Trouble', 'True', 'Try', 'Turn', 'Twelve',
            'Twenty', 'Two', 'Type', 'Unclear', 'Under', 'Understand',
            'Unless', 'Until', 'Up', 'Use', 'Usually', 'Value',
            'Various', 'Very', 'View', 'Visit', 'Vote', 'Wait',
            'Walk', 'Want', 'War', 'Watch', 'Water', 'Way',
            'We', 'Wear', 'Week', 'Weekend', 'Well', 'What',
            'Whatever', 'When', 'Where', 'Whether', 'Which', 'While',
            'White', 'Who', 'Whole', 'Why', 'Wife', 'Will',
            'Win', 'Window', 'Wish', 'With', 'Within', 'Without',
            'Woman', 'Wonder', 'Wonderful', 'Word', 'Work', 'World',
            'Worry', 'Worth', 'Would', 'Write', 'Wrong', 'Yeah',
            'Year', 'Yes', 'Yesterday', 'Yet', 'You', 'Young',
            'Yourself',
        ]
        return stimuli_value


class FiftyWordReading(Experiment):
    """
    Patient reading the keywords. Each word is shown for a few seconds on screen. Word presented for 1s.
    Fixation (Stimulus_Code 1) lasts 1.5s, ISI 1s. Entire duration 4.5s.
    For this experiment, trial and stimuli indices are the same.
    """
    def _stimuli_extraction(self, entry_condition: Callable, exit_condition: Callable) -> List[Tuple[str, int, int]]:
        start = None
        label = None
        result = []
        for i in range(len(self.stimulus_code)):
            if entry_condition(self.stimulus_code[i]) and start is None:
                start = i
                label = self.stimuli[self.stimulus_code[i]]

            if exit_condition(self.stimulus_code[i]) and start is not None:
                result.append((label, start, i))
                start = None
                label = None

        return result

    def get_trial_indices(self) -> List[Tuple[str, int, int]]:
        return self.get_stimuli_indices()

    def get_stimuli_indices(self) -> List[Tuple[str, int, int]]:
        entry_condition = partial(lambda stimulus_code: stimulus_code != 0)
        exit_condition = partial(lambda stimulus_code: stimulus_code == 0)
        return self._stimuli_extraction(entry_condition=entry_condition, exit_condition=exit_condition)

    @staticmethod
    def get_stimuli_values() -> List[str]:
        stimuli_value = [
            'Am', 'Are', 'Bad', 'Bring', 'Clean', 'Closer', 'Comfortable',
            'Coming', 'Computer', 'Do', 'Faith', 'Family', 'Feel', 'Glasses',
            'Going', 'Good', 'Goodbye', 'Have', 'Hello','Help', 'Here', 'Hope', 'How',
            'Hungry', 'I', 'Is', 'It', 'Like', 'Music', 'My', 'Need', 'No',
            'Not', 'Nurse', 'Okay', 'Outside', 'Please', 'Right', 'Success',
            'Tell', 'That', 'They', 'Thirsty', 'Tired', 'Up', 'Very', 'What',
            'Where', 'Yes', 'You', 'Silence',
        ]
        return stimuli_value


class ExperimentMapping(dict):
    """
    Map experiment names to Experiment class for extracting trial indices.
    """
    def __init__(self):
        super().__init__()
        mapping = {
            'SyllableRepetition': SyllableRepetition,
            'NGSLS': NGSLSWordReading,
            '50word': FiftyWordReading,
        }
        self.update(mapping)

    @staticmethod
    def get_experiment_class(mat_filename: str) -> Optional[Experiment]:
        """
        Based on the filename, return the appropriate experiment class which can be used to extract trial indices.
        """
        filename = os.path.basename(mat_filename)
        mapping = ExperimentMapping()

        for key in mapping.keys():
            if key in filename:
                return mapping[key]

        return None

    @staticmethod
    def extract_stimuli_values(mat: dict) -> List[str]:
        """
        Helper function to extract stimuli values from a loaded .mat file. This function takes care that a list of
        stimuli values is returned.
        """
        stimuli = mat['parameters']['Stimuli']['Value']
        if stimuli.ndim == 1:
            return [stimuli[0]]
        else:
            return stimuli[0].tolist()
# endregion


# region High-Gamma Activity extractor based on Herff et al. (2015) Brain2Test
class HighGammaExtractor:
    """
    Base class for the feature extraction that provides common functionalities for both the online and the offline
    computation.
    """
    def __init__(self, fs, nb_electrodes, window_length=0.05, window_shift=0.01, l_freq: int = 70, h_freq: int = 170,
                 pre_transforms: Optional[List[Callable]] = None, post_transforms: Optional[List[Callable]] = None):
        self.fs = fs
        self.nb_electrodes = nb_electrodes
        self.window_length = window_length
        self.window_shift = window_shift
        self.model_order = 4
        self.step_size = 5
        self.pre_transform = pre_transforms
        self.post_transform = post_transforms
        self.framebuffer = WarmStartFrameBuffer(frame_length=window_length, frame_shift=window_shift, fs=fs,
                                                nb_channels=nb_electrodes)

        if self.pre_transform is not None:
            self.pre_transform = self._compose_functions(*pre_transforms)
        if self.post_transform is not None:
            self.post_transform = self._compose_functions(*post_transforms)

        if not ((60 < l_freq < 120) or (120 < h_freq < 180)):
            logger.warning('l_freq and h_freq seem not to be in the recommended ranges!!')

        # Initialize filters and filter states
        iir_params = {'order': 8, 'ftype': 'butter'}
        self.hg_filter = self.create_filter(fs, l_freq, h_freq, method='iir', iir_params=iir_params)["sos"]
        self.fh_filter = self.create_filter(fs, 122, 118, method='iir', iir_params=iir_params)["sos"]

        hg_state = sosfilt_zi(self.hg_filter)
        fh_state = sosfilt_zi(self.fh_filter)

        self.hg_state = np.repeat(hg_state, nb_electrodes, axis=-1).reshape([hg_state.shape[0], hg_state.shape[1], -1])
        self.fh_state = np.repeat(fh_state, nb_electrodes, axis=-1).reshape([fh_state.shape[0], fh_state.shape[1], -1])

    @staticmethod
    def _compose_functions(*functions):
        return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

    @staticmethod
    def create_filter(sr, l_freq, h_freq, method='fir', iir_params=None):
        iir_params, method = mne.filter._check_method(method, iir_params)
        filt = mne.filter.create_filter(None, sr, l_freq, h_freq, 'auto', 'auto',
                                        'auto', method, iir_params, 'zero', 'hamming', 'firwin', verbose=30)
        return filt

    def extract_features(self, data: np.ndarray):
        # Apply pre-transforms
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Extract high-gamma activity
        data, self.hg_state = sosfilt(self.hg_filter, data, axis=0, zi=self.hg_state)
        data, self.fh_state = sosfilt(self.fh_filter, data, axis=0, zi=self.fh_state)

        # Compute power features
        data = self.framebuffer.insert(data)
        data = compute_log_power_features(data, self.fs, self.window_length, self.window_shift)

        # Apply post-transform
        if self.post_transform is not None:
            data = self.post_transform(data)
        return data
# endregion
