import numpy as np
import math
from scipy.fftpack import dct
from scipy.signal.windows import hann
from typing import Tuple, List


class MelFilterBank:
    def __init__(self, specSize, numCoefficients, sampleRate):
        numBands = int(numCoefficients)

        # Set up center frequencies
        minMel = 0
        maxMel = self.freqToMel(sampleRate / 2.0)
        melStep = (maxMel - minMel) / (numBands + 1)

        melFilterEdges = np.arange(0, numBands + 2) * melStep

        # Convert center frequencies to indices in spectrum
        centerIndices = list(
            map(lambda x: self.freqToBin(math.floor(self.melToFreq(x)), sampleRate, specSize), melFilterEdges))

        # Prepare matrix
        filterMatrix = np.zeros((numBands, specSize))

        # Construct matrix with triangular filters
        for i in range(numBands):
            start, center, end = centerIndices[i:i + 3]
            k1 = float(center - start)
            k2 = float(end - center)
            up = (np.array(range(start, center)) - start) / k1
            down = (end - np.array(range(center, end))) / k2

            filterMatrix[i][start:center] = up
            filterMatrix[i][center:end] = down

        # Save matrix and its best-effort inverse
        self.melMatrix = filterMatrix.transpose()
        self.melMatrix = self.makeNormal(self.melMatrix / self.normSum(self.melMatrix))

        self.melInvMatrix = self.melMatrix.transpose()
        self.melInvMatrix = self.makeNormal(self.melInvMatrix / self.normSum(self.melInvMatrix))

    def normSum(self, x):
        retSum = np.sum(x, axis=0)
        retSum[np.where(retSum == 0)] = 1.0
        return retSum

    def fuzz(self, x):
        return x + 0.0000001

    def freqToBin(self, freq, sampleRate, specSize):
        return int(math.floor((freq / (sampleRate / 2.0)) * specSize))

    def freqToMel(self, freq):
        return 2595.0 * math.log10(1.0 + freq / 700.0)

    def melToFreq(self, mel):
        return 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0)

    def toMelScale(self, spectrogram):
        return (np.dot(spectrogram, self.melMatrix))

    def fromMelScale(self, melSpectrogram):
        return (np.dot(melSpectrogram, self.melInvMatrix))

    def makeNormal(self, x):
        nanIdx = np.isnan(x)
        x[nanIdx] = 0

        infIdx = np.isinf(x)
        x[infIdx] = 0

        return (x)

    def toMels(self, spectrogram):
        return (self.toMelScale(spectrogram))

    def fromMels(self, melSpectrogram):
        return (self.fromMelScale(melSpectrogram))

    def toLogMels(self, spectrogram):
        return (self.makeNormal(np.log(self.fuzz(self.toMelScale(spectrogram)))))

    def fromLogMels(self, melSpectrogram):
        return (self.makeNormal(self.fromMelScale(np.exp(melSpectrogram))))


class EnergyBasedVad:
    """
    Energy based VAD computation. Should be equal to compute-vad from Kaldi.

    Arguments:
        log_mels (numpy array): log-Mels which get transformed into MFCCs
        mfcc_coeff (int): Number of MFCC coefficients (exclusive first one)
        energy_threshold (float): If this is set to s, to get the actual threshold we let m be
            the mean log-energy of the file, and use s*m + vad-energy-threshold (float, default = 0.5)
        energy_mean_scale (float): Constant term in energy threshold for MFCC0 for VAD
            (also see --vad-energy-mean-scale) (float, default = 5)
        frames_context (int): Number of frames of context on each side of central frame,
            in window for which energy is monitored (int, default = 0)
        proportion_threshold (float): Parameter controlling the proportion of frames within
            the window that need to have more energy than the threshold (float, default = 0.6)
        export_to_file (str): filename to export the VAD in .lab file format (readable with audacity)
    """
    def __init__(self, energy_threshold=4, energy_mean_scale=1, frames_context=5, proportion_threshold=0.6):
        self.vad_energy_threshold = energy_threshold
        self.vad_energy_mean_scale = energy_mean_scale
        self.vad_frames_context = frames_context
        self.vad_proportion_threshold = proportion_threshold
        self.mfcc_coeff = 13
        self.frame_shift = 0.01
        self.window_length = 0.05

    def from_wav(self, wav, sampling_rate=16000):
        # segment audio into windows
        window_size = int(sampling_rate * self.window_length)
        window_shift = int(sampling_rate * self.frame_shift)
        nb_windows = math.floor((len(wav) - window_size) / window_shift) + 1

        audio_segments = np.zeros((nb_windows, window_size))
        for win in range(nb_windows):
            start_audio = int(round(win * window_shift))
            stop_audio = int(round(start_audio + window_size))

            audio_segment = wav[start_audio:stop_audio]
            audio_segments[win, :] = audio_segment

        # create spectrogram from wav
        spectrogram = np.zeros((audio_segments.shape[0], window_size // 2 + 1), dtype='complex')

        win = hann(window_size)
        for w in range(nb_windows):
            a = audio_segments[w, :] / (2 ** 15)
            spec = np.fft.rfft(win * a)
            spectrogram[w, :] = spec

        mfb = MelFilterBank(spectrogram.shape[1], 40, sampling_rate)
        log_mels = (mfb.toLogMels(np.abs(spectrogram)))

        return self.from_log_mels(log_mels=log_mels)

    def from_log_mels(self, log_mels):
        self.mfccs = dct(log_mels)
        self.mfccs = self.mfccs[:, 0:self.mfcc_coeff + 2]

        return self.from_mfccs(self.mfccs)

    def from_mfccs(self, mfccs):
        self.mfccs = mfccs
        vad = self._compute_vad()
        return vad

    def _compute_vad(self):
        # VAD computation
        log_energy = self.mfccs[:, 0]
        output_voiced = np.empty(len(self.mfccs), dtype=bool)

        energy_threshold = self.vad_energy_threshold
        if self.vad_energy_mean_scale != 0:
            assert self.vad_energy_mean_scale > 0
            energy_threshold += self.vad_energy_mean_scale * np.sum(log_energy) / len(log_energy)

        assert self.vad_frames_context >= 0
        assert 0.0 < self.vad_proportion_threshold < 1

        for frame_idx in range(0, len(self.mfccs)):
            num_count = 0.0
            den_count = 0.0

            for t2 in range(frame_idx - self.vad_frames_context, frame_idx + self.vad_frames_context):
                if 0 <= t2 < len(self.mfccs):
                    den_count += 1
                    if log_energy[t2] > energy_threshold:
                        num_count += 1

            if num_count >= den_count * self.vad_proportion_threshold:
                output_voiced[frame_idx] = True
            else:
                output_voiced[frame_idx] = False

        return output_voiced

    def convert_vad_to_lab(self, filename, vad):
        last_i = None
        s = None
        r = ''

        for t, i in enumerate(vad):
            if last_i is None:
                last_i = i
                s = 0

            if i != last_i:
                e = t * self.frame_shift  # 10 ms
                r += '{:.2f}\t{:.2f}\t{}\n'.format(s, e, int(last_i))

                s = t * 0.01
                last_i = i

        r += '{:.2f}\t{:.2f}\t{}\n'.format(s, len(vad) * self.frame_shift, int(last_i))

        with open(filename, 'w+') as f:
            f.write(r)


class SpeechSegmentHistory:
    """
    Class which stores frames identified as speech in a ringbuffer and returns the complete speech segment as soon as
    the voice activity detection reports that the speech segment has ended.
    """
    def __init__(self, nb_features: int, buffer_size: int, context: int = 0):
        self.buffer = np.zeros((buffer_size, nb_features), dtype=np.float32)
        self.write_pointer = 0
        self.context = context
        self.speech_frame_counter = 0
        self.future_frame_counter = 0

    @staticmethod
    def _get_positions(read_pointer: int, write_pointer: int, buffer_size: int) -> List[int]:
        """
        Return an array of indices that specifies which elements in the array are being selected. Takes care of the
        buffer size and wraps around the end of the ringbuffer while remaining the order.
        """
        result = []
        while read_pointer != write_pointer:
            result.append(read_pointer)
            read_pointer = (read_pointer + 1) % buffer_size

        return result

    def insert(self, data: np.ndarray, speech_labels: np.ndarray) -> List[np.ndarray]:
        """
        Insert each frame into a ringbuffer and return a full speech segment after patient finished with speaking.
        :param data: Array of high-gamma frames
        :param speech_labels: Array of voice activity labels
        :return: List of speech segments that have been marked as completed, otherwise empty list
        """
        result = []
        for index in range(len(speech_labels)):
            frame = data[index, :]
            label = speech_labels[index]

            # Insert frame in ringbuffer
            self.buffer[self.write_pointer, :] = frame
            self.write_pointer = (self.write_pointer + 1) % len(self.buffer)

            # If we have seen a speech frame, increase the internal counter
            if label:
                self.speech_frame_counter += 1

            if not label and self.speech_frame_counter > 0:
                self.future_frame_counter += 1

                if self.future_frame_counter >= self.context:
                    stop = self.write_pointer if self.context > 0 else (self.write_pointer - 1) % len(self.buffer)
                    start = (stop - 2 * self.context - self.speech_frame_counter) % len(self.buffer)

                    positions = self._get_positions(start, stop, buffer_size=len(self.buffer))
                    result.append(self.buffer[positions])

                    # Reset counters
                    self.speech_frame_counter = 0
                    self.future_frame_counter = 0

        return result


class VoiceActivityDetectionSmoothing:
    """
    Class with considers each VAD label with respect to its neighbors (both past and future) to correct potential
    misclassification from the neural VAD model. Based on the number of context frames, which will be applied on both
    sides of each frame, this class will introduce a delay which is proportional to the number of future frames. In
    order to keep the alignment between speech and neural data, this class uses a ringbuffer and outputs frames only at
    times when it can make a reliable prediction on the whole window.
    """
    def __init__(self, nb_features: int, context_frames: int, proportion_threshold: float = 0.6, shift: float = 0.01):
        self.frameshift = shift
        self.nb_features = nb_features
        self.vad_context_frames = context_frames
        self.vad_proportion_threshold = proportion_threshold
        self.buffer_size = 2 * self.vad_context_frames + 1
        self.buffer = np.zeros((self.buffer_size, self.nb_features), dtype=np.float32)
        self.labels = np.zeros(self.buffer_size, dtype=bool)
        self.write_pointer = 2 * self.vad_context_frames
        self.read_pointer = 0

    def insert(self, data: np.ndarray, speech_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        resulting_label = np.zeros(len(speech_labels), dtype=bool)
        resulting_data = np.zeros((len(speech_labels), self.nb_features), dtype=np.float32)

        for i in range(len(speech_labels)):
            # Insert label into ringbuffer
            label = speech_labels[i]
            self.labels[self.write_pointer] = label

            # Insert high-gamma frame into ringbuffer
            frame = data[i]
            self.buffer[self.write_pointer, :] = frame

            # Compute output label based on the current window
            ratio = np.count_nonzero(self.labels) / self.buffer_size
            output_label = True if ratio >= self.vad_proportion_threshold else False
            resulting_label[i] = output_label
            resulting_data[i, :] = self.buffer[self.read_pointer, :]

            # Advance both read and write pointer
            self.write_pointer = (self.write_pointer + 1) % self.buffer_size
            self.read_pointer = (self.read_pointer + 1) % self.buffer_size

        return resulting_data, resulting_label

    def __repr__(self):
        return f"VAD Smoothing(Window size: {self.buffer_size * self.frameshift:.02f} s " \
               f"(introduced delay: {math.floor(self.buffer_size / 2) * self.frameshift} s), " \
               f"requires {self.vad_proportion_threshold * 100:.01f}% of frames to be speech)"
