import logging
import os
import zmq
import zmq.asyncio
import json
import time
import struct
import torch
import torch.nn as nn
import numpy as np
import ezmsg.core as ez
from pathlib import Path
from io import TextIOWrapper
from dataclasses import replace, field
from local.architecture import LstmState
from local.corticom import HighGammaExtractor
from ezmsg.eeg.eegmessage import TimeSeriesMessage
from typing import Optional, AsyncGenerator, List, Callable, Iterable


logger = logging.getLogger("units.py")


class ClosedLoopMessage(TimeSeriesMessage):
    """
    Extension to the TimeseriesMessage from the ezmsg.eeg module to contain an optional received_at variable which can
    be used to propagate the timestamp of previous messages to compute the processing time at a final unit.
    """
    received_at: Optional[float] = None
    previous_frames: Optional[float] = None


# region BCI2000 -> ZMQ connector
class ZMQConnectorSettings(ez.Settings):
    fs: int
    port: int = 5556
    address: str = 'localhost'


class ZMQConnectorState(ez.State):
    context: Optional[zmq.asyncio.Context] = None
    socket: Optional[zmq.asyncio.Socket] = None
    header: struct.Struct = struct.Struct('=BBB HH')
    topic: Optional[bytes] = None


class ZMQConnector(ez.Unit):
    """
    Temporary connector class for a better understanding of what is going on behind th scenes. Might be removed later.
    """
    SETTINGS: ZMQConnectorSettings
    STATE: ZMQConnectorState

    OUTPUT = ez.OutputStream(ClosedLoopMessage)

    def initialize(self) -> None:
        # Packet decoding
        self.STATE.topic = struct.Struct('=BBB').pack(4, 1, 2)

        # ZMQ networking
        address = f'tcp://{self.SETTINGS.address}:{self.SETTINGS.port}'
        self.STATE.context = zmq.asyncio.Context()
        self.STATE.socket = self.STATE.context.socket(zmq.SUB)
        self.STATE.socket.setsockopt(zmq.RCVHWM, 1)
        self.STATE.socket.connect(address)
        self.STATE.socket.subscribe(self.STATE.topic)

    def shutdown(self) -> None:
        self.STATE.socket.unsubscribe(self.STATE.topic)
        self.STATE.socket.close()
        self.STATE.context.destroy()

    def interpret_bytes(self, data: bytes) -> np.ndarray:
        descriptor, supplement, dtype, n_channels, n_samples = self.STATE.header.unpack(data[:self.STATE.header.size])
        array = np.frombuffer(data[self.STATE.header.size:], dtype=np.float32).reshape(n_channels, n_samples)
        array = np.transpose(array).astype(np.float64, order='C', copy=True)
        return array

    @ez.publisher(OUTPUT)
    async def process(self) -> AsyncGenerator:
        while not self.STATE.socket.closed:
            data = await self.STATE.socket.recv()
            data = self.interpret_bytes(data)
            yield self.OUTPUT, ClosedLoopMessage(data=data, fs=self.SETTINGS.fs, received_at=time.time())
# endregion


# region Feature extraction
Transforms = Optional[List[Callable]]


class HighGammaActivitySettings(ez.Settings):
    """
    Settings for the high-gamma activity unit. Window length and window shift need to be provided in seconds.
    """
    fs: int
    nb_electrodes: int
    window_length: float = 0.05
    window_shift: float = 0.01
    l_freq: int = 70
    h_freq: int = 170
    pre_transforms: Transforms = None
    post_transforms: Transforms = None


class HighGammaActivityState(ez.State):
    """
    Filter states of the high-gamma filter and the one for the first harmonic.
    """
    hg_extractor: Optional[HighGammaExtractor] = None


class HighGammaActivity(ez.Unit):
    """
    Unit for extraction of the high-gamma band in the range of 70 to 170 Hz. It also filters out the first harmonic
    of the line noise and provides options on how to extract features and how to transform them.
    """
    SETTINGS: HighGammaActivitySettings
    STATE: HighGammaActivityState

    INPUT: ez.InputStream = ez.InputStream(TimeSeriesMessage)
    OUTPUT: ez.OutputStream = ez.OutputStream(TimeSeriesMessage)

    def initialize(self) -> None:
        self.STATE.hg_extractor = HighGammaExtractor(
            fs=self.SETTINGS.fs, nb_electrodes=self.SETTINGS.nb_electrodes,
            window_length=self.SETTINGS.window_length, window_shift=self.SETTINGS.window_shift,
            pre_transforms=self.SETTINGS.pre_transforms, post_transforms=self.SETTINGS.post_transforms
        )

    @ez.publisher(OUTPUT)
    @ez.subscriber(INPUT)
    async def process(self, msg: TimeSeriesMessage) -> AsyncGenerator:
        features = self.STATE.hg_extractor.extract_features(msg.data)
        yield self.OUTPUT, replace(msg, data=features, fs=1/self.SETTINGS.window_shift)
# endregion


# region Neural Network Unit
class RecurrentNeuralDecodingModelSettings(ez.Settings):
    """
    Path to model weights reflects the path in which state_dict is stored. The model class variable refers to
    the pytorch module that can instantiate an architecture. If this model requires parameters to be set in its __ini__
    function, these can be set via the params optional dict.
    """
    path_to_model_weights: Optional[Path]
    model: nn.Module
    params: Optional[dict]


class RecurrentNeuralDecodingModelState(ez.State):
    """
    The neural decoding model state keeps track of the instantiated pytorch model and the device on which the model
    was deployed on.
    """
    frame_counter: int
    vad_model: Optional[nn.Module] = None
    device: Optional[str] = None
    H: Optional[LstmState] = None


class RecurrentNeuralDecodingModel(ez.Unit):
    """
    Ezmsg unit for the PyTorch model to do the VAD predictions
    """
    SETTINGS: RecurrentNeuralDecodingModelSettings
    STATE: RecurrentNeuralDecodingModelState

    INPUT = ez.InputStream(TimeSeriesMessage)
    OUTPUT = ez.OutputStream(TimeSeriesMessage)

    def initialize(self) -> None:
        """
        Instantiate pytorch model with specified parameters from the settings dataclass and deploys it on the cuda
        device. If no cuda device is accessible, it gets deployed on the cpu.
        """
        params = self.SETTINGS.params if self.SETTINGS.params is not None else dict()

        # Determine device
        self.STATE.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load pytorch architecture
        self.STATE.vad_model = self.SETTINGS.model(**params).to(self.STATE.device)
        if self.SETTINGS.path_to_model_weights is not None:
            self.STATE.vad_model.load_state_dict(torch.load(self.SETTINGS.path_to_model_weights,
                                                            map_location=self.STATE.device))
        self.STATE.vad_model.eval()
        self.STATE.H = self.STATE.vad_model.initial_states(batch_size=1, device=self.STATE.device)
        self.STATE.frame_counter = 0

    @ez.subscriber(INPUT)
    @ez.publisher(OUTPUT)
    async def decode(self, msg: TimeSeriesMessage) -> AsyncGenerator:
        """
        Message contains the extracted neural features.
        """
        ecog_tensor = torch.from_numpy(np.expand_dims(msg.data, 0)).float().to(self.STATE.device)
        predictions, self.STATE.H = self.STATE.vad_model(ecog_tensor, self.STATE.H)
        predictions = predictions.detach().cpu().numpy()
        predictions = np.squeeze(predictions, axis=0)
        predictions = np.argmax(predictions, axis=1)

        yield self.OUTPUT, replace(msg, data=predictions, fs=100, previous_frames=self.STATE.frame_counter)
        self.STATE.frame_counter += len(msg.data)
# endregion


# region Logging units
class LoggerSettings(ez.Settings):
    """
    General settings for speech related message loggers
    """
    filename: str
    overwrite: bool


class BinaryLoggerState(ez.State):
    """
    The binary logger state only contains the file descriptor which updates its position on each write operation
    """
    file_descriptor: Optional[TextIOWrapper] = None
    shape: Optional[Iterable[int]] = None


class BinaryLogger(ez.Unit):
    """
    Write the data field from a TimeSeriesMessage into a binary log file. For restoring the file contents, one dimension
    (for example the number of columns) and the data type need to be known. With this information, np.frombuffer() or
    np.fromfile can be used to restore all written data into a single array.

    Example:
        data = np.fromfile(log_filename, dtype=...).reshape((-1, ...)).astype(dtype, order='C', copy=True)
    """
    SETTINGS: LoggerSettings
    STATE: BinaryLoggerState
    INPUT: ez.InputStream = ez.InputStream(TimeSeriesMessage)

    def initialize(self) -> None:
        """
        Check if filename is a valid path (otherwise create the path) and check if the specified filename is already
        present. If that is the case and overwrite iin settings is set to False, an exception is raised.
        """
        filename = os.path.abspath(self.SETTINGS.filename)  # TODO use pathlib
        extension = os.path.basename(filename).split('.')[-1]
        storage_dir = os.path.dirname(filename)

        if not os.path.exists(storage_dir): os.makedirs(storage_dir)
        if os.path.exists(filename) and os.path.isfile(filename) and not self.SETTINGS.overwrite:
            raise PermissionError(f'The specified .{extension} file already exists and overwrite is disabled.')

        self.STATE.file_descriptor = open(filename, mode='wb')

    def shutdown(self) -> None:
        """
        Write the remaining data in the buffer to file and close the descriptor.
        """
        self.STATE.file_descriptor.flush()
        self.STATE.file_descriptor.close()

    @ez.subscriber(INPUT)
    async def write(self, message: TimeSeriesMessage) -> None:
        if self.STATE.shape is None:
            self.STATE.shape = list(message.data.shape)
            if len(self.STATE.shape) > 1:
                self.STATE.shape.pop(message.time_dim)
        self.STATE.file_descriptor.write(message.data.tobytes())


class VoiceActivityDetectionLoggerState(ez.State):
    """
    The VAD logger state contains the file descriptor which is used to write vad entries in the .lab file format
    (tab separated textfile containing the columns start, stop and label, see:
    https://manual.audacityteam.org/man/importing_and_exporting_labels.html for a detailed description).
    Furthermore, the other three fields are used to keep track of the running segment before it can be written to file.
    """
    file_descriptor: Optional[TextIOWrapper] = None
    current_voice_activity: Optional[int] = None
    frame_counter: int = 0
    last_change: int = 0


class VoiceActivityDetectionLogger(ez.Unit):
    """
    Logger unit which directly output acoustic samples into awave file format.
    """
    SETTINGS: LoggerSettings
    STATE: VoiceActivityDetectionLoggerState
    INPUT: ez.InputStream = ez.InputStream(ClosedLoopMessage)

    def initialize(self) -> None:
        """
        Setup up the file descriptor for logging the voice activity segments.
        """
        filename = os.path.abspath(self.SETTINGS.filename)
        storage_dir = os.path.dirname(filename)

        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        if os.path.exists(filename) and os.path.isfile(filename) and not self.SETTINGS.overwrite:
            raise PermissionError('The specified .lab file already exists and overwrite is disabled.')

        self.STATE.file_descriptor = open(filename, mode='w')

    def shutdown(self) -> None:
        """
        Make sure to write all remaining entries to file before closing.
        """
        self.STATE.file_descriptor.flush()
        self.STATE.file_descriptor.close()

    @ez.subscriber(INPUT)
    async def write(self, message: ClosedLoopMessage) -> None:
        """
        Message contains all the high-gamma activity frames that have been extracted from the voice activity
        detection module.
        """
        if self.STATE.current_voice_activity is None:
            self.STATE.current_voice_activity = message.data[0].item()

        for value in message.data:
            # Check if the voice activity has changed and requires that an entry needs to be stored to file
            if value.item() != self.STATE.current_voice_activity:
                # Determine start and stop times for the entry
                start = self.STATE.last_change * 0.01
                stop = self.STATE.frame_counter * 0.01

                # Store voice activity to file
                label = "Speech" if self.STATE.current_voice_activity == 1 else "Silence"
                if label == "Speech":
                    self.STATE.file_descriptor.write(f"{start:.02f}\t{stop:.02f}\t{label}\n")

                # Update last change to the current position
                self.STATE.last_change = self.STATE.frame_counter + 1

                # Update to the new voice activity state
                self.STATE.current_voice_activity = value.item()

            # Increase the frame counter
            self.STATE.frame_counter += 1
# endregion


# region Output Unit
class NAVIConnectorSettings(ez.Settings):
    """
    Specify if the unit should measure computing time.
    """
    output_time_complexity: bool


class NAVIConnectorState(ez.State):
    """
    State class tracking about the last state switch to only send out the SET STATE command once
    """
    state: Optional[int] = None
    time_measurements: List[float] = field(default_factory=list)


class NAVIConnector(ez.Unit):
    """
    Connector unit for communicating the VAD information to BCI2000
    """
    SETTINGS: NAVIConnectorSettings
    STATE: NAVIConnectorState

    INPUT = ez.InputStream(int)
    OUTPUT = ez.OutputStream(str)

    def shutdown(self) -> None:
        print("Shutting down NAVIConnector", flush=True)
        if self.SETTINGS.output_time_complexity:
            m = np.mean(self.STATE.time_measurements)
            s = np.std(self.STATE.time_measurements)
            print(f"Time measurements: {m:.04f} ± {s:.04f}")

    @ez.publisher(OUTPUT)
    @ez.subscriber(INPUT)
    async def process(self, msg: ClosedLoopMessage) -> AsyncGenerator:
        """
        If the state changes to 1, send out one message to BCI2000 to set the state on the testing machine. When no
        voice activity is detected anymore, set the state back to 0 once
        """
        if self.STATE.state is None:
            self.STATE.state = msg.data[0]

        if self.SETTINGS.output_time_complexity:
            current_time = time.time()
            delta = current_time - msg.received_at
            self.STATE.time_measurements.append(delta)

        for value in msg.data:
            if value != self.STATE.state:
                # New state is different from the last recorded one
                match value:
                    case 1:
                        click = json.dumps(dict(opcode="E", id=0, contents="SET STATE ControlClick 1"))
                    case _:
                        click = json.dumps(dict(opcode="E", id=0, contents="SET STATE ControlClick 0"))

                print(click)
                yield self.OUTPUT, click

            self.STATE.state = value
# endregion
