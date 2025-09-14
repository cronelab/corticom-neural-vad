import argparse
import configparser
import os
import sys
import logging
import json
import ezmsg.core as ez
import numpy as np
from pathlib import Path
from local.units import ZMQConnector, ZMQConnectorSettings, NAVIConnector, NAVIConnectorSettings
from local.units import HighGammaActivity, HighGammaActivitySettings
from local.units import RecurrentNeuralDecodingModel, RecurrentNeuralDecodingModelSettings
from local.units import BinaryLogger, VoiceActivityDetectionLogger, LoggerSettings
from local.utils import ZScoreNormalization, CommonAverageReferencing
from local.utils import SelectElectrodesFromBothGrids, SelectElectrodesOverSpeechAreas
from local.architecture import UnidirectionalVoiceActivityDetector
from ezmsg.websocket import WebsocketClient, WebsocketSettings
from typing import Iterable, Optional, Tuple, Any


logger = logging.getLogger('decode_online.py')


# region BCI System
class CorticomVADSystemSettings(ez.Settings):
    """
    General settings for running the VAD system online
    """
    destination_dir: Path
    address: str
    fs: int
    package_size: int
    bad_channels: Optional[Iterable] = None
    vad_model_weights: Optional[Path] = None
    normalization_statistics: Optional[Path] = None
    print_time_complexity: bool = False


class CorticomVADSystem(ez.System):
    """
    Real-time VAD system
    """
    CONNECTOR = ZMQConnector()
    FEATURE_EXTRACTOR = HighGammaActivity()
    VAD_MODEL = RecurrentNeuralDecodingModel()
    NAVI_CONNECTOR = NAVIConnector()
    # BCI2000 = WebsocketClient()

    # Logging units
    RAW_LOGGER = BinaryLogger()
    HGA_LOGGER = BinaryLogger()
    VAD_LOGGER = VoiceActivityDetectionLogger()
    RES_LOGGER = BinaryLogger()

    # System settings
    SETTINGS: CorticomVADSystemSettings

    def configure_feature_transforms(self) -> Tuple[Any, Any, int]:
        """
        Configure pre- and post-transform lists to extract high-gamma features.
        """
        select_both_grids = SelectElectrodesFromBothGrids()
        pre_transforms = [select_both_grids, ]

        # Apply CAR filter
        speech_grid = np.flip(np.arange(64, dtype=np.int16).reshape((8, 8)) + 1, axis=0)
        motor_grid = np.flip(np.arange(64, dtype=np.int16).reshape((8, 8)) + 65, axis=0)
        layout = np.arange(128) + 1
        car = CommonAverageReferencing(exclude_channels=[19, 38, 48, 52], grids=[speech_grid, motor_grid],
                                       layout=layout)
        pre_transforms.append(car)

        # Select only relevant electrodes
        channel_selection = SelectElectrodesOverSpeechAreas()
        pre_transforms.append(channel_selection)

        if self.SETTINGS.normalization_statistics is None:
            logger.info("Found no normalization data. Going to use zero-mean and unit-variance.")
            channel_means = np.zeros(len(channel_selection), dtype=np.float32)
            channel_stds = np.ones(len(channel_selection), dtype=np.float32)
        else:
            logger.info(f"Found normalizations statistics in {self.SETTINGS.normalization_statistics.as_posix()}.")
            statistics = np.load(self.SETTINGS.normalization_statistics.as_posix())
            channel_means = statistics[0, :]
            channel_stds = statistics[1, :]

        post_transforms = ZScoreNormalization(channel_means=channel_selection(channel_means.reshape((1, -1))),
                                              channel_stds=channel_selection(channel_stds.reshape((1, -1))))

        return pre_transforms, post_transforms, len(channel_selection)

    def configure(self) -> None:
        # Configure the ZMQ connector for communication with the amplifier
        self.CONNECTOR.apply_settings(
            ZMQConnectorSettings(
                fs=self.SETTINGS.fs, address=self.SETTINGS.address, port=5556
            )
        )

        # Settings for extracting high-gamma activity
        pre_transforms, post_transforms, nb_features = self.configure_feature_transforms()
        self.FEATURE_EXTRACTOR.apply_settings(
            HighGammaActivitySettings(
                fs=self.SETTINGS.fs,
                nb_electrodes=nb_features,
                pre_transforms=pre_transforms,
                post_transforms=[post_transforms]
            )
        )

        # Initialize speech filtering unit
        logger.info(f"VAD model weights: {self.SETTINGS.vad_model_weights}")
        nb_electrodes = len(SelectElectrodesOverSpeechAreas())
        self.VAD_MODEL.apply_settings(
            RecurrentNeuralDecodingModelSettings(
                path_to_model_weights=self.SETTINGS.vad_model_weights,
                model=UnidirectionalVoiceActivityDetector,
                params=dict(nb_electrodes=64, nb_layer=2, nb_hidden_units=100)
            )
        )

        self.NAVI_CONNECTOR.apply_settings(
            NAVIConnectorSettings(
                output_time_complexity=self.SETTINGS.print_time_complexity
            )
        )

        # Connect to BCI2000
        # self.BCI2000.apply_settings(
        #     WebsocketSettings(host="Corticom-Testing", port=80)
        # )

        # Configure logging units
        # RAW logger: samples x 64
        # HGA logger: samples x 64
        # VAD logger: one detected speech segment per line (start, stop, number of frames)
        # Res logger: Useful to compare results between online and offline detection
        raw_logger_filename = os.path.join(self.SETTINGS.destination_dir, 'log.raw.f64')
        hga_logger_filename = os.path.join(self.SETTINGS.destination_dir, 'log.hga.f64')
        vad_logger_filename = os.path.join(self.SETTINGS.destination_dir, 'log.vad.lab')
        res_logger_filename = os.path.join(self.SETTINGS.destination_dir, 'log.res.i64')

        self.RAW_LOGGER.apply_settings(LoggerSettings(filename=raw_logger_filename, overwrite=True))  # Raw ECoG signals
        self.HGA_LOGGER.apply_settings(LoggerSettings(filename=hga_logger_filename, overwrite=True))  # High-y features
        self.VAD_LOGGER.apply_settings(LoggerSettings(filename=vad_logger_filename, overwrite=True))  # VAD segments
        self.RES_LOGGER.apply_settings(LoggerSettings(filename=res_logger_filename, overwrite=True))  # results logger

    # Define Connections
    def network(self) -> ez.NetworkDefinition:
        return (
            # Main route
            (self.CONNECTOR.OUTPUT, self.FEATURE_EXTRACTOR.INPUT),
            (self.FEATURE_EXTRACTOR.OUTPUT, self.VAD_MODEL.INPUT),
            (self.VAD_MODEL.OUTPUT, self.NAVI_CONNECTOR.INPUT),
            # (self.NAVI_CONNECTOR.OUTPUT, self.BCI2000.INPUT),

            # Connect the waveform generation component both with the loudspeaker and write acoustic samples to file
            (self.CONNECTOR.OUTPUT, self.RAW_LOGGER.INPUT),
            (self.FEATURE_EXTRACTOR.OUTPUT, self.HGA_LOGGER.INPUT),
            (self.VAD_MODEL.OUTPUT, self.VAD_LOGGER.INPUT),
            (self.VAD_MODEL.OUTPUT, self.RES_LOGGER.INPUT),
        )
# endregion


def main(settings: CorticomVADSystemSettings) -> None:
    """
    Start the online system. Terminate with Ctrl-C.
    """
    system = CorticomVADSystem(settings)
    ez.run_system(system)


def collect_settings(settings_filename: str, run_name: str, time_comp: bool = False) -> CorticomVADSystemSettings:
    """
    Extract all necessary fields for the VADSystemSettings class from the config file
    """
    settings_config = configparser.ConfigParser()
    settings_config.read(settings_filename)

    # Load path to model weights if provided, otherwise None
    model_weights_path = settings_config.get("VAD", "model_weights")
    model_weights_path = None if model_weights_path == "" else Path(model_weights_path)

    # Load bad channels if present, otherwise None
    bad_channels_entry = settings_config.get("VAD", "bad_channels")
    bad_channels = None if bad_channels_entry == "" else json.loads(bad_channels_entry)

    # Load path to normalization statistics if provided, otherwise None
    normalization_statistics_entry = settings_config.get("VAD", "initial_normalization_statistics")
    normalization_statistics = None if normalization_statistics_entry == "" else Path(normalization_statistics_entry)

    # Set destination dir
    destination_dir = Path(settings_config.get("VAD", "base_out_dir"))
    destination_dir = destination_dir / settings_config.get("VAD", "session") / run_name

    settings = CorticomVADSystemSettings(
        destination_dir=destination_dir,
        address=settings_config.get("VAD", "address"),
        fs=settings_config.getint("VAD", "fs"),
        package_size=settings_config.getint("VAD", "package_size"),
        bad_channels=bad_channels,
        vad_model_weights=model_weights_path,
        normalization_statistics=normalization_statistics,
        print_time_complexity=time_comp
    )

    return settings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online VAD system")
    parser.add_argument("config", help="Path of the config file on how to set up the BCI system.")
    parser.add_argument("--run", default="test_run", help="Name of the run folder")
    parser.add_argument("--overwrite", default=False, action="store_true",
                        help="Specify if that run already exists if it should be overwritten.")
    parser.add_argument("--time", default=False, action="store_true",)

    args = parser.parse_args()
    settings = collect_settings(args.config, args.run, args.time)
    try:
        os.makedirs(settings.destination_dir, exist_ok=args.overwrite)
    except FileExistsError:
        logger.error("The file path of the destination directory already exists and the --overwrite flag is not set.")
        exit(1)

    # initialize logging handler
    log_filename = os.path.join(settings.destination_dir, "log.run.txt")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s",
                        datefmt="%d.%m.%y %H:%M:%S",
                        handlers=[logging.FileHandler(log_filename, "w+"), logging.StreamHandler(sys.stderr)])

    overwrite = "--overwrite" if args.overwrite else ""
    time_comp = "--time" if args.time else ""
    logger.info(f"python decode_online.py {args.config} --run {args.run} {overwrite} {time_comp}")
    logger.info(f"Setting destination dir to {settings.destination_dir}")

    main(settings)
