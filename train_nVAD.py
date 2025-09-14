import argparse
import logging
import numpy as np
import os
import torch
import torch.nn as nn
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from local.architecture import UnidirectionalVoiceActivityDetector
from torchvision.transforms import Compose
from local.utils import get_trial_ids_for_day, ContinuousSampling
from local.utils import compute_trial_based_error
from local.utils import SelectElectrodesOverSpeechAreas, SequentialSpeechTrials
from local.corticom import LeaveOneDayOut


logger = logging.getLogger("train_nVAD.py")


# region Training configuration
nb_epochs = 10
batch_size = 32
num_workers = 4
# endregion


def main(output_dir: Path, corpus: Path, dev_day: str, target_label: str = "ticc_labels"):
    # region Train setup
    torch.set_default_dtype(torch.float32)

    # Get all precomputed feature files per day
    feature_files = list(corpus.rglob('*/*.hdf'))
    feature_files = [f for f in feature_files if f.parent.name != dev_day]
    groups_by_day = defaultdict(list)
    for feature_file in feature_files:
        day = feature_file.parent.name
        groups_by_day[day].append(feature_file)

    kf = LeaveOneDayOut()
    for fold, (train_days, test_day) in enumerate(kf.split(groups_by_day.keys(), start_with_day="2022_11_29")):
        logger.info(f"Processing test day {test_day}")

        summary_writer = SummaryWriter(log_dir=(output_dir / f"Day_{fold + 1:02d}" / "tensorboard").as_posix())
        os.makedirs(output_dir / f"Day_{fold + 1:02d}" / "dev", exist_ok=True)

        # Organize feature files in train, test and validation sets
        tr_files = sorted([f.as_posix() for f in feature_files if f.parent.name in train_days])
        va_files = sorted([f.as_posix() for f in (corpus / dev_day).glob("*.hdf")])
        te_files = sorted([f.as_posix() for f in feature_files if f.parent.name == test_day])

        # Initialize datasets
        input_transforms = Compose([
            SelectElectrodesOverSpeechAreas(),
            torch.tensor,
        ])

        tr_dataset = ContinuousSampling(feature_files=tr_files, n_timesteps=400, transform=input_transforms,
                                        target_label=target_label)
        va_dataset = SequentialSpeechTrials(feature_files=va_files, target_specifier=target_label,
                                            transform=input_transforms)
        te_dataset = SequentialSpeechTrials(feature_files=te_files, target_specifier="acoustic_labels",
                                            transform=input_transforms)
        logger.info(f"Total number of samples in the training set: {tr_dataset.get_total_num_of_frames()}")

        # Initialize the dataloader for all three datasets
        dataloader_params = dict(num_workers=num_workers, pin_memory=True)
        tr_dataloader = DataLoader(tr_dataset, **dataloader_params, shuffle=True, batch_size=batch_size)
        va_dataloader = DataLoader(va_dataset, **dataloader_params, shuffle=False, batch_size=1)
        te_dataloader = DataLoader(te_dataset, **dataloader_params, shuffle=False, batch_size=1)

        # Setup nVAD architecture
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Setting device to: {device}")

        model = UnidirectionalVoiceActivityDetector(nb_electrodes=64, nb_layer=2, nb_hidden_units=100)

        nb_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total number of trainable parameters of the {type(model).__name__} model: {nb_train_params:,}")

        # Setup nVAD training
        optim = torch.optim.Adam(model.parameters(), lr=3e-4)
        cfunc = nn.CrossEntropyLoss()  # Cost function

        model.to(device)

        # Train the model
        best_model_weights = output_dir / f"Day_{fold + 1:02d}" / "best_model.pth"
        model.optimize(nb_epochs, tr_dataloader, va_dataloader, cfunc, optim, device, 50, best_model_weights)
        model.load_state_dict(torch.load(best_model_weights.as_posix()))

        # Compute performance on the test set
        pred, orig = model.predict(te_dataloader, device=device)

        trial_ids = get_trial_ids_for_day(corpus / test_day)
        error = compute_trial_based_error(pred, orig, trial_ids)
        np.save(output_dir / f"Day_{fold + 1:02d}" / "result.npy", np.vstack([pred, orig, trial_ids]))

        logger.info(f"Finished processing test day {test_day} (error: {np.median(error):.02f} sec)")
        # endregion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for the neural VAD model.")
    parser.add_argument("out_dir", help="Path to the output folder.")
    parser.add_argument("corpus", help="Path where the TICC labels were written to disc.")
    parser.add_argument("dev", help="String of the development day in the format YYYY-MM-DD.")
    parser.add_argument("target", default="ticc_labels", help="Target label used for the train set.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f'python train_nVAD.py {args.out_dir} {args.corpus} {args.dev}')
    main(output_dir=Path(args.out_dir), corpus=Path(args.corpus), dev_day=args.dev, target_label=str(args.target))
