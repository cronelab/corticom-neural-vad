import argparse
import logging
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision.transforms import Compose
from local.architecture import LeNetBasedVAD
from local.utils import SelectElectrodesOverSpeechAreas, load_data_for_days
from local.utils import compute_trial_based_error
from corticom.auxilary.evaluation import LeaveOneDayOut
from corticom.speech.external.hga_optimized import stack_features_spatio_temporal
from torch.utils.data import DataLoader
from local.utils import TensorDataset, GaussianNoise as Noise
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("baseline_computations.py")


# region Training configuration for Logistic Regression
model_order = 6  # Context of 300ms
step_size = 5    # Skip frames to reduce overlapping
n_worker = 4
batch_size = 32
# endregion


def main(corpus_root: Path, n_epochs: int, dev_day: str, output_dir: Path, target_label: str = "acoustic_labels"):
    recording_days = sorted([f.name for f in corpus_root.iterdir() if f.is_dir() and f.name != dev_day])

    # region Do cross validation
    channel_selector = SelectElectrodesOverSpeechAreas()

    # Create output directories if not present
    os.makedirs(output_dir / "lr", exist_ok=True)
    os.makedirs(output_dir / "cnn", exist_ok=True)

    kf = LeaveOneDayOut()
    for fold, (train_days, test_day) in enumerate(kf.split(recording_days, start_with_day=recording_days[0])):
        logger.info(f"Processing fold {fold + 1} with {test_day} as test day.")

        # Gather training data
        x_transform = Compose([channel_selector,
                               lambda x: np.array(stack_features_spatio_temporal(x, model_order, step_size, 8, 8)),
                               np.squeeze])
        y_transform = Compose([lambda x: x[model_order * step_size:], ])
        valid_day = train_days[-1]
        train_days = train_days[:-1]
        logger.info(f"Train days: {train_days}, valid day: {valid_day}")

        x_train, y_train, _ = load_data_for_days(train_days, corpus_root, target_label,
                                                 x_transform=x_transform, y_transform=y_transform)

        # Gather validation data
        x_val, y_val, val_ids = load_data_for_days([valid_day, ], corpus_root, target_label,
                                                   x_transform=x_transform, y_transform=y_transform,
                                                   id_transform=lambda x: x[model_order * step_size:])

        # Gather test data
        x_test, y_test, test_ids = load_data_for_days([test_day, ], corpus_root, "acoustic_labels",
                                                   x_transform=x_transform, y_transform=y_transform,
                                                   id_transform=lambda x: x[model_order * step_size:])

        # Compute VAD with logistic regression
        est = LogisticRegression(penalty="l1", solver="saga")
        est.fit(x_train.reshape(len(x_train), -1), y_train)
        with open(output_dir / "lr" / f"fold_{fold + 1:02d}.pkl", 'wb') as fh:
            pickle.dump(est, fh)

        pred = est.predict(x_test.reshape(len(x_test), -1))
        lr_data = np.vstack([pred, y_test, test_ids])
        logger.info(f"Finished Logistic Regression training for fold {fold + 1}. "
                    f"Error: {np.median(compute_trial_based_error(lr_data[0, :], lr_data[1, :], lr_data[2, :])):.02f}")

        np.save(output_dir / "lr" / f"fold_{fold + 1:02d}.npy", lr_data)

        # Compute VAD with the CNN model
        tr_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train),
                                   transforms=[Noise(mean=0.0, std=0.1)])
        va_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val), torch.tensor(val_ids))
        te_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test), torch.tensor(test_ids))

        tr_dataloader = DataLoader(tr_dataset, batch_size, num_workers=n_worker, pin_memory=True, shuffle=True)
        va_dataloader = DataLoader(va_dataset, batch_size, num_workers=n_worker, pin_memory=True, shuffle=False)
        te_dataloader = DataLoader(te_dataset, batch_size, num_workers=n_worker, pin_memory=True, shuffle=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = LeNetBasedVAD(in_channels=7)
        nb_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if fold == 0:
            print(f"Setting device to: {device}")
            print(f"Total number of trainable parameters of the {type(model).__name__} model: {nb_train_params:,}")

        # Setup nVAD training
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        cfunc = nn.CrossEntropyLoss()  # Cost function

        # Put model on device
        model.to(device)

        # Define file path where the weights of the best epoch will be stored
        best_model_filename = output_dir / "cnn" / f"fold_{fold + 1:02d}.pth"

        # Run the training procedure
        model.optimize(n_epochs, tr_dataloader, va_dataloader, cfunc=cfunc, optim=optim,
                       best_model=best_model_filename, device=device)

        # Re-load best model weights
        model.load_state_dict(torch.load(best_model_filename, map_location=device))

        # Predict on held-out test data (offline)
        pred, orig = model.predict(te_dataloader, device=device)

        # Store results for plotting
        nn_data = np.vstack([pred, orig, test_ids])
        logger.info(f"Finished CNN (LeNet-like) training for fold {fold + 1}. "
                    f"Error: {np.median(compute_trial_based_error(nn_data[0, :], nn_data[1, :], nn_data[2, :])):.02f}")
        np.save(output_dir / "cnn" / f"fold_{fold + 1:02d}.npy", nn_data)
    # endregion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute baseline models on the data.")
    parser.add_argument("corpus", help="Path to the high-gamma features.")
    parser.add_argument("out", help="Path to the output folder.")
    parser.add_argument("target", default="acoustic_labels", help="Target label used for the train set.")
    parser.add_argument("--epochs", default="1", help="Number of epochs for the LeNet model.")
    parser.add_argument("--dev-day", default="2022_11_18", help="Day used as development set.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f"python baseline_computations.py {args.corpus} {args.out} --epochs {args.epochs} "
                f"--dev-day {args.dev_day}")
    main(Path(args.corpus), n_epochs=int(args.epochs), dev_day=args.dev_day, output_dir=Path(args.out),
         target_label=str(args.target))
