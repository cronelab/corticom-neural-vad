import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from local.architecture import UnidirectionalVoiceActivityDetector
from local.utils import SequentialSpeechTrials, SelectElectrodesOverSpeechAreas
from local.utils import get_trial_ids_for_day, compute_trial_based_error
from torchvision.transforms import Compose


logger = logging.getLogger("decode_offline.py")


def main(output_dir: Path, corpus: Path, weights_path: Path):
    # Get all precomputed feature files per day
    feature_files = sorted(corpus.glob('*.hdf'))
    torch.set_default_dtype(torch.float32)

    # Initialize datasets
    input_transforms = Compose([
        SelectElectrodesOverSpeechAreas(),
        torch.tensor,
    ])

    dataset = SequentialSpeechTrials(feature_files=[f.as_posix() for f in feature_files],
                                     target_specifier="acoustic_labels",
                                     transform=input_transforms)

    dataloader = DataLoader(dataset, num_workers=4, pin_memory=True, shuffle=False, batch_size=1)

    # Setup nVAD architecture
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Setting device to: {device}")

    model = UnidirectionalVoiceActivityDetector(nb_electrodes=64, nb_layer=2, nb_hidden_units=100)
    model.load_state_dict(torch.load(weights_path.as_posix()))
    model.to(device)

    # Compute performance on the test set
    pred, orig = model.predict(dataloader, device=device)

    trial_ids = get_trial_ids_for_day(corpus)
    error = compute_trial_based_error(pred, orig, trial_ids)

    logger.info(f"Performance results for weights {weights_path}: {np.median(error):.02f}")
    day_nr = int(weights_path.parent.name.split("_")[1])
    np.save(output_dir / f"rnn_day_{day_nr:02d}.npy", np.vstack([pred, orig, trial_ids]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Decode data...")
    parser.add_argument("out_dir", help="Path to the output folder.")
    parser.add_argument("corpus", help="Path to the HDF5 containers with features.")
    parser.add_argument("weights", help="Path to the model weights.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f'python train_nVAD.py {args.out_dir} {args.corpus} {args.weights}')
    main(output_dir=Path(args.out_dir), corpus=Path(args.corpus), weights_path=Path(args.weights))
