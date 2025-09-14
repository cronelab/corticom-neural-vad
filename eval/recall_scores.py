import argparse
import torch
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import recall_score
from local.architecture import UnidirectionalVoiceActivityDetector
from local.utils import SelectElectrodesOverSpeechAreas, SequentialSpeechTrials
from torchvision.transforms import Compose
from corticom.speech.external.hga_optimized import stack_features_spatio_temporal
from torch.utils.data import DataLoader


logger = logging.getLogger("recall_Scores.py")


def main(base_rnn: Path, nvad_rnn: Path, unseen_corpus: Path, ):
    # Compute recall scores for the full NGSLS word list
    nvad_scores = []
    for results_filename in sorted(nvad_rnn.rglob("result.npy")):
        data = np.load(results_filename.as_posix())
        nvad_scores.append(recall_score(data[1, :], data[0, :], average='binary', pos_label=1))

    logger.info(f"nVAD recall on full NGSLS word list: {np.mean(nvad_scores):.02f}")

    # Compute recall scores for the CNN model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup model and data transforms
    channel_selector = SelectElectrodesOverSpeechAreas()
    input_transforms = Compose([channel_selector, torch.tensor])

    test_dataset = SequentialSpeechTrials(
        feature_files=[p.as_posix() for p in unseen_corpus.rglob("*.hdf")],
        target_specifier="acoustic_labels",
        transform=input_transforms
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)

    vase_rnn_scores = []
    for weights_filename in sorted(base_rnn.rglob("best_model.pth")):
        model = UnidirectionalVoiceActivityDetector(nb_electrodes=64, nb_layer=2, nb_hidden_units=128)
        model.to(device)
        model.load_state_dict(torch.load(weights_filename))

        pred, orig = model.predict(test_dataloader, device=device)
        vase_rnn_scores.append(recall_score(orig, pred, average='binary', pos_label=1))

    logger.info(f"RNN (baseline) recall on full NGSLS word list: {np.mean(vase_rnn_scores):.02f}")


if __name__ == "__main__":
    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    # read command line arguments
    parser = argparse.ArgumentParser("Report how many trials the approach identified more than 50% of the speech.")
    parser.add_argument("base_rnn", help="Path to baseline RNN folder in baseline.")
    parser.add_argument("nvad_rnn", help="Path to the nVAD RNN folder.")
    parser.add_argument("unseen_corpus", help="Path to the feature files for the unseen data.")
    args = parser.parse_args()

    logger.info(f'python recall_scores.py {args.base_rnn} {args.nvad_rnn} {args.unseen_corpus}')
    main(Path(args.base_rnn), Path(args.nvad_rnn), Path(args.unseen_corpus))
