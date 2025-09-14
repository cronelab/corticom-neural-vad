import argparse
import warnings
import logging
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from typing import Tuple
from local.utils import compute_trial_based_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


logger = logging.getLogger("statistical_analysis.py")


def load_numpy_data(filename: Path) -> np.ndarray:
    """
    Load results from numpy array and compute trial errors.
    """
    data = np.load(filename.as_posix())
    errors = compute_trial_based_error(data[0, :], data[1, :], data[2, :])
    return errors


def load_results(path: Path, is_baseline: bool = False) -> Tuple[list, list, list, list]:
    """
    Load and accumulate results from the different conditions for the statistical analysis.
    """
    accuracy, days, model_type, ground_truth = [], [], [], []

    for model in ["rnn", "cnn", "lr"]:
        # Iterate over all days
        for day in range(1, 10):
            npy_file = "result.npy" if not is_baseline and model == "rnn" else f"fold_{day:02d}.npy"

            # Not optimal, but since data is stored in different names...
            filename = path / model
            filename /= npy_file if model in ["cnn", "lr"] else f"Day_{day:02d}/{npy_file}"

            data = load_numpy_data(filename)
            accuracy.extend(data)
            days.extend([day] * len(data))
            model_type.extend([model] * len(data))
            ground_truth.extend([is_baseline] * len(data))

    return accuracy, days, model_type, ground_truth


def main(nvad_path: Path, base_path: Path):
    # Prepare data
    accuracy, days, model_type, ground_truth = load_results(nvad_path)
    base_acc, base_days, base_model_type, base_ground_truth = load_results(base_path, is_baseline=True)
    accuracy.extend(base_acc)
    days.extend(base_days)
    model_type.extend(base_model_type)
    ground_truth.extend(base_ground_truth)

    df = pd.DataFrame({"accuracy": accuracy, "session": days, "model_type": model_type, "ground_truth": ground_truth})
    logger.info(f"Dataframe head\n{df.head()}")

    # Run the statistics part and print results on command line
    model_interaction = smf.mixedlm(
        formula="accuracy ~ model_type * ground_truth",
        data=df,
        groups=df["session"],
        re_formula="1"
    )

    lme_result_interaction = model_interaction.fit()
    logger.info(lme_result_interaction.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Conduct a linear mixed-effect model analysis")
    parser.add_argument("nvad", help="Path to the nVAD folder from the results.")
    parser.add_argument("baseline", help="Path to the baseline folder.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    nvad_path = Path(args.nvad)
    base_path = Path(args.baseline)
    logger.info(f"python statistical_analysis.py {nvad_path.as_posix()} {base_path.as_posix()}")

    main(nvad_path, base_path)
