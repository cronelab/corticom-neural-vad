# Corticom Unsupervised Neural VAD
Code repository for the paper "Real-time detection of spoken speech from unlabeled ECoG signals: A pilot study with an ALS participant" by Angrick et al.

## Installation

All dependencies can be found in the [requirements.txt](requirements.txt) file. This repository was tested with Python version 3.10. Additionally, the TICC algorithm is linked as a submodule.
```bash
pip install -r requirements.txt
pip install extensions/hga
git submodules init
git submodules update
```

## Replicate results

There is a [replicate shell script](replicate.sh) which runs all steps from the paper to reproduce the results. 
It needs to configured right i the beginning to point to the right paths for accessing the data and storing all results.

## Structure of the output folder

The scripts will all output their results in a dedicated folder. This folder will contain the following sub folders:

- **analysis**: This folder holds all rendered figures after running the [replicate shell script](replicate.sh). 
- **corpus**: The corpus folder will be created by the [prepare_corpus.py](./prepare_corpus.py) script during the computation of the high-gamma features (and normalizations). All results are stored in the HDF5 fle format.
- **normalization**: This folder will be created by the [prepare_corpus.py](./prepare_corpus.py) script. It contains for each particular day the normalization statistics.
- **temporal_context**: This folder contains the alignment errors (according to the Levenshtein distance) for each trial. It will be created by the [compute_temporal_context.py](./eval/compute_temporal_context.py) script.
- **gen_labels_ticc**: Folder containing the estimated labels from the TICC algorithm. Results are stored in the HDF5 file format and contain datasets for the high-gamma features, the acoustic VAD alignments and the alignments from the TICC algorithm. This folder will be created by [estimate_vad_labels.py](estimate_vad_labels.py).
- **baseline**: Results from the baseline computation (Leave-one-day-out cross-validation), split into folders for the CNN and the logistic regression approaches. This folder will be created by [baseline_computations.py](baseline_computations.py).