#!/usr/bin/env zsh

# Configuration
data_folder=<... Path to the data folder ...>
temp_folder=<... Path to destination where experiment results will be written to ...>
contamination_package_path=<... Path to the matlab package from Roussels paper about acoustic contamination ...>
settings=config/settings.ini

stage=1
stop_stage=24

# -------------------------------------------------------------------------------------------------------
# CONTAMINATION ANALYSIS
# -------------------------------------------------------------------------------------------------------
mkdir -p $temp_folder
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Step 1: Running the contamination analysis part"

  # Check if matlab command is available
  if ! command -v matlab &> /dev/null; then echo "Matlab command is not available."; exit 1; fi

  mkdir -p $temp_folder/contamination
  env PYTHONPATH=./eval/contamination:$PYTHONPATH python eval/contamination/aggregate_per_day.py  \
    --corpus-root $data_folder/50words                                                            \
    --acc-path $temp_folder/contamination/aggregated_by_day                                       \
    --timing-path $temp_folder/contamination/timings

  mkdir -p $temp_folder/contamination/prepared
  matlab -nodesktop -nosplash -r "addpath(genpath('eval/contamination')); \
    data_preparation('$temp_folder/contamination/prepared', \
    '$temp_folder/contamination/aggregated_by_day', \
    '$contamination_package_path'); exit;"

  mkdir -p $temp_folder/contamination/analysis
  matlab -nodesktop -nosplash -r "addpath(genpath('eval/contamination')); \
    run_contamination_analysis('$temp_folder/contamination/analysis', \
    '$temp_folder/contamination/prepared', \
    '$temp_folder/contamination/timings', \
    '$contamination_package_path'); exit;"

  mkdir -p $temp_folder/analysis
  env PYTHONPATH=./eval/contamination:$PYTHONPATH python eval/contamination/gen_contamination_report.py  \
    $temp_folder/contamination                                                                           \
    --out $temp_folder/analysis
fi

# -------------------------------------------------------------------------------------------------------
# CREATE SUPPLEMENTARY FIGURE 3
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Step 2: Render supplementary figure 3 (Days and timings of the experiment sessions)."

  python eval/supplementary_fig_3.py $data_folder/50words $data_folder/SyllableRepetition --out $temp_folder/analysis
fi

# -------------------------------------------------------------------------------------------------------
# COMPUTE FEATURES FOR THE 50 WORD TASKS
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Step 3: Preparing the neural features"

  python prepare_corpus.py $temp_folder/corpus  \
    $data_folder/SyllableRepetition             \
    $data_folder/50words

  python prepare_PY17N009.py $temp_folder/PY17N009  \
    $data_folder/PY17N009/SyllableRepetition_R04.mat
fi

# -------------------------------------------------------------------------------------------------------
# ESTIMATE HYPERPARAMETERS FROM PATIENT PY17N009
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Step 4: Estimate suitable hyperparameters from PY17N009"

  python hyperparam_optim.py $temp_folder/PY17N009/SyllableRepetition_R04.hdf  \
    --betas 0 50 100 150 200                                                   \
    --lambdas 0.11 0.0011 0.000011 0.00000011
fi

# -------------------------------------------------------------------------------------------------------
# RUN TICC ALGORITHM TO ESTIMATE ALIGNMENT USED FOR TRAINING
# -------------------------------------------------------------------------------------------------------
param_file=$temp_folder/PY17N009/result.json
beta=`python -c "import json; f=open('$param_file', 'r'); print(json.load(f)['beta']); f.close()"`
lambda=`python -c "import json; f=open('$param_file', 'r'); print(json.load(f)['lam']); f.close()"`

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Step 5: Estimating spoken speech labels using te TICC algorithm"

  python  estimate_vad_labels.py $temp_folder/gen_labels_ticc  \
    $temp_folder/corpus                                        \
    --beta $beta                                               \
    --lamb $lambda
fi

# -------------------------------------------------------------------------------------------------------
# RENDER FIGURE 2
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Step 6: Plot Figure 2"

  mkdir -p $temp_folder/analysis
  env PYTHONPATH=./eval:$PYTHONPATH python eval/plot_figure2.py $data_folder/50words/2022_11_29/50word_Overt_R01.wav  \
    $temp_folder/gen_labels_ticc/2022_11_29/50word_Overt_R01.hdf                                                      \
    $data_folder/50words/2022_11_29/50word_Overt_R01_trials.lab                                                       \
    $temp_folder/gen_labels_ticc/2022_11_18                                                                           \
    --start 0 --end 36 --out $temp_folder/analysis
fi

# -------------------------------------------------------------------------------------------------------
# COMPUTE ALIGNMENTS WITH RESPECT TO DIFFERENT TEMPORAL CONTEXTS
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Step 7: Temporal context analysis (This step may take some time)"
  mkdir -p $temp_folder/temporal_context

  # Compute the optimal temporal context on the data from the epilepsy patient
  env PYTHONPATH=./eval:$PYTHONPATH python eval/compute_temporal_context.py  \
    $temp_folder/corpus/2022_11_18                                           \
    $temp_folder/temporal_context                                            \
    --start 1 --end 7                                                        \
    --beta $beta --lamb $lambda

  # Compute the alignment errors on the window size on the patient data for the progression plot
  env PYTHONPATH=./eval:$PYTHONPATH python eval/compute_temporal_context.py  \
    $temp_folder/PY17N009                                                    \
    $temp_folder/temporal_context                                            \
    --start 1 --end 7                                                        \
    --beta $beta --lamb $lambda
fi

# -------------------------------------------------------------------------------------------------------
# RENDER THE TEMPORAL CONTEXT PLOT
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Step 8: Create the temporal context plot"

  env PYTHONPATH=./eval:$PYTHONPATH python eval/plot_context_analysis.py  \
    $temp_folder/temporal_context                                         \
    --out $temp_folder/analysis
fi

# -------------------------------------------------------------------------------------------------------
# RENDER SUPPLEMENTARY FIGURE 1
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Step 9: Create supplementary figure 1 (stability plot)"

  env PYTHONPATH=./eval:$PYTHONPATH python eval/plot_normalization_si.py  \
    $temp_folder/normalization                                            \
    --out $temp_folder/analysis
fi

# -------------------------------------------------------------------------------------------------------
# DETERMINE DIFFERENCES IN IDENTIFIED CLUSTERS
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  echo "Step 10: Compute cluster parameter differences to interpret results"

  mkdir -p $temp_folder/interpretation
  env PYTHONPATH=./eval:$PYTHONPATH python eval/compute_cluster_interpretation.py  \
    $temp_folder/corpus/2022_11_18                                                 \
    $temp_folder/interpretation                                                    \
    --beta $beta                                                                   \
    --lamb $lambda
fi

# -------------------------------------------------------------------------------------------------------
# RENDER FIGURE 4
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  echo "Step 11: Render cluster interpretability plot"

  env PYTHONPATH=./eval:$PYTHONPATH python eval/plot_cluster_interpretation.py  \
    $temp_folder/interpretation/parameter_differences.npy                       \
    img/brain_plot.jpg                                                          \
    --out $temp_folder/analysis
fi

# -------------------------------------------------------------------------------------------------------
# RENDER FIGURE 1 ASSETS
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
  echo "Step 12: Render Figure 1 assets"

  env PYTHONPATH=./eval:$PYTHONPATH python eval/plot_figure_1_assets.py  \
    $temp_folder/corpus/2022_11_18/50word_Overt_R02.hdf                  \
    $data_folder/50words/2022_11_18/50word_Overt_R02.wav                 \
    $data_folder/50words/2022_11_18/50word_Overt_R02_trials.lab          \
    450 1250 --out $temp_folder/analysis
fi

# -------------------------------------------------------------------------------------------------------
# COMPUTE BASELINE RESULTS WITH RESPECT TO METHODS DESCRIBED BY SOROUSH ET AL.
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
  echo "Step 13: Compute baseline results"

  mkdir -p $temp_folder/baseline/lr
  mkdir -p $temp_folder/baseline/cnn
  mkdir -p $temp_folder/baseline/rnn
  python baseline_computations.py  \
    $temp_folder/corpus            \
    $temp_folder/baseline          \
    acoustic_labels                \
    --epochs 10                    \
    --dev-day 2022_11_18

  python train_nVAD.py         \
    $temp_folder/baseline/rnn  \
    $temp_folder/corpus        \
    2022_11_18                 \
    acoustic_labels

  # Rename result files to mathc the other result file names from CNN and logistic regression
  for rnn_result in `find $temp_folder/baseline/rnn -name result.npy`;
  do
    k=`python -c "import sys; from pathlib import Path; print(Path(sys.argv[1]).parent.name[-2:])" $rnn_result`
    echo mv $rnn_result `dirname $rnn_result`/fold_$k.npy
  done
fi

# -------------------------------------------------------------------------------------------------------
# COMPUTE VAD RESULTS WITH RESPECT TO PROPOSED METHOD
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
  echo "Step 14: Compute decoding results"

   mkdir -p $temp_folder/nVAD/rnn
   python train_nVAD.py            \
     $temp_folder/nVAD             \
     $temp_folder/gen_labels_ticc  \
     2022_11_18                    \
     ticc_labels

  # Compute CNN and LR results
  mkdir -p $temp_folder/nVAD/cnn
  mkdir -p $temp_folder/nVAD/lr

  # Use the baseline script from before, but this time use the tick labels
  # instead of the acoustic ground truth labels
  python baseline_computations.py  \
    $temp_folder/gen_labels_ticc   \
    $temp_folder/nVAD              \
    ticc_labels                    \
    --epochs 10                    \
    --dev-day 2022_11_18
fi

# -------------------------------------------------------------------------------------------------------
# RENDER FIGURE 5
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
  echo "Step 15: Render decoding results plot"

  env PYTHONPATH=./eval:$PYTHONPATH python eval/plot_figure5.py  \
    $temp_folder/baseline                                        \
    $temp_folder/nVAD                                            \
    --out $temp_folder/analysis
fi

# -------------------------------------------------------------------------------------------------------
# COMPUTE FEATURES FOR THE NGSLS WORDS USED AS OUT-OF-VOCABULARY WORDS
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
  echo "Step 16: Prepare features for the out-of-vocabulary words"

  mkdir -p $temp_folder/corpus_unseen
  python prepare_corpus.py $temp_folder/corpus_unseen    \
    $data_folder/SyllableRepetition                      \
    $data_folder/NGSLSWordReading
fi

# -------------------------------------------------------------------------------------------------------
# COMPUTE VAD RESULTS ON UNSEEN WORDS
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
  echo "Step 17: Compute voice activity detection results on unseen data"

  mkdir -p $temp_folder/unseen_results
  for i in $(seq 1 9); do
    python decode_offline.py $temp_folder/unseen_results    \
      `ls -d "$temp_folder/corpus_unseen/"*`                \
      $temp_folder/nVAD/Day_0$i/best_model.pth
  done
fi

# -------------------------------------------------------------------------------------------------------
# COMPUTE VAD RESULTS ON UNSEEN WORDS
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ]; then
  echo "Step 18: Compute numbers for unseen data for paper"

  env PYTHONPATH=./eval:$PYTHONPATH python eval/eval_generalization.py $temp_folder/unseen_results
fi

# -------------------------------------------------------------------------------------------------------
# COMPUTE DAY-SPECIFIC NORMALIZATION DATA
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ]; then
    echo "Stage 19: Compute normalization statistics based on SyllableRepetition data for the unseen words"

    mkdir -p $temp_folder/normalization
    python normalization_statistics.py $data_folder/SyllableRepetition/2022_11_09/SyllableRepetition_Overt.mat  \
      --out $temp_folder/normalization/2022_11_09.npy --overwrite
fi

# -------------------------------------------------------------------------------------------------------
# START THE ONLINE CORTICOM-VAD-SYSTEM
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ]; then
    echo "Stage 20: Starting online VAD system"

    python decode_online.py $settings --run replicate --overwrite &
fi

# -------------------------------------------------------------------------------------------------------
# STREAM DATA LOCALLY IN THE BACKGROUND USING THE DEVELOPMENT AMPLIFIER
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ]; then
    echo "Stage 21: Play one file from the online test days locally to synthesize speech (will run for 60 seconds)"

    python development_amplifier.py $data_folder/NGSLSWordReading/2022_11_09/NGSLSWordReading_Block1.mat  \
      --seconds 60 --package_size 40
fi

# -------------------------------------------------------------------------------------------------------
# AVERAGE SPEECH DURATION
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ]; then
    echo "Stage 22: Determine average duration for the participant to say a word."

    echo "50 Word Corpus:"
    env PYTHONPATH=./eval:$PYTHONPATH python eval/avg_speech_duration.py $temp_folder/corpus

    echo "50 Word Corpus (Development set only):"
    env PYTHONPATH=./eval:$PYTHONPATH python eval/avg_speech_duration.py $temp_folder/corpus/2022_11_18

    echo "NGSLS Word List:"
    env PYTHONPATH=./eval:$PYTHONPATH python eval/avg_speech_duration.py $temp_folder/corpus_unseen
fi

# -------------------------------------------------------------------------------------------------------
# DISCUSSION SCORES
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ]; then
    echo "Stage 23: Compute recall scores for discussion section."

    # env PYTHONPATH=./eval:$PYTHONPATH python eval/majority_of_speech.py $temp_folder/nVAD/rnn
    # env PYTHONPATH=./eval:$PYTHONPATH python eval/majority_of_speech.py $temp_folder/unseen_results
    env PYTHONPATH=./eval:$PYTHONPATH python eval/recall_scores.py $temp_folder/baseline/rnn  \
      $temp_folder/nVAD/rnn                                                                   \
      $temp_folder/corpus_unseen
fi

# -------------------------------------------------------------------------------------------------------
# STATISTICAL ANALYSIS
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ]; then
    echo "Stage 24: Run the Linear mixed-effects model statistical analysis."

    env PYTHONPATH=./eval:$PYTHONPATH python eval/statistical_analysis.py  \
      $temp_folder/nVAD                                                    \
      $temp_folder/baseline
fi
