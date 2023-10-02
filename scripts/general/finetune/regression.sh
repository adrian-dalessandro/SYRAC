DATA_DIR="../datasets/"
TRAIN="/DS_NOISY_SYNTH"
TEST="$1"
N="$2"
EXPERIMENT_PATH="$3"
MODEL_PATH="$4"
PARAMS_DIR="./config"
EXP_NOTES="LOG"

python3 finetune_wrapper.py --experiment finetuning/unsupervised/noisy_synth_regress \
                         --data_dir $DATA_DIR \
                         --train_data $TRAIN \
                         --test_data $TEST \
                         --N $N \
                         --experiment_path $EXPERIMENT_PATH \
                         --model_path $MODEL_PATH \
                         --params_dir $PARAMS_DIR \