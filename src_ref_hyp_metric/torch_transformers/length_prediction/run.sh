#!/bin/sh
# Default values of arguments

# general setting
EXP_NAME="monoBERT_uncased"
EXP_ID=0
DUMP_PATH="/ahc/work3/kosuke-t/SRHDA/length_prediction/log/"
MODEL_NAME="bert-base-uncased"
# LANGS="de-en,ru-en,tr-en,zh-en"
EMPTY_DUMP="False"
TRAIN="True"
TEST="True"

# hyperparameters
BATCH_SIZE=16
EPOCH_SIZE=3
OPTIMIZER=()
LR_LAMBDA="0.707"
DROPOUT="0.0"

# model setting
AMP="True"
LOAD_MODEL="False"
LOAD_MODEL_PATH="/ahc/work3/kosuke-t/SRHDA/transformers/log/${EXP_NAME}/model.pth"
SAVE_MODEL_NAME="model.pth"
SAVE_MODEL_PATH=""

# data setting
SRC_TRAIN="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/length_prediction/train.src"
SRC_VALID="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/length_prediction/valid.src"
SRC_TEST="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/length_prediction/test.src"
LABEL_TRAIN="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/length_prediction/train.label"
LABEL_VALID="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/length_prediction/valid.label"
LABEL_TEST="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/length_prediction/test.label"

# Loop through arguments and process them
while [ -n "$1" ];
do
    arg="$1"
    case $arg in
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --exp_id)
            EXP_ID="$2"
            shift 2
            ;;
        --empty_dump)
            EMPTY_DUMP="$2"
            shift 2
            ;;
        --train)
            TRAIN="$2"
            shift 2
            ;;
        --test)
            TEST="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --optimizer)
            shift
            for ar in "$@"
                do
                case $ar in
                    adam,lr=*)
                        OPTIMIZER+=("$ar")
                        shift
                        ;;
                    adadelta,lr=*)
                        OPTIMIZER+=("$ar")
                        shift
                        ;;
                    adagrad,lr=*)
                        OPTIMIZER+=("$ar")
                        shift
                        ;;
                    adam_inverse_sqrt,lr=*)
                        OPTIMIZER+=("$ar")
                        shift
                        ;;
                    adamax,lr=*)
                        OPTIMIZER+=("$ar")
                        shift
                        ;;
                    asgd,lr=*)
                        OPTIMIZER+=("$ar")
                        shift
                        ;;
                    rmsprop,lr=*)
                        OPTIMIZER+=("$ar")
                        shift
                        ;;
                    rprop,lr=*)
                        OPTIMIZER+=("$ar")
                        shift
                        ;;
                    sgd,lr=*)
                        OPTIMIZER+=("$ar")
                        shift
                        ;;
                    *)
                        break
                        ;;
                esac
            done
            ;;
        --dump_path)
            DUMP_PARH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --lr_lambda)
            LR_LAMBDA="$2"
            shift 2
            ;;
        --epoch_size)
            EPOCH_SIZE="$2"
            shift 2
            ;;
        --amp)
            AMP="$2"
            shift 2
            ;;
        --load_model)
            LOAD_MODEL="$2"
            shift 2
            ;;
        --load_model_path)
            LOAD_MODEL_PATH="$2"
            shift 2
            ;;
        --save_model_name)
            SAVE_MODEL_NAME="$2"
            shift 2
            ;;
        --save_model_path)
            SAVE_MODEL_PATH="$2"
            shift 2
            ;;
        --src_train)
            SRC_TRAIN="$2"
            shift 2
            ;;
        --src_valid)
            SRC_VALID="$2"
            shift 2
            ;;
        --src_test)
            SRC_TEST="$2"
            shift 2
            ;;
        --label_train)
            LABEL_TRAIN="$2"
            shift 2
            ;;
        --label_valid)
            LABEL_VALID="$2"
            shift 2
            ;;
        --label_test)
            LABEL_TEST="$2"
            shift 2
            ;;
        
        *)
        echo NOT DEFINED ARGUMENTS: "$arg"
        echo "$arg"
        exit
        ;;
    esac
done

if [ ${#OPTIMIZER[@]} -eq 0 ]; then
  OPTIMIZER=()
  OPTIMIZER=("adam,lr=0.00001")
fi

python trainer_debug.py \
        --exp_name "$EXP_NAME" \
        --exp_id "$EXP_ID" \
        --dump_path "$DUMP_PATH" \
        --model_name "$MODEL_NAME" \
        --empty_dump "$EMPTY_DUMP" \
        --train "$TRAIN" \
        --test "$TEST" \
        --batch_size "$BATCH_SIZE" \
        --epoch_size "$EPOCH_SIZE" \
        --optimizer "$OPTIMIZER" \
        --lr_lambda "$LR_LAMBDA" \
        --dropout "$DROPOUT" \
        --amp "$AMP" \
        --load_model "$LOAD_MODEL" \
        --load_model_path "$LOAD_MODEL_PATH" \
        --save_model_name "$SAVE_MODEL_NAME" \
        --save_model_path "$SAVE_MODLE_PATH" \
        --src_train "$SRC_TRAIN" \
        --src_valid "$SRC_VALID" \
        --src_test "$SRC_TEST" \
        --label_train "$LABEL_TRAIN" \
        --label_valid "$LABEL_VALID" \
        --label_test "$LABEL_TEST"