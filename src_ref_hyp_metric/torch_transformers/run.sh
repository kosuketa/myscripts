#!/bin/sh
# Default values of arguments

# general setting
EXP_NAME="test"
EXP_ID=0
TRIAL_TIMES=1
TMP_PATH='/home/is/kosuke-t/tmp/tmp_log'
DUMP_PATH="/ahc/work3/kosuke-t/SRHDA/transformers/log/"
MODEL_NAME="xlm-roberta-large"

# lang for WMT19 all-all
#LANGS="de-cs,de-en,de-fr,en-cs,en-de,en-fi,en-gu,en-kk,en-lt,en-ru,en-zh,fi-en,fr-de,gu-en,kk-en,lt-en,ru-en,zh-en"
# lang for WMT19 all-en
#LANGS="de-en,fi-en,gu-en,kk-en,lt-en,ru-en,zh-en"
# lang for WMT19 en-all
#LANGS="en-cs,en-de,en-fi,en-gu,en-kk,en-lt,en-ru,en-zh"

# lang for WMT18 for all-all
LANGS="cs-en,de-en,en-cs,en-de,en-et,en-fi,en-ru,en-tr,en-zh,et-en,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en"

# lang for WMT17
LANGS="cs-en,de-en,lv-en,fi-en,ro-en,ru-en,tr-en,zh-en"

# lang for XLM15
# LANGS="de-en,ru-en,tr-en,zh-en"



EMPTY_DUMP="False"
TRAIN="True"
TEST="True"

# hyperparameters
BATCH_SIZE=()
EPOCH_SIZE=3
OPTIMIZER=()
LR_LAMBDA="0.707"
DROPOUT="0.0"

# model setting
AMP="True"
LOAD_MODEL="False"
LOAD_MODEL_PATH=""
SAVE_MODEL_NAME="model.pth"
SAVE_MODEL_PATH=""
HYP_REF="False"
HYP_SRC="False"
HYP_SRC_HYP_REF="False"
HYP_SRC_REF="False"

# data setting
SRC_TRAIN="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/train.src"
SRC_VALID="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/valid.src"
SRC_TEST="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/test.src"
REF_TRAIN="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/train.ref"
REF_VALID="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/valid.ref"
REF_TEST="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/test.ref"
HYP_TRAIN="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/train.hyp"
HYP_VALID="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/valid.hyp"
HYP_TEST="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/test.hyp"
LABEL_TRAIN="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/train.label"
LABEL_VALID="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/valid.label"
LABEL_TEST="/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/test.label"
DARR='False'

TRAIN_SHRINK="1.0"

DEBUG="False"

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
        --trial_times)
            TRIAL_TIMES="$2"
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
        --langs)
            LANGS="$2"
            shift 2
            ;;
        --batch_size)
            shift
            for ar in "$@"
                do
                case $ar in
                    batch=*)
                        BATCH_SIZE+=("$ar")
                        shift
                        ;;
                    *)
                        break
                        ;;
                esac
            done
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
        --hyp_ref)
            HYP_REF="$2"
            shift 2
            ;;
        --hyp_src)
            HYP_SRC="$2"
            shift 2
            ;;
        --hyp_src_hyp_ref)
            HYP_SRC_HYP_REF="$2"
            shift 2
            ;;
        --hyp_src_ref)
            HYP_SRC_REF="$2"
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
        --ref_train)
            REF_TRAIN="$2"
            shift 2
            ;;
        --ref_valid)
            REF_VALID="$2"
            shift 2
            ;;
        --ref_test)
            REF_TEST="$2"
            shift 2
            ;;
        --hyp_train)
            HYP_TRAIN="$2"
            shift 2
            ;;
        --hyp_valid)
            HYP_VALID="$2"
            shift 2
            ;;
        --hyp_test)
            HYP_TEST="$2"
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
        --darr)
            DARR="$2"
            shift 2
            ;;
        --train_shrink)
            TRAIN_SHRINK="$2"
            shift 2
            ;;
        --debug)
            DEBUG="$2"
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
if [ ${#BATCH_SIZE[@]} -eq 0 ]; then
  BATCH_SIZE=()
  BATCH_SIZE=("batch=8")
fi


for mini_batch in "${BATCH_SIZE[@]}" ; do
    for opt in "${OPTIMIZER[@]}" ; do
        EXP_ID=`expr "$EXP_ID" + 1`
#         for i in `seq "${TRIAL_TIMES}"` ; do
            N_TRIAL="1"
            python trainer.py \
            --exp_name "$EXP_NAME" \
            --exp_id "$EXP_ID" \
            --trial_times "$TRIAL_TIMES" \
            --n_trial "$N_TRIAL" \
            --dump_path "$DUMP_PATH" \
            --model_name "$MODEL_NAME" \
            --langs "$LANGS" \
            --empty_dump "$EMPTY_DUMP" \
            --train "$TRAIN" \
            --test "$TEST" \
            --batch_size "$mini_batch" \
            --epoch_size "$EPOCH_SIZE" \
            --optimizer "$opt" \
            --lr_lambda "$LR_LAMBDA" \
            --dropout "$DROPOUT" \
            --amp "$AMP" \
            --load_model "$LOAD_MODEL" \
            --load_model_path "$LOAD_MODEL_PATH" \
            --save_model_name "$SAVE_MODEL_NAME" \
            --save_model_path "$SAVE_MODLE_PATH" \
            --hyp_ref "$HYP_REF" \
            --hyp_src "$HYP_SRC" \
            --hyp_src_hyp_ref "$HYP_SRC_HYP_REF" \
            --hyp_src_ref "$HYP_SRC_REF" \
            --src_train "$SRC_TRAIN" \
            --src_valid "$SRC_VALID" \
            --src_test "$SRC_TEST" \
            --ref_train "$REF_TRAIN" \
            --ref_valid "$REF_VALID" \
            --ref_test "$REF_TEST" \
            --hyp_train "$HYP_TRAIN" \
            --hyp_valid "$HYP_VALID" \
            --hyp_test "$HYP_TEST" \
            --label_train "$LABEL_TRAIN" \
            --label_valid "$LABEL_VALID" \
            --label_test "$LABEL_TEST" \
            --darr "$DARR" \
            --train_shrink "$TRAIN_SHRINK" \
            --debug "$DEBUG"
#         done
    done
done
#python results_summary.py --exp_name "$EXP_NAME" --langs "$RG_STEPS" --n_operation "$OP_TIMES" --valid_lang_separate True
# 