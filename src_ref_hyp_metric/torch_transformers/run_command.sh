HOME="/home/is/kosuke-t"
PROJECT_DISC="${HOME}/project_disc"
# DATA_PATH="${PROJECT_DISC}/data/SRHDA/WMT15_19_DA"
DATA_PATH="${PROJECT_DISC}/data/SRHDA/WMT15_18_DA"
# DATA_PATH="${PROJECT_DISC}/data/SRHDA/WMT15_17_DA_HUME"

YEAR="18"

MODEL_PATH="${PROJECT_DISC}/model"
# MODEL_NAME="bert-base-uncased"
# MODEL_NAME="bert-base-multilingual-cased"
# MODEL_NAME="roberta-base"
# MODEL_NAME="roberta-large-openai-detector"
# MODEL_NAME="xlm-mlm-100-1280"
# MODEL_NAME="xlm-mlm-tlm-xnli15-1024"
MODEL_NAME="bert-base-uncased"
OPTIMIZER=("adam,lr=0.000009")
BATCH_SIZE=("16")

DARR="True"

# lang for WMT19 all-all
#LANGS="de-cs,de-en,de-fr,en-cs,en-de,en-fi,en-gu,en-kk,en-lt,en-ru,en-zh,fi-en,fr-de,gu-en,kk-en,lt-en,ru-en,zh-en"
# lang for WMT19 all-en
#LANGS="de-en,fi-en,gu-en,kk-en,lt-en,ru-en,zh-en"
# lang for WMT19 en-all
#LANGS="en-cs,en-de,en-fi,en-gu,en-kk,en-lt,en-ru,en-zh"

# lang for WMT18 for all-all
#LANGS="cs-en,de-en,en-cs,en-de,en-et,en-fi,en-ru,en-tr,en-zh,et-en,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en"
# lang for WMT18 for all-en
LANGS="cs-en,de-en,et-en,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en"
# lang for WMT18 for en-all
#LANGS="en-cs,en-de,en-et,en-fi,en-ru,en-tr,en-zh"

# lang for WMT17 for all-all
# LANGS="cs-en,de-en,en-ru,en-zh,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en"
# lang for WMT17 all-en
# LANGS="cs-en,de-en,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en"

HYP_REF="False"
HYP_SRC="False"
HYP_SRC_HYP_REF="True"
HYP_SRC_REF="False"

# EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_ref"
# EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_src"
EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_src_hyp_ref"
# EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_src_ref"

for mini_batch in "${BATCH_SIZE[@]}" ; do
    for opt in "${OPTIMIZER[@]}" ; do
        for N_TRIAL in `seq "1"` ; do
            python ${HOME}/scripts/src_ref_hyp_metric/torch_transformers/trainer.py \
            --exp_name "${EXP_NAME}" \
            --exp_id "0" \
            --n_trial "${N_TRIAL}" \
            --tmp_path "${HOME}/tmp/tmp_log/" \
            --dump_path "${PROJECT_DISC}/SRHDA/transformers/log/" \
            --model_name "${MODEL_NAME}" \
            --langs "${LANGS}" \
            --empty_dump "False" \
            --train "True" \
            --test "False" \
            --batch_size "${mini_batch}" \
            --epoch_size "1" \
            --optimizer "${opt}" \
            --lr_lambda "0.707" \
            --dropout "0.0" \
            --amp "True" \
            --load_model "False" \
            --load_model_path "" \
            --save_model_name "model.pth" \
            --save_model_path "" \
            --hyp_ref "${HYP_REF}" \
            --hyp_src "${HYP_SRC}" \
            --hyp_src_hyp_ref "${HYP_SRC_HYP_REF}" \
            --hyp_src_ref "${HYP_SRC_REF}" \
            --model_path "${MODEL_PATH}" \
            --src_train "${DATA_PATH}/train.src" \
            --src_valid  "${DATA_PATH}/valid.src" \
            --src_test "${DATA_PATH}/test.src" \
            --ref_train "${DATA_PATH}/train.ref" \
            --ref_valid "${DATA_PATH}/valid.ref" \
            --ref_test "${DATA_PATH}/test.ref" \
            --hyp_train "${DATA_PATH}/train.hyp" \
            --hyp_valid "${DATA_PATH}/valid.hyp" \
            --hyp_test "${DATA_PATH}/test.hyp" \
            --label_train "${DATA_PATH}/train.label" \
            --label_valid "${DATA_PATH}/valid.label" \
            --label_test "${DATA_PATH}/test.label" \
            --darr "${DARR}" \
            --train_shrink "1.0" \
            --debug "False"
        done
    done
done

python ${HOME}/scripts/src_ref_hyp_metric/torch_transformers/trainer.py \
--exp_name "${EXP_NAME}" \
--exp_id "0" \
--n_trial "${N_TRIAL}" \
--tmp_path "${HOME}/tmp/tmp_log/" \
--dump_path "${PROJECT_DISC}/SRHDA/transformers/log/" \
--model_name "${MODEL_NAME}" \
--langs "${LANGS}" \
--empty_dump "False" \
--train "False" \
--test "True" \
--batch_size "${mini_batch}" \
--epoch_size "10" \
--optimizer "${opt}" \
--lr_lambda "0.707" \
--dropout "0.0" \
--amp "True" \
--load_model "False" \
--load_model_path "" \
--save_model_name "model.pth" \
--save_model_path "" \
--hyp_ref "${HYP_REF}" \
--hyp_src "${HYP_SRC}" \
--hyp_src_hyp_ref "${HYP_SRC_HYP_REF}" \
--hyp_src_ref "${HYP_SRC_REF}" \
--model_path "${MODEL_PATH}" \
--src_train "${DATA_PATH}/train.src" \
--src_valid  "${DATA_PATH}/valid.src" \
--src_test "${DATA_PATH}/test.src" \
--ref_train "${DATA_PATH}/train.ref" \
--ref_valid "${DATA_PATH}/valid.ref" \
--ref_test "${DATA_PATH}/test.ref" \
--hyp_train "${DATA_PATH}/train.hyp" \
--hyp_valid "${DATA_PATH}/valid.hyp" \
--hyp_test "${DATA_PATH}/test.hyp" \
--label_train "${DATA_PATH}/train.label" \
--label_valid "${DATA_PATH}/valid.label" \
--label_test "${DATA_PATH}/test.label" \
--darr "${DARR}" \
--train_shrink "1.0" \
--debug "False"



# HYP_REF="False"
# HYP_SRC="True"
# HYP_SRC_HYP_REF="False"
# HYP_SRC_REF="False"
# # EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_ref"
# EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_src"
# # EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_src_hyp_ref"
# # EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_src_ref"

# for mini_batch in "${BATCH_SIZE[@]}" ; do
#     for opt in "${OPTIMIZER[@]}" ; do
#         for N_TRIAL in `seq "10"` ; do
#             python ${HOME}/scripts/src_ref_hyp_metric/torch_transformers/trainer.py \
#             --exp_name "${EXP_NAME}" \
#             --exp_id "0" \
#             --n_trial "${N_TRIAL}" \
#             --tmp_path "${HOME}/tmp/tmp_log/" \
#             --dump_path "${PROJECT_DISC}/SRHDA/transformers/log/" \
#             --model_name "${MODEL_NAME}" \
#             --langs "${LANGS}" \
#             --empty_dump "False" \
#             --train "True" \
#             --test "False" \
#             --batch_size "${mini_batch}" \
#             --epoch_size "10" \
#             --optimizer "${opt}" \
#             --lr_lambda "0.707" \
#             --dropout "0.0" \
#             --amp "True" \
#             --load_model "False" \
#             --load_model_path "" \
#             --save_model_name "model.pth" \
#             --save_model_path "" \
#             --hyp_ref "${HYP_REF}" \
#             --hyp_src "${HYP_SRC}" \
#             --hyp_src_hyp_ref "${HYP_SRC_HYP_REF}" \
#             --hyp_src_ref "${HYP_SRC_REF}" \
#             --model_path "${MODEL_PATH}" \
#             --src_train "${DATA_PATH}/train.src" \
#             --src_valid  "${DATA_PATH}/valid.src" \
#             --src_test "${DATA_PATH}/test.src" \
#             --ref_train "${DATA_PATH}/train.ref" \
#             --ref_valid "${DATA_PATH}/valid.ref" \
#             --ref_test "${DATA_PATH}/test.ref" \
#             --hyp_train "${DATA_PATH}/train.hyp" \
#             --hyp_valid "${DATA_PATH}/valid.hyp" \
#             --hyp_test "${DATA_PATH}/test.hyp" \
#             --label_train "${DATA_PATH}/train.label" \
#             --label_valid "${DATA_PATH}/valid.label" \
#             --label_test "${DATA_PATH}/test.label" \
#             --darr "${DARR}" \
#             --train_shrink "1.0" \
#             --debug "False"
#         done
#     done
# done

# python ${HOME}/scripts/src_ref_hyp_metric/torch_transformers/trainer.py \
# --exp_name "${EXP_NAME}" \
# --exp_id "0" \
# --n_trial "${N_TRIAL}" \
# --tmp_path "${HOME}/tmp/tmp_log/" \
# --dump_path "${PROJECT_DISC}/SRHDA/transformers/log/" \
# --model_name "${MODEL_NAME}" \
# --langs "${LANGS}" \
# --empty_dump "False" \
# --train "False" \
# --test "True" \
# --batch_size "${mini_batch}" \
# --epoch_size "10" \
# --optimizer "${opt}" \
# --lr_lambda "0.707" \
# --dropout "0.0" \
# --amp "True" \
# --load_model "False" \
# --load_model_path "" \
# --save_model_name "model.pth" \
# --save_model_path "" \
# --hyp_ref "${HYP_REF}" \
# --hyp_src "${HYP_SRC}" \
# --hyp_src_hyp_ref "${HYP_SRC_HYP_REF}" \
# --hyp_src_ref "${HYP_SRC_REF}" \
# --model_path "${MODEL_PATH}" \
# --src_train "${DATA_PATH}/train.src" \
# --src_valid  "${DATA_PATH}/valid.src" \
# --src_test "${DATA_PATH}/test.src" \
# --ref_train "${DATA_PATH}/train.ref" \
# --ref_valid "${DATA_PATH}/valid.ref" \
# --ref_test "${DATA_PATH}/test.ref" \
# --hyp_train "${DATA_PATH}/train.hyp" \
# --hyp_valid "${DATA_PATH}/valid.hyp" \
# --hyp_test "${DATA_PATH}/test.hyp" \
# --label_train "${DATA_PATH}/train.label" \
# --label_valid "${DATA_PATH}/valid.label" \
# --label_test "${DATA_PATH}/test.label" \
# --darr "${DARR}" \
# --train_shrink "1.0" \
# --debug "False"


# HYP_REF="False"
# HYP_SRC="False"
# HYP_SRC_HYP_REF="True"
# HYP_SRC_REF="False"
# # EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_ref"
# # EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_src"
# EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_src_hyp_ref"
# # EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_src_ref"

# for mini_batch in "${BATCH_SIZE[@]}" ; do
#     for opt in "${OPTIMIZER[@]}" ; do
#         for N_TRIAL in `seq "10"` ; do
#             python ${HOME}/scripts/src_ref_hyp_metric/torch_transformers/trainer.py \
#             --exp_name "${EXP_NAME}" \
#             --exp_id "0" \
#             --n_trial "${N_TRIAL}" \
#             --tmp_path "${HOME}/tmp/tmp_log/" \
#             --dump_path "${PROJECT_DISC}/SRHDA/transformers/log/" \
#             --model_name "${MODEL_NAME}" \
#             --langs "${LANGS}" \
#             --empty_dump "False" \
#             --train "True" \
#             --test "False" \
#             --batch_size "${mini_batch}" \
#             --epoch_size "10" \
#             --optimizer "${opt}" \
#             --lr_lambda "0.707" \
#             --dropout "0.0" \
#             --amp "True" \
#             --load_model "False" \
#             --load_model_path "" \
#             --save_model_name "model.pth" \
#             --save_model_path "" \
#             --hyp_ref "${HYP_REF}" \
#             --hyp_src "${HYP_SRC}" \
#             --hyp_src_hyp_ref "${HYP_SRC_HYP_REF}" \
#             --hyp_src_ref "${HYP_SRC_REF}" \
#             --model_path "${MODEL_PATH}" \
#             --src_train "${DATA_PATH}/train.src" \
#             --src_valid  "${DATA_PATH}/valid.src" \
#             --src_test "${DATA_PATH}/test.src" \
#             --ref_train "${DATA_PATH}/train.ref" \
#             --ref_valid "${DATA_PATH}/valid.ref" \
#             --ref_test "${DATA_PATH}/test.ref" \
#             --hyp_train "${DATA_PATH}/train.hyp" \
#             --hyp_valid "${DATA_PATH}/valid.hyp" \
#             --hyp_test "${DATA_PATH}/test.hyp" \
#             --label_train "${DATA_PATH}/train.label" \
#             --label_valid "${DATA_PATH}/valid.label" \
#             --label_test "${DATA_PATH}/test.label" \
#             --darr "${DARR}" \
#             --train_shrink "1.0" \
#             --debug "False"
#         done
#     done
# done

# python ${HOME}/scripts/src_ref_hyp_metric/torch_transformers/trainer.py \
# --exp_name "${EXP_NAME}" \
# --exp_id "0" \
# --n_trial "${N_TRIAL}" \
# --tmp_path "${HOME}/tmp/tmp_log/" \
# --dump_path "${PROJECT_DISC}/SRHDA/transformers/log/" \
# --model_name "${MODEL_NAME}" \
# --langs "${LANGS}" \
# --empty_dump "False" \
# --train "False" \
# --test "True" \
# --batch_size "${mini_batch}" \
# --epoch_size "10" \
# --optimizer "${opt}" \
# --lr_lambda "0.707" \
# --dropout "0.0" \
# --amp "True" \
# --load_model "False" \
# --load_model_path "" \
# --save_model_name "model.pth" \
# --save_model_path "" \
# --hyp_ref "${HYP_REF}" \
# --hyp_src "${HYP_SRC}" \
# --hyp_src_hyp_ref "${HYP_SRC_HYP_REF}" \
# --hyp_src_ref "${HYP_SRC_REF}" \
# --model_path "${MODEL_PATH}" \
# --src_train "${DATA_PATH}/train.src" \
# --src_valid  "${DATA_PATH}/valid.src" \
# --src_test "${DATA_PATH}/test.src" \
# --ref_train "${DATA_PATH}/train.ref" \
# --ref_valid "${DATA_PATH}/valid.ref" \
# --ref_test "${DATA_PATH}/test.ref" \
# --hyp_train "${DATA_PATH}/train.hyp" \
# --hyp_valid "${DATA_PATH}/valid.hyp" \
# --hyp_test "${DATA_PATH}/test.hyp" \
# --label_train "${DATA_PATH}/train.label" \
# --label_valid "${DATA_PATH}/valid.label" \
# --label_test "${DATA_PATH}/test.label" \
# --darr "${DARR}" \
# --train_shrink "1.0" \
# --debug "False"


# HYP_REF="False"
# HYP_SRC="False"
# HYP_SRC_HYP_REF="False"
# HYP_SRC_REF="True"
# # EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_ref"
# # EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_src"
# # EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_src_hyp_ref"
# EXP_NAME="wmt${YEAR}_all_to_all_${MODEL_NAME}_hyp_src_ref"


# for mini_batch in "${BATCH_SIZE[@]}" ; do
#     for opt in "${OPTIMIZER[@]}" ; do
#         for N_TRIAL in `seq "10"` ; do
#             python ${HOME}/scripts/src_ref_hyp_metric/torch_transformers/trainer.py \
#             --exp_name "${EXP_NAME}" \
#             --exp_id "0" \
#             --n_trial "${N_TRIAL}" \
#             --tmp_path "${HOME}/tmp/tmp_log/" \
#             --dump_path "${PROJECT_DISC}/SRHDA/transformers/log/" \
#             --model_name "${MODEL_NAME}" \
#             --langs "${LANGS}" \
#             --empty_dump "False" \
#             --train "True" \
#             --test "False" \
#             --batch_size "${mini_batch}" \
#             --epoch_size "10" \
#             --optimizer "${opt}" \
#             --lr_lambda "0.707" \
#             --dropout "0.0" \
#             --amp "True" \
#             --load_model "False" \
#             --load_model_path "" \
#             --save_model_name "model.pth" \
#             --save_model_path "" \
#             --hyp_ref "${HYP_REF}" \
#             --hyp_src "${HYP_SRC}" \
#             --hyp_src_hyp_ref "${HYP_SRC_HYP_REF}" \
#             --hyp_src_ref "${HYP_SRC_REF}" \
#             --model_path "${MODEL_PATH}" \
#             --src_train "${DATA_PATH}/train.src" \
#             --src_valid  "${DATA_PATH}/valid.src" \
#             --src_test "${DATA_PATH}/test.src" \
#             --ref_train "${DATA_PATH}/train.ref" \
#             --ref_valid "${DATA_PATH}/valid.ref" \
#             --ref_test "${DATA_PATH}/test.ref" \
#             --hyp_train "${DATA_PATH}/train.hyp" \
#             --hyp_valid "${DATA_PATH}/valid.hyp" \
#             --hyp_test "${DATA_PATH}/test.hyp" \
#             --label_train "${DATA_PATH}/train.label" \
#             --label_valid "${DATA_PATH}/valid.label" \
#             --label_test "${DATA_PATH}/test.label" \
#             --darr "${DARR}" \
#             --train_shrink "1.0" \
#             --debug "False"
#         done
#     done
# done

# python ${HOME}/scripts/src_ref_hyp_metric/torch_transformers/trainer.py \
# --exp_name "${EXP_NAME}" \
# --exp_id "0" \
# --n_trial "${N_TRIAL}" \
# --tmp_path "${HOME}/tmp/tmp_log/" \
# --dump_path "${PROJECT_DISC}/SRHDA/transformers/log/" \
# --model_name "${MODEL_NAME}" \
# --langs "${LANGS}" \
# --empty_dump "False" \
# --train "False" \
# --test "True" \
# --batch_size "${mini_batch}" \
# --epoch_size "10" \
# --optimizer "${opt}" \
# --lr_lambda "0.707" \
# --dropout "0.0" \
# --amp "True" \
# --load_model "False" \
# --load_model_path "" \
# --save_model_name "model.pth" \
# --save_model_path "" \
# --hyp_ref "${HYP_REF}" \
# --hyp_src "${HYP_SRC}" \
# --hyp_src_hyp_ref "${HYP_SRC_HYP_REF}" \
# --hyp_src_ref "${HYP_SRC_REF}" \
# --model_path "${MODEL_PATH}" \
# --src_train "${DATA_PATH}/train.src" \
# --src_valid  "${DATA_PATH}/valid.src" \
# --src_test "${DATA_PATH}/test.src" \
# --ref_train "${DATA_PATH}/train.ref" \
# --ref_valid "${DATA_PATH}/valid.ref" \
# --ref_test "${DATA_PATH}/test.ref" \
# --hyp_train "${DATA_PATH}/train.hyp" \
# --hyp_valid "${DATA_PATH}/valid.hyp" \
# --hyp_test "${DATA_PATH}/test.hyp" \
# --label_train "${DATA_PATH}/train.label" \
# --label_valid "${DATA_PATH}/valid.label" \
# --label_test "${DATA_PATH}/test.label" \
# --darr "${DARR}" \
# --train_shrink "1.0" \
# --debug "False"

