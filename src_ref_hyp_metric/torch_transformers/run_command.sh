HOME="/home/is/kosuke-t"
PROJECT_DISC="${HOME}/project_disc"
DATA_PATH="${PROJECT_DISC}/data/SRHDA/WMT15_17_DA_HUME"
MODEL_PATH="${PROJECT_DISC}/model"
MODEL_NAME="bert-base-uncased"

# lang for WMT19 all-all
#LANGS="de-cs,de-en,de-fr,en-cs,en-de,en-fi,en-gu,en-kk,en-lt,en-ru,en-zh,fi-en,fr-de,gu-en,kk-en,lt-en,ru-en,zh-en"
# lang for WMT19 all-en
#LANGS="de-en,fi-en,gu-en,kk-en,lt-en,ru-en,zh-en"
# lang for WMT19 en-all
#LANGS="en-cs,en-de,en-fi,en-gu,en-kk,en-lt,en-ru,en-zh"

# lang for WMT18 for all-all
#LANGS="cs-en,de-en,en-cs,en-de,en-et,en-fi,en-ru,en-tr,en-zh,et-en,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en"
# lang for WMT18 for all-en
#LANGS="cs-en,de-en,et-en,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en"
# lang for WMT18 for en-all
#LANGS="en-cs,en-de,en-et,en-fi,en-ru,en-tr,en-zh"

# lang for WMT17 for all-all
#LANGS="cs-en,de-en,en-ru,en-zh,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en"
# lang for WMT17 all-en
LANGS="cs-en,de-en,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en"

python ${HOME}/scripts/src_ref_hyp_metric/torch_transformers/trainer.py \
--exp_name "wmt17_all_to_all_${MODEL_NAME}" \
--exp_id "0" \
--trial_times "2" \
--tmp_path "${HOME}/tmp/tmp_log/" \
--dump_path "${PROJECT_DISC}/SRHDA/transformers/log/" \
--model_name "${MODEL_NAME}" \
--langs "${LANGS}" \
--empty_dump "False" \
--train "True" \
--test "True" \
--batch_size "batch=32" \
--epoch_size "2" \
--optimizer "adam,lr=0.000003" \
--lr_lambda "0.707" \
--dropout "0.0" \
--amp "True" \
--load_model "False" \
--load_model_path "" \
--save_model_name "model.pth" \
--save_model_path "" \
--hyp_ref "True" \
--hyp_src "False" \
--hyp_src_hyp_ref "False" \
--hyp_src_ref "False" \
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
--darr "False" \
--train_shrink "1.0" \
--debug "False"


