DATA_PATH=/home/is/kosuke-t/project_disc/SRHDA/WMT15_17_DA_HUME
MODEL_PATH=/home/is/kosuke-t/project_disc/model

python /home/is/kosuke-t/scripts/src_ref_hyp_metric/torch_transformers/trainer.py \
--exp_name "wmt17_all_to_all_xlmr-large" \
--exp_id "0" \
--trial_times "10" \
--tmp_path "/home/ksudoh/kosuke-t/tmp/tmp_log/" \
--dump_path "/home/ksudoh/kosuke-t/data_link/SRHDA/transformers/log/" \
--model_name "xlm-roberta-large" \
--langs "cs-en,de-en,en-ru,en-zh,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en" \
--empty_dump "False" \
--train "True" \
--test "True" \
--batch_size "batch=8" \
--epoch_size "10" \
--optimizer "adam,lr=0.000003" \
--lr_lambda "0.707" \
--dropout "0.0" \
--amp "True" \
--load_model "False" \
--load_model_path "" \
--save_model_name "model.pth" \
--save_model_path "" \
--hyp_ref "False" \
--hyp_src "False" \
--hyp_src_hyp_ref "True" \
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

# lang for WMT19 all-all
#--langs "de-cs,de-en,de-fr,en-cs,en-de,en-fi,en-gu,en-kk,en-lt,en-ru,en-zh,fi-en,fr-de,gu-en,kk-en,lt-en,ru-en,zh-en" \
# lang for WMT19 all-en
#--langs "de-en,fi-en,gu-en,kk-en,lt-en,ru-en,zh-en" \
# lang for WMT19 en-all
#--langs "en-cs,en-de,en-fi,en-gu,en-kk,en-lt,en-ru,en-zh" \

# lang for WMT18 for all-all
#--langs"cs-en,de-en,en-cs,en-de,en-et,en-fi,en-ru,en-tr,en-zh,et-en,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en" \
# lang for WMT18 for all-en
#--langs "cs-en,de-en,et-en,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en" \
# lang for WMT18 for en-all
#--langs "en-cs,en-de,en-et,en-fi,en-ru,en-tr,en-zh" \

# lang for WMT17 for all-all
#--langs "cs-en,de-en,en-ru,en-zh,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en" \
# lang for WMT17 all-en
#--langs "cs-en,de-en,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en" \
