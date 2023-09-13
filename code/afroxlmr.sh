#!/bin/bash

mkdir -p output_african

export MAX_LENGTH=164
export BATCH_SIZE=16
export NUM_EPOCHS=20
export SAVE_STEPS=500000
export BERT_MODEL=Davlan/afro-xlmr-large
export dir_name=../data/annotated

for LANG in aeb_Arab ary_Arab arz_Arab afr_Latn aka_Latn amh_Ethi bam_Latn bem_Latn cjk_Latn dik_Latn dyu_Latn ewe_Latn fon_Latn fuv_Latn gaz_Latn hau_Latn ibo_Latn kab_Latn kam_Latn kbp_Latn knc_Arab knc_Latn kik_Latn kin_Latn kmb_Latn kon_Latn lin_Latn lua_Latn lug_Latn luo_Latn mos_Latn nso_Latn nus_Latn nya_Latn plt_Latn run_Latn sag_Latn sna_Latn som_Latn sot_Latn ssw_Latn swh_Latn taq_Latn taq_Tfng tir_Ethi tsn_Latn tso_Latn tum_Latn twi_Latn tzm_Tfng umb_Latn wol_Latn xho_Latn yor_Latn zul_Latn
do
  for j in 1 2 3 4 5
  do
    export SEED=$j
    export OUTPUT_FILE=test_result_${LANG}_$j
    export OUTPUT_PREDICTION=test_predictions_${LANG}_$j
    export DATA_DIR=${dir_name}/${LANG}
    export OUTPUT_DIR=output_african/${LANG}_afroxlmr

    CUDA_VISIBLE_DEVICES=1 python3 train_textclass.py --data_dir $DATA_DIR \
    --model_type xlmroberta \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --output_result $OUTPUT_FILE \
    --output_prediction_file $OUTPUT_PREDICTION \
    --max_seq_length  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate 1e-5 \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --per_gpu_eval_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --gradient_accumulation_steps 2 \
    --labels $DATA_DIR/labels.txt \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir

    rm -rf $OUTPUT_DIR/pytorch_model.bin
    rm -rf $OUTPUT_DIR/sentencepiece.bpe.model
    rm -rf $OUTPUT_DIR/tokenizer.json
    rm -rf $OUTPUT_DIR/tokenizer_config.json
    rm -rf $OUTPUT_DIR/config.json
    rm -rf $OUTPUT_DIR/training_args.bin
    rm -rf $OUTPUT_DIR/special_tokens_map.json
    rm -rf $OUTPUT_DIR/sentencepiece.model

    done
  done
