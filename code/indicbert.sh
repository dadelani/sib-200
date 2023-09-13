#!/bin/bash

mkdir -p output_indic

export MAX_LENGTH=164
export BATCH_SIZE=16
export NUM_EPOCHS=20
export SAVE_STEPS=500000
export BERT_MODEL=ai4bharat/IndicBERTv2-MLM-only
export dir_name=../data/annotated

for LANG in asm_Beng awa_Deva ben_Beng bho_Deva guj_Gujr hin_Deva hne_Deva kan_Knda kas_Arab kas_Deva mag_Deva mai_Deva mal_Mlym mar_Deva mni_Beng npi_Deva ory_Orya pan_Guru san_Deva sat_Olck snd_Arab tam_Taml tel_Telu urd_Arab
do
  for j in 1 2 3 4 5
  do
    export SEED=$j
    export OUTPUT_FILE=test_result_${LANG}_$j
    export OUTPUT_PREDICTION=test_predictions_${LANG}_$j
    export DATA_DIR=${dir_name}/${LANG}
    export OUTPUT_DIR=output_indic/${LANG}_indicbert

    CUDA_VISIBLE_DEVICES=1 python3 train_textclass.py --data_dir $DATA_DIR \
    --model_type bert \
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
