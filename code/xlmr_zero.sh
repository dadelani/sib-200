#!/bin/bash

mkdir -p output_zero

export MAX_LENGTH=164
export BATCH_SIZE=16
export NUM_EPOCHS=20
export SAVE_STEPS=500000
export BERT_MODEL=xlm-roberta-large
export dir_name=../data/annotated
export SRC_LANG=eng_Latn

for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
  export SEED=$j
  export OUTPUT_FILE=test_result_${SRC_LANG}_$j
  export OUTPUT_PREDICTION=test_predictions_${SRC_LANG}_$j
  export SRC_DATA_DIR=${dir_name}/${SRC_LANG}
  export OUTPUT_DIR=output_zero/${SRC_LANG}_xlmr

  CUDA_VISIBLE_DEVICES=1 python3 train_textclass.py --data_dir $SRC_DATA_DIR \
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
  --labels $SRC_DATA_DIR/labels.txt \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_output_dir

  for filename in "$dir_name"/*
  do
    export SEED=$j
    export LANG=${filename:18:24}
    export OUTPUT_FILE=test_result_${LANG}_$j
    export OUTPUT_PREDICTION=test_predictions_${LANG}_$j
    export DATA_DIR=${dir_name}/${LANG}
    export OUTPUT_DIR=output_zero/${SRC_LANG}_xlmr

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
    --do_predict \
    --overwrite_output_dir
    done

  rm -rf $OUTPUT_DIR/pytorch_model.bin
  rm -rf $OUTPUT_DIR/sentencepiece.bpe.model
  rm -rf $OUTPUT_DIR/tokenizer.json
  rm -rf $OUTPUT_DIR/tokenizer_config.json
  rm -rf $OUTPUT_DIR/config.json
  rm -rf $OUTPUT_DIR/training_args.bin
  rm -rf $OUTPUT_DIR/special_tokens_map.json
  rm -rf $OUTPUT_DIR/sentencepiece.model

  done
