export MAX_LENGTH=164
export BATCH_SIZE=16
export NUM_EPOCHS=20
export SAVE_STEPS=500000
export BERT_MODEL=cis-lmu/glot500-base
export dir_name=../data/annotated

for filename in "$dir_name"/*
do
  LANG=${filename:18:24}
  for j in 1 2 3 4 5
  do
    export SEED=$j
    export OUTPUT_FILE=test_result_${LANG}_$j
    export OUTPUT_PREDICTION=test_predictions_${LANG}_$j
    export DATA_DIR=${dir_name}/${LANG}
    export OUTPUT_DIR=outputs/${LANG}_glot

    CUDA_VISIBLE_DEVICES=0 python3 train_textclass.py --data_dir $DATA_DIR \
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
