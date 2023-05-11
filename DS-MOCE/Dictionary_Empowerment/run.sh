#!/bin/bash
CHECKPOINT_PATH=./checkpoints
PYTHON=.venv/bin/python3
DICTIONARY=../data/all_fields_concepts.csv
TEMPLATE='[MASK]领域中有很多重要概念，其中[concept]是本节课的重点。'
OUTPUT=../data/output_embeddings.csv 

MPSIZE=1
MAXSEQLEN=512
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=40
TOPP=0

MODEL_TYPE="blocklm-large-chinese"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-position-embeddings 1024 \
            --tokenizer-type ChineseSPTokenizer \
            --fix-command-token \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-large-chinese"



# ${PYTHON} -m torch.distributed.launch --nproc_per_node=$MPSIZE  generate_samples.py \
${PYTHON} generate_field.py \
       --DDP-impl none \
       --model-parallel-size $MPSIZE \
       $MODEL_ARGS \
       --master_port $MASTER_PORT \
       --concept_dictionary $DICTIONARY \
       --prompt_template $TEMPLATE \
       --short_setting \
       --concept_output_embeddings $OUTPUT \
       --fp16 \
       --cache-dir cache \
       --out-seq-length $MAXSEQLEN \
       --seq-length 512 \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP
