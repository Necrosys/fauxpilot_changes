#!/bin/bash

MODEL=${1}
NUM_GPUS=${2}

echo "Converting model ${MODEL} with ${NUM_GPUS} GPUs"

cp -r models/${MODEL}-${NUM_GPUS}gpu /models

if [ $MODEL != "fine-tuned-codegen" ]; then 
    python3 codegen_gptj_convert.py --code_model Salesforce/${MODEL} ${MODEL}-hf
else
    echo "converting the fine-tuned model to GPTJ"
    python3 codegen_gptj_convert.py --code_model ${MODEL} --finetune_dir ./fine-tuned-codegen ${MODEL}-hf
fi
python3 triton_config_gen.py --template config_template.pbtxt --model ${MODEL} --model_store /models --hf_model_dir ${MODEL}-hf  --num_gpu ${NUM_GPUS}
python3 huggingface_gptj_convert.py -in_file ${MODEL}-hf -saved_dir /models/${MODEL}-${NUM_GPUS}gpu/fastertransformer/1 -infer_gpu_num ${NUM_GPUS}
#rm -rf ${MODEL}-hf
