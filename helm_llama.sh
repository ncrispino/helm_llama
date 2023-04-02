#BSUB -N
#BSUB -o logs/llama_test.%J
#BSUB -n 8
#BSUB -m chenguang02
#BSUB -gpu "num=8:gmodel=NVIDIARTXA6000"

export MP=8
export TARGET_FOLDER="weights"
export MODEL_SIZE=7B

torchrun --nproc_per_node $MP defi_llama.py --ckpt_dir $TARGET_FOLDER/$MODEL_SIZE --tokenizer_path $TARGET_FOLDER/tokenizer.model  --max_length 2048 --batch_size 1 --max_new_tokens 100
