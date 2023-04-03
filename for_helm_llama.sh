#BSUB -N
#BSUB -o logs/llama_test.%J
#BSUB -n 1
#BSUB -m chenguang02
#BSUB -gpu "num=1:gmodel=NVIDIARTXA6000"

# We may only be able to run on one GPU due to LLaMA's code (MP supposed to be 1 for 7B model).
export MP=1
export TARGET_FOLDER="weights"
export MODEL_SIZE=7B

for PROMPT_ID in 1 4 6 7 9
do
	torchrun --nproc_per_node $MP helm_llama.py --ckpt_dir $TARGET_FOLDER/$MODEL_SIZE --tokenizer_path $TARGET_FOLDER/tokenizer.model  --max_seq_len 2048 --max_batch_size 1 --temperature 0 --max_new_tokens 100 --data_id 20 --p_id $PROMPT_ID --k 0 --num_instances 100
done

