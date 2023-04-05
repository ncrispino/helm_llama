# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
from helm_process import get_data, get_helm_data_list, isolate_output

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from get_datasets import helm_dataset_map, get_helm_data_name, get_prompt_map


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0, # may need to set differently by dataset
    top_p: float = 1, # previously 0.95
    max_seq_len: int = 2048,
    max_batch_size: int = 1,
    prepend_text: str = "",
    k: int = 5,
    num_examples: int = 5,
    max_new_tokens: int = 100, # TODO: change this to be a function of the dataset
    data_id: int = 0,
    p_id: int = 0,
    num_instances: int = 0 # how many trials to run
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    tokenizer = Tokenizer(tokenizer_path)

    # get dataset based on id
    dataset_file = helm_dataset_map()[data_id]

    # get prompt based on prompt id
    prepend_text = get_prompt_map()[p_id]

    dataset_path = "/storage1/chenguangwang/Active/llama_system/dataset/{dataset_file}"
    print("dataset path: ", dataset_path)
    input_list_batched = get_helm_data_list(f"/storage1/chenguangwang/Active/llama_system/dataset/{dataset_file}", prepend_text, k, tokenizer, max_seq_len, num_examples = num_examples, batch_size = max_batch_size, num_instances = num_instances)
    data_name = get_helm_data_name(dataset_file)
    print('data name: ', data_name)
    print('len of data: ', len(input_list_batched))    
    output_list = []
    i = 0
    for input_list, instance_id in input_list_batched:
        i += len(input_list)
        if i % 5 == 0:
            print("i: ", i)
        prompts = input_list
        results = generator.generate(
            prompts, max_gen_len=max_new_tokens, temperature=temperature, top_p=top_p
        )

        outputs = isolate_output(prompts, results)
        output_list.append(outputs)
        # for result in results:        
        #    print(result)
        #    print("\n==================================\n")
    output_dict = dict(zip(range(1, len(output_list) + 1), output_list))

    output_dir = "/storage1/chenguangwang/Active/llama_system/output"
    # output_dir = "output"
    with open(f'{output_dir}/{data_id}_{p_id}_{k}_samples{num_instances}.json', 'w') as f:
        json.dump(output_dict, f)
    # return output_list


if __name__ == "__main__":
    fire.Fire(main)
