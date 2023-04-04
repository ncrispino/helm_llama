"""Given json file for HELM examples, will obtain the non-perturbated inputs and truncate them given the few-shot examples. 

Then, it will place them into batches to be read into by the LLaMA tokenizer.

Note that prepend_text will always have a newline added after.
If user puts in a -k, it will mean they want the tokens in reverse.
"""
import urllib.request, json
import pandas as pd
import numpy as np
from llama.tokenizer import Tokenizer
from ast import literal_eval

def get_data(data_url):
    with urllib.request.urlopen(data_url) as url:
       data = json.load(url)
       data = data["request_states"]
    return pd.json_normalize(data)

def get_helm_data_list(df_file, prepend_text, k, tokenizer, context_window, num_examples=5, batch_size=1, max_gen_len=100, num_instances = 0):
    df = pd.read_csv(df_file, quotechar='"') # Need quotechar for literal_eval to work.
    df = df.query("eval_instance_block != 'eval_instance_block'").copy()
    df["train_instance_blocks"] = df["train_instance_blocks"].apply(literal_eval)
    instructions = df["instructions_block"][0]
    assert instructions != "instructions_block"
    # print("instructions: ", instructions)
    if type(instructions) is not str and np.isnan(instructions):
        instructions = ""
    # print("instructions: ", instructions)
    instance_prefix = df["instance_prefix"][0]
    # print("instance prefix: ", instance_prefix)
    
    few_shot = df["train_instance_blocks"][0]    
    input_list = df["eval_instance_block"].tolist()
    if num_instances > 0:
        input_list = input_list[:num_instances]
    input_list = [truncate_example(prepend_text, k, instructions, text, few_shot, tokenizer, context_window, max_gen_len) for text in input_list]
	
	# Now put into batches
    input_list_batched = [input_list[i: i + batch_size] for i in range(0, len(input_list), batch_size)] 
    
    return input_list_batched

def get_helm_labels(label_file, num_instances):
    """Given label file, will return list of labels. If num_instances > 0, will only return that many labels."""
    df = pd.read_csv(label_file)
    labels = df["isolated_output"].to_list()
    if num_instances > 0:
        labels = labels[:num_instances]
    return dict(zip(range(1, len(labels) + 1), labels))


def get_data_list(df, prepend_text, k, tokenizer, context_window, num_examples=5, batch_size=1, max_gen_len=100, num_instances = 0):
    """Given a pandas df will get all samples from first trial and truncate based on system instruction and k tokens.

    Returns a list of lists, each of length batch_size.
    """
    # Find no perturbations
    no_perturbations = df["instance.perturbation.computed_on"].isna()
    df_no_perturbations = df[no_perturbations]
    # Only select those in valid and test
    df_final = df_no_perturbations[df_no_perturbations["instance.split"] != 'train']
    # As there are 3 in each? I'm just selecting first. Maybe want to see what they change
    df_final = df_final.groupby("instance.id").first()
    input_list = df_final["instance.input.text"].to_list()
    if num_instances > 0:
        input_list = input_list[:num_instances]

    # Get list prompts including few-shot examples.
    beginning_prompt = df_final["request.prompt"][0].split(": ")[0] + ":" # Gets first part of input before ":", e.g. is "Passage:" in narrativeQA.
    few_shot = df_final["request.prompt"].str.split(beginning_prompt)[0][1:num_examples + 1]
    few_shot = [ex.strip() for ex in few_shot]
    # TODO: Add instance ids so each example can be identified (though is probably in correct order so can go 1 by 1?)
    input_list = [truncate_example(prepend_text, k, instructions, text, few_shot, tokenizer, context_window, max_gen_len) for text in input_list]
    
    # Now put into batches
    input_list_batched = [input_list[i: i + batch_size] for i in range(0, len(input_list), batch_size)] 
    
    return input_list_batched

def truncate_example(prepend_text, k, instructions, text, few_shot, tokenizer, context_window, max_gen_len):
        """Given input prompt of text, will truncate by removing few-shot examples one-by-one until they fit the context window of the model.
        
        Same as in HELM.

        """
        current_text = get_full_text(prepend_text, k, instructions, text, few_shot, len(few_shot))
        few_shot_instances = len(few_shot)
        while few_shot_instances > 0:
                if not fits_within_context_window(current_text, context_window, max_gen_len, tokenizer):
                        few_shot_instances -= 1
                        current_text =  get_full_text(prepend_text, k, instructions, text, few_shot, few_shot_instances)
                else:
                        removed_train_instances_count = len(few_shot) - few_shot_instances
                        #if removed_train_instances_count > 0:
                        #        print(
                        #        f"The original constructed prompt exceeded the max context length. "
                        #        f"Removed {removed_train_instances_count} in-context examples to fit "
                        #        f"it within the context window."
                        #        )
                        return current_text     
        return truncate_from_right(current_text, context_window, max_gen_len, tokenizer)

def get_full_text(prepend_text, k, instructions, text, few_shot, few_shot_instances):
        """Given text and few-shot examples, will return the full text. If k < 0 will reverse use k words in reverse instead."""
        prepend_text = prepend_text + "\n" if prepend_text != '' else ''
        k_words = " ".join(text.split()[-abs(k):])
        if k < 0:
                k_words = " ".join(reversed(k_words.split()))
        k_words = k_words + "\n\n" if k != 0 else ''
        instructions = instructions + "\n" if instructions != '' else ''
        return prepend_text + k_words + instructions + "\n".join(few_shot[:few_shot_instances]) + "\n" + text # In their example, each few-shot separated by \n\n

def fits_within_context_window(full_text, context_window, max_gen_len, tokenizer):
        """
        Checks if the given text fits within the context window given by `max_request_length`
        taking to account the expected completion length (defaults to 0).
        """
        # print("checking if beyond context window: ", len(tokenizer.encode(full_text, bos=True, eos=False, max_seq_len=context_window)) + max_gen_len + 1)
        return (
                # TODO: SHOULD I add 1 to the second part?
                len(tokenizer.encode(full_text, bos=True, eos=False, max_seq_len=context_window)) + max_gen_len + 1
                <= context_window
        )

def truncate_from_right(x, context_window, max_gen_len, tokenizer):
        # print("All few-shot examples were removed as the original constructed prompt plus any amount of few-shot examples exceeded the max prompt length.")
        return tokenizer.decode(tokenizer.encode(x, bos=True, eos=False, max_seq_len=context_window + max_gen_len, truncate=True))
        # return tokenizer.encode(x, bos=True, eos=False, max_seq_len=context_window + max_gen_len, truncate=True)            

def isolate_output(prompts, decoded):
		"""Given list of prompts and decoded outputs, will return list of only outputs. 
        Also cuts off end of output after "\n\n" if it exists.
        """
		outputs = []
		for prompt, decode in zip(prompts, decoded):
			output = decode[len(prompt) + 1:] # +1 to remove space after prompt			
			if "\n\n" in output:
				output = output.split("\n\n")[0]
			outputs.append(output)
		return outputs

if __name__ == "__main__":        
        # data_url = "https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/narrative_qa:model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json"
        # df = get_data(data_url)
        tokenizer = Tokenizer("weights/tokenizer.model")
        prepend_text = "You are an attention mechanism."
        k = 0
        context_window = 2048
        data_name = 'natural_qa:mode=openbook_longans,model=huggingface_gpt-j-6b.csv' #'quac:model=huggingface_gpt-j-6b.csv' #'raft:subset=one_stop_english,model=huggingface_gpt-j-6b.csv' #'truthful_qa:task=mc_single,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv' #'summarization_xsum:temperature=0.3,device=cpu,model=huggingface_gpt-j-6b.csv' #'summarization_cnndm:temperature=0.3,device=cpu,model=huggingface_gpt-j-6b.csv' #'boolq:model=huggingface_gpt-j-6b.csv' #'civil_comments:demographic=white,model=huggingface_gpt-j-6b.csv' #'commonsense:dataset=hellaswag,method=multiple_choice_separate_original,model=huggingface_gpt-j-6b.csv' #'commonsense:dataset=openbookqa,method=multiple_choice_separate_calibrated,model=huggingface_gpt-j-6b.csv' #'imdb:model=huggingface_gpt-j-6b.csv' #"mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv" #"msmarco:track=regular,valid_topk=30,model=huggingface_gpt-j-6b.csv" #"msmarco:track=trec,valid_topk=30,model=huggingface_gpt-j-6b.csv" #"narrative_qa:model=huggingface_gpt-j-6b.csv"
        data_file = f"/scratch/cnicholas/helm/dataset/{data_name}"
        input_list_batched = get_helm_data_list(data_file, prepend_text, k, tokenizer, context_window, num_examples = 5, batch_size = 1, num_instances = 2)
        # input_list_batched = get_data_list(df, prepend_text, k, tokenizer, context_window, num_examples = 5, batch_size = 1)
        print(input_list_batched)
