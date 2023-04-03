"""Given json file for HELM examples, will obtain the non-perturbated inputs and truncate them given the few-shot examples. 

Then, it will place them into batches to be read into by the LLaMA tokenizer.

Note that prepend_text will always have a newline added after.
If user puts in a -k, it will mean they want the tokens in reverse.
"""
import urllib.request, json
import pandas as pd
from llama.tokenizer import Tokenizer

def get_data(data_url):
    with urllib.request.urlopen(data_url) as url:
       data = json.load(url)
       data = data["request_states"]
    return pd.json_normalize(data)

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
    input_list = [truncate_example(prepend_text, k, text, few_shot, tokenizer, context_window, max_gen_len) for text in input_list]
    
    # Now put into batches
    input_list_batched = [input_list[i: i + batch_size] for i in range(0, len(input_list), batch_size)] 
    
    return input_list_batched

def truncate_example(prepend_text, k, text, few_shot, tokenizer, context_window, max_gen_len):
        """Given input prompt of text, will truncate by removing few-shot examples one-by-one until they fit the context window of the model.
        
        Same as in HELM.

        """
        current_text = get_full_text(prepend_text, k, text, few_shot, len(few_shot))
        few_shot_instances = len(few_shot)
        while few_shot_instances > 0:
                if not fits_within_context_window(current_text, context_window, max_gen_len, tokenizer):
                        few_shot_instances -= 1
                        current_text =  get_full_text(prepend_text, k, text, few_shot, few_shot_instances)
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

def get_full_text(prepend_text, k, text, few_shot, few_shot_instances):
        """Given text and few-shot examples, will return the full text. If k < 0 will reverse use k words in reverse instead."""
        prepend_text = prepend_text + "\n" if prepend_text != '' else ''
        k_words = " ".join(text.split()[-abs(k):])
        if k < 0:
                k_words = " ".join(reversed(k_words.split()))
        k_words = k_words + "\n\n" if k_words != 0 else ''
        return prepend_text + k_words + "\n\n".join(few_shot[:few_shot_instances]) + "\n\n" + text # In their example, each few-shot separated by \n\n

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

if __name__ == "__main__":
        data_url = "https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/narrative_qa:model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json"
        df = get_data(data_url)
        tokenizer = Tokenizer("weights/tokenizer.model")
        prepend_text = "You are an attention mechanism."
        k = 5
        context_window = 2048
        input_list_batched = get_data_list(df, prepend_text, k, tokenizer, context_window, num_examples = 5, batch_size = 1)
        print(input_list_batched)
