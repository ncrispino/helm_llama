"""Gets datasets from HELM JSON files using helm_process.py
"""
import urllib.request, json
import pandas as pd
from llama.tokenizer import Tokenizer
from helm_process import get_data, get_data_list

def dataset_map():
    urls = ['https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/mmlu:subject=abstract_algebra,method=multiple_choice_joint,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json', 
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/boolq:model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/truthful_qa:task=mc_single,method=multiple_choice_joint,model=microsoft_TNLGv2_530B,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/natural_qa:mode=closedbook,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/natural_qa:mode=openbook_longans,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/quac:model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/commonsense:dataset=hellaswag,method=multiple_choice_separate_original,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/commonsense:dataset=openbookqa,method=multiple_choice_separate_calibrated,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/narrative_qa:model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json'
]
    return dict(zip(range(1, len(urls) + 1), urls))

def get_prompt_map():
    # Only accounts for prompt id.
    prompts = ['', '', '', 'Pay attention to this:', 'You are an attention model.']
    return dict(zip(range(1, len(prompts) + 1), prompts))
    
def get_data_name(url):
    return url.split('v0.2.2/')[1].split(":")[0]

if __name__=="__main__":
    print(dataset_map())

    tokenizer = Tokenizer("weights/tokenizer.model")
    prepend_text = "You are an attention mechanism."
    k = 5
    context_window = 2048


    urls = list(dataset_map().values())

    for data_url in urls:
        df = get_data(data_url)
        input_list_batched = get_data_list(df, prepend_text, k, tokenizer, context_window, num_examples = 5, batch_size = 1)
        data_name = get_data_name(data_url)
        print('data name: ', data_name)
        print('len of data: ', len(input_list_batched))
        with open(f'datasets/{data_name}.json', 'w') as f:
            json.dump(input_list_batched, f)
        # print(input_list_batched)


