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

def get_data_name(url):
    return url.split('v0.2.2/')[1].split(":")[0]

def helm_dataset_map():
    """Returns a dictionary mapping dataset id to dataset name depending on location in dir. 
    
    ONLY APPEND TO THIS else order may be messed up (unless eval uses our code as well).

    """    
    files = [
        'boolq:model=huggingface_gpt-j-6b.csv',

        # 'civil_comments:demographic=all,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=black,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=christian,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=female,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=LGBTQ,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=male,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=muslim,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=other_religions,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=white,model=huggingface_gpt-j-6b.csv',

        'commonsense:dataset=hellaswag,method=multiple_choice_separate_original,model=huggingface_gpt-j-6b.csv',

        'commonsense:dataset=openbookqa,method=multiple_choice_separate_calibrated,model=huggingface_gpt-j-6b.csv',

        'imdb:model=huggingface_gpt-j-6b.csv',

        'mmlu:subject=abstract_algebra,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv',
        'mmlu:subject=college_chemistry,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv',
        'mmlu:subject=computer_security,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv',
        'mmlu:subject=econometrics,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv',
        'mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv',

        'msmarco:track=regular,valid_topk=30,model=huggingface_gpt-j-6b.csv',
        'msmarco:track=trec,valid_topk=30,model=huggingface_gpt-j-6b.csv',

        'narrative_qa:model=huggingface_gpt-j-6b.csv',

        'natural_qa:mode=closedbook,model=huggingface_gpt-j-6b.csv',
        'natural_qa:mode=openbook_longans,model=huggingface_gpt-j-6b.csv',

        'quac:model=huggingface_gpt-j-6b.csv',

        'raft:subset=ade_corpus_v2,model=huggingface_gpt-j-6b.csv',
        'raft:subset=banking_77,model=huggingface_gpt-j-6b.csv',
        'raft:subset=neurips_impact_statement_risks,model=huggingface_gpt-j-6b.csv',
        'raft:subset=one_stop_english,model=huggingface_gpt-j-6b.csv',
        'raft:subset=overruling,model=huggingface_gpt-j-6b.csv',
        'raft:subset=semiconductor_org_types,model=huggingface_gpt-j-6b.csv',
        'raft:subset=systematic_review_inclusion,model=huggingface_gpt-j-6b.csv',
        'raft:subset=tai_safety_research,model=huggingface_gpt-j-6b.csv',
        'raft:subset=terms_of_service,model=huggingface_gpt-j-6b.csv',
        'raft:subset=tweet_eval_hate,model=huggingface_gpt-j-6b.csv',
        'raft:subset=twitter_complaints,model=huggingface_gpt-j-6b.csv',

        'summarization_cnndm:temperature=0.3,device=cpu,model=huggingface_gpt-j-6b.csv',
        'summarization_xsum:temperature=0.3,device=cpu,model=huggingface_gpt-j-6b.csv',
        
        'truthful_qa:task=mc_single,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv'
    ]
    return dict(zip(range(1, len(files) + 1), files))

def get_helm_data_name(file):    
    return file.split(':')[0]


def get_prompt_map():
    # Only accounts for prompt id.
    prompts = ['', '', '', 'Pay attention to this:', 'You are an attention model.']
    return dict(zip(range(1, len(prompts) + 1), prompts))

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


